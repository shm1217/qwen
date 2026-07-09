import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from typing import List
import numpy as np

import rclpy
from rclpy.node import Node
from collections import deque
from cv_bridge import CvBridge

from frontier_ws.msg import BboxImg, EmbArray

from torchreid.reid.utils import FeatureExtractor


class OSNetSimilarity(Node):
    def __init__(self):
        super().__init__("osnet_similarity")

        self.bridge = CvBridge()
        self.robot_id = self.declare_parameter("robot_id", "robot1").value
        self.bbox_image_topic = self.declare_parameter(
            "bbox_image_topic",
            self.scoped_topic("bbox_image"),
        ).value
        self.embedding_topic = self.declare_parameter(
            "embedding_topic",
            self.scoped_topic("embedding"),
        ).value
        self.app_pair_topic = self.declare_parameter(
            "app_pair_topic",
            self.scoped_topic("app_pair"),
        ).value

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # OSNet model
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            device=self.device
        )

        self.sub = self.create_subscription(
            BboxImg,
            self.bbox_image_topic,
            self.img_callback,
            1000
        )

        self.pub = self.create_publisher(
            EmbArray,
            self.embedding_topic,
            1000
        )

        self.app_pair_sub = self.create_subscription(
            BboxImg,
            self.app_pair_topic,
            self.app_pair_callback,
            1000
        )

        self.app_pair_timer = self.create_timer(
            0.01,
            self.app_pair_timer_callback
        )

        self.timer_ = self.create_timer(
            0.01,
            self.timer_callback
        )

        self.msg_queue = deque(maxlen=1000)
        self.app_pair_queue = deque(maxlen=1000)
        self.current_app_frame_id = None
        self.app_frame_items = []

        self.get_logger().info(
            f"[{self.robot_id}] OSNet similarity node started. "
            f"sub={self.bbox_image_topic}, app_pair={self.app_pair_topic}, "
            f"pub={self.embedding_topic}"
        )

    def scoped_topic(self, topic: str) -> str:
        clean = topic.lstrip("/")
        if not self.robot_id:
            return f"/{clean}"
        return f"/{self.robot_id}/{clean}"

    def img_callback(self, msg: BboxImg):
        self.msg_queue.append(msg)

    def ros_images_to_pils(
        self,
        ros_imgs
    ) -> List[PILImage.Image]:

        pil_images = []

        for ros_img in ros_imgs:
            cv_img = self.bridge.imgmsg_to_cv2(
                ros_img,
                desired_encoding="bgr8"
            )

            pil_img = PILImage.fromarray(
                cv_img[:, :, ::-1]
            )

            pil_images.append(pil_img)

        return pil_images

    def get_embedding_batch(self, pil_images: List[PILImage.Image]):

        if len(pil_images) == 0:
            return torch.empty((0, 0))

        np_images = [
            np.array(img.convert("RGB"))
            for img in pil_images
        ]

        feats = self.extractor(np_images)

        feats = F.normalize(feats, p=2, dim=1)

        return feats

    def tensor_to_flat_list(self, emb):

        if emb.numel() == 0:
            return []

        return emb.detach().cpu().reshape(-1).tolist()
    
    def app_pair_callback(self, msg: BboxImg):
        self.app_pair_queue.append(msg)

    def app_pair_timer_callback(self): 
        if len(self.app_pair_queue) == 0: 
            return 
        
        msg = self.app_pair_queue.popleft() 
        det_count = len(msg.det_crops) 
        track_count = len(msg.track_crops) 

        if det_count == 0 or track_count == 0: 
            self.get_logger().warn("app_pair 비어있음") 
            return 
        
        if len(msg.det_ids) != det_count or len(msg.track_ids) != track_count: 
            self.get_logger().warn("app_pair ids length != crops length") 
            return 
        
        det_pils = self.ros_images_to_pils(msg.det_crops) 
        track_pils = self.ros_images_to_pils(msg.track_crops) 
        pair_count = min(det_count, track_count) 

        for i in range(pair_count): 
            det_img = det_pils[i] 
            track_img = track_pils[i] 
            h = max(det_img.height, track_img.height) 
            w = det_img.width + track_img.width 
            new_img = PILImage.new("RGB", (w, h), (0, 0, 0)) 
            new_img.paste(det_img, (0, 0)) 
            new_img.paste(track_img, (det_img.width, 0)) 
            filename = ( 
                f"{msg.frame_id}" 
                f"_det{msg.det_ids[i]}" 
                f"_track{msg.track_ids[i]}" 
                f"_{msg.sim}.png" ) 
            
            # self.get_logger().info(f"sub frame: {msg.frame_id}")
            # new_img.save(filename) 
            # self.get_logger().info(f"saved {filename}")

    def timer_callback(self):

        if len(self.msg_queue) == 0:
            return

        msg = self.msg_queue.popleft()

        det_count = len(msg.det_crops)
        track_count = len(msg.track_crops)

        if det_count == 0 and track_count == 0:
            return

        if (
            len(msg.det_ids) != det_count
            or len(msg.track_ids) != track_count
        ):
            self.get_logger().warn(
                "ids length != crops length"
            )
            return

        det_pils = (
            self.ros_images_to_pils(msg.det_crops)
            if det_count > 0 else []
        )

        track_pils = (
            self.ros_images_to_pils(msg.track_crops)
            if track_count > 0 else []
        )

        det_emb = (
            self.get_embedding_batch(det_pils)
            if det_count > 0
            else torch.empty((0, 0))
        )

        track_emb = (
            self.get_embedding_batch(track_pils)
            if track_count > 0
            else torch.empty((0, 0))
        )

        out = EmbArray()

        out.frame_id = msg.frame_id

        out.det_ids = list(msg.det_ids)

        out.det_dim = (
            int(det_emb.shape[1])
            if det_emb.ndim == 2 and det_emb.shape[0] > 0
            else 0
        )

        out.det_data = self.tensor_to_flat_list(det_emb)

        out.track_ids = list(msg.track_ids)

        out.track_dim = (
            int(track_emb.shape[1])
            if track_emb.ndim == 2 and track_emb.shape[0] > 0
            else 0
        )

        out.track_data = self.tensor_to_flat_list(track_emb)

        self.pub.publish(out)

        # self.get_logger().info(
        #     f"publish emb frame={out.frame_id} "
        #     f"det={len(out.det_ids)} "
        #     f"track={len(out.track_ids)}"
        # )


def main(args=None):

    rclpy.init(args=args)

    node = OSNetSimilarity()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
