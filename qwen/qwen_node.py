import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from PIL import ImageDraw
from dataclasses import dataclass
from typing import Optional, List

import rclpy
from rclpy.node import Node
from collections import deque
from cv_bridge import CvBridge
from detect_obs.msg import BboxImg, EmbArray
from scipy.optimize import linear_sum_assignment

from transformers import CLIPProcessor, CLIPModel


class QwenSimilarity(Node):
    def __init__(self):
        super().__init__("qwen_similarity")

        self.bridge = CvBridge()

        self.model_name = "openai/clip-vit-base-patch32"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)

        self.model.eval()

        self.sub = self.create_subscription(
            BboxImg,
            "/bbox_image",
            self.img_callback,
            1000
        )

        self.timer_ = self.create_timer(0.01, self.timer_callback)

        self.pub = self.create_publisher(
            EmbArray,
            "/embedding",
            1000
        )

        self.app_pair_sub = self.create_subscription(
            BboxImg,
            "/app_pair",
            self.app_pair_callback,
            1000
        )

        self.app_pair_timer = self.create_timer(
            0.01,
            self.app_pair_timer_callback
        )

        self.msg_queue = deque(maxlen=1000)
        self.app_pair_queue = deque(maxlen=1000)
        self.current_app_frame_id = None
        self.app_frame_items = []

        self.get_logger().info("Qwen similarity node started.")

    def ros_images_to_pils(self, ros_imgs) -> List[PILImage.Image]:
        pil_images: List[PILImage.Image] = []

        for ros_img in ros_imgs:
            cv_img = self.bridge.imgmsg_to_cv2(ros_img, desired_encoding="bgr8")
            pil_img = PILImage.fromarray(cv_img[:, :, ::-1])  # BGR -> RGB
            pil_images.append(pil_img)

        return pil_images

    # def get_embedding_batch(
    #     self,
    #     pil_images: List[PILImage.Image]
    # ) -> torch.Tensor:
    #     if len(pil_images) == 0:
    #         return torch.empty((0, 0), dtype=torch.float32)

    #     inputs = self.processor(
    #         images=pil_images,
    #         return_tensors="pt",
    #         padding=True
    #     ).to(self.device)

    #     with torch.no_grad():
    #         emb = self.model.get_image_features(**inputs)

    #     emb = F.normalize(emb, p=2, dim=1)

    #     return emb

    def get_embedding_batch(
        self,
        pil_images: List[PILImage.Image]
    ) -> torch.Tensor:

        if len(pil_images) == 0:
            return torch.empty((0, 0), dtype=torch.float32)

        inputs = self.processor(
            images=pil_images,
            return_tensors="pt",
            padding=True
        )

        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=pixel_values,
                return_dict=True
            )

            pooled = vision_outputs.pooler_output

            emb = self.model.visual_projection(pooled)

        if not torch.is_tensor(emb):
            self.get_logger().error(f"emb is not tensor: {type(emb)}")
            return torch.empty((0, 0), dtype=torch.float32)

        emb = F.normalize(emb, p=2, dim=1)

        return emb

    def tensor_to_flat_list(self, emb: torch.Tensor) -> List[float]:
        if emb.numel() == 0:
            return []
        return emb.detach().cpu().reshape(-1).tolist()

    def img_callback(self, msg: BboxImg):
        self.msg_queue.append(msg)
                 
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

    # def app_pair_timer_callback(self):
    #         if len(self.app_pair_queue) == 0:
    #             return

    #         msg = self.app_pair_queue.popleft()

    #         det_count = len(msg.det_crops)
    #         track_count = len(msg.track_crops)

    #         if det_count == 0 or track_count == 0:
    #             self.get_logger().warn("app_pair 비어있음")
    #             return

    #         if len(msg.det_ids) != det_count or len(msg.track_ids) != track_count:
    #             self.get_logger().warn("app_pair ids length != crops length")
    #             return

    #         # frame이 바뀌면 이전 frame 저장
    #         if self.current_app_frame_id is not None and msg.frame_id != self.current_app_frame_id:
    #             self.save_app_frame_items()
    #             self.app_frame_items = []

    #         self.current_app_frame_id = msg.frame_id

    #         det_pils = self.ros_images_to_pils(msg.det_crops)
    #         track_pils = self.ros_images_to_pils(msg.track_crops)

    #         pair_count = min(det_count, track_count)

    #         for i in range(pair_count):
    #             det_img = det_pils[i]
    #             track_img = track_pils[i]

    #             h = max(det_img.height, track_img.height)
    #             w = det_img.width + track_img.width

    #             new_img = PILImage.new("RGB", (w, h), (0, 0, 0))
    #             new_img.paste(det_img, (0, 0))
    #             new_img.paste(track_img, (det_img.width, 0))

    #             self.app_frame_items.append({
    #                 "frame_id": msg.frame_id,
    #                 "det_id": msg.det_ids[i],
    #                 "track_id": msg.track_ids[i],
    #                 "sim": float(msg.sim),
    #                 "img": new_img
    #             })
        
    # def save_app_frame_items(self):
    #     if len(self.app_frame_items) == 0:
    #         return

    #     best_indices_by_det = {}

    #     for idx, item in enumerate(self.app_frame_items):
    #         det_id = item["det_id"]

    #         if det_id not in best_indices_by_det:
    #             best_indices_by_det[det_id] = idx
    #         else:
    #             prev_idx = best_indices_by_det[det_id]
    #             if item["sim"] > self.app_frame_items[prev_idx]["sim"]:
    #                 best_indices_by_det[det_id] = idx

    #     best_indices = set(best_indices_by_det.values())

    #     for idx, item in enumerate(self.app_frame_items):
    #         new_img = item["img"]

    #         if idx in best_indices:
    #             draw = ImageDraw.Draw(new_img)
    #             border = 8
    #             for b in range(border):
    #                 draw.rectangle(
    #                     [b, b, new_img.width - 1 - b, new_img.height - 1 - b],
    #                     outline=(255, 0, 0)
    #                 )

    #         filename = (
    #             f"{item['frame_id']}"
    #             f"_det{item['det_id']}"
    #             f"_track{item['track_id']}"
    #             f"_sim{item['sim']:.3f}.png"
    #         )

    #         new_img.save(filename)
    #         self.get_logger().info(f"saved {filename}")
        

    def timer_callback(self):
        if len(self.msg_queue) == 0:
            return

        msg = self.msg_queue.popleft()
        # self.get_logger().info(f"processing frame id: {msg.frame_id}")

        det_count = len(msg.det_crops)
        track_count = len(msg.track_crops)

        if(det_count == 0 and track_count == 0): # msg가 비어있으면
            return
        if len(msg.det_ids) != det_count or len(msg.track_ids) != track_count:
            self.get_logger().warn(
                f"ids length != crops length"
            )
            return

        det_pils = self.ros_images_to_pils(msg.det_crops) if det_count > 0 else []
        track_pils = self.ros_images_to_pils(msg.track_crops) if track_count > 0 else []
                    
        # emb pub
        det_emb = self.get_embedding_batch(det_pils) if det_count > 0 else torch.empty((0, 0))
        track_emb = self.get_embedding_batch(track_pils) if track_count > 0 else torch.empty((0, 0))

        out = EmbArray()
        out.frame_id = msg.frame_id

        out.det_ids = list(msg.det_ids)
        out.det_dim = int(det_emb.shape[1]) if det_emb.ndim == 2 and det_emb.shape[0] > 0 else 0
        out.det_data = self.tensor_to_flat_list(det_emb)

        out.track_ids = list(msg.track_ids)
        out.track_dim = int(track_emb.shape[1]) if track_emb.ndim == 2 and track_emb.shape[0] > 0 else 0
        out.track_data = self.tensor_to_flat_list(track_emb)

        self.pub.publish(out)
        
      
def main(args=None):
    rclpy.init(args=args)
    node = QwenSimilarity()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()