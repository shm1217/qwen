####### CLIP으로 특징 text 이용해서 대표 이미지 저장하기 
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from typing import List

import rclpy
from rclpy.node import Node
from collections import deque
from cv_bridge import CvBridge
from detect_obs.msg import BboxImg

from transformers import CLIPProcessor, CLIPModel


# -------------------- CLIP --------------------
class ClipSimilarity:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.model.eval()

    def image_text_score(self, pil_images, text):
        if not isinstance(pil_images, list):
            pil_images = [pil_images]

        inputs = self.processor(
            text=[text] * len(pil_images),
            images=pil_images,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        image_emb = F.normalize(outputs.image_embeds, p=2, dim=1)
        text_emb = F.normalize(outputs.text_embeds, p=2, dim=1)

        scores = image_emb @ text_emb.T
        return scores.diag()  # [N]


# -------------------- ROS Node --------------------
class QwenSimilarity(Node):
    def __init__(self):
        super().__init__("clip_similarity")

        self.bridge = CvBridge()
        self.clip_sim = ClipSimilarity()

        self.sub = self.create_subscription(
            BboxImg,
            "/bbox_image",
            self.img_callback,
            500
        )

        self.timer_ = self.create_timer(0.01, self.timer_callback)

        self.msg_queue = deque(maxlen=100)
        
        self.best_score = -1.0
        self.best_img = None

        self.get_logger().info("CLIP similarity node started.")

    def ros_images_to_pils(self, ros_imgs) -> List[PILImage.Image]:
        pil_images = []
        for ros_img in ros_imgs:
            cv_img = self.bridge.imgmsg_to_cv2(ros_img, desired_encoding="bgr8")
            pil_img = PILImage.fromarray(cv_img[:, :, ::-1])  # BGR -> RGB
            pil_images.append(pil_img)
        return pil_images

    def img_callback(self, msg: BboxImg):
        self.msg_queue.append(msg)

    def timer_callback(self):
        if not self.msg_queue:
            return

        msg = self.msg_queue.popleft()

        det_count = len(msg.det_crops)
        if det_count == 0:
            return

        if len(msg.det_ids) != det_count:
            self.get_logger().warn("ids length != crops length")
            return

        det_pils = self.ros_images_to_pils(msg.det_crops)
        if not det_pils:
            return

        text = "a small man wearing green shirt"

        scores = self.clip_sim.image_text_score(det_pils, text)

        cur_best_idx = torch.argmax(scores).item()
        cur_best_score = scores[cur_best_idx].item()  # Tensor → float
        cur_best_img = det_pils[cur_best_idx]

        if cur_best_score > self.best_score:
            self.best_score = cur_best_score
            self.best_img = cur_best_img
            self.best_img.save(f"best_{msg.frame_id}_{self.best_score:.3f}.png")

# -------------------- main --------------------
def main(args=None):
    rclpy.init(args=args)
    node = QwenSimilarity()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

# ##### CLIP 이용해서 text embedding 구해서 저장
# import torch
# import torch.nn.functional as F
# from transformers import CLIPProcessor, CLIPModel

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model_name = "openai/clip-vit-base-patch32"

# model = CLIPModel.from_pretrained(model_name).to(device)
# processor = CLIPProcessor.from_pretrained(model_name)

# model.eval()

# texts = ["white"]

# inputs = processor(
#     text=texts,
#     return_tensors="pt",
#     padding=True
# ).to(device)

# with torch.no_grad():
#     outputs = model.text_model(
#         input_ids=inputs["input_ids"],
#         attention_mask=inputs["attention_mask"]
#     )

#     text_emb = outputs.pooler_output
#     text_emb = model.text_projection(text_emb)

# text_emb = F.normalize(text_emb, p=2, dim=1)

# # txt 저장
# emb = text_emb[0].cpu()

# with open("white_embedding.txt", "w") as f:
#     for v in emb:
#         f.write(f"{v.item()}\n")

# print(text_emb.shape)
# print("saved white_embedding.txt")