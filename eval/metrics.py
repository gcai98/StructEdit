"""参考图保真度评测: DINOv2 + CLIP 图像嵌入余弦相似度。"""
import torch, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel

DINO_PATH = "/root/autodl-tmp/weights/dinov2-base"
CLIP_PATH = "/root/autodl-tmp/weights/clip-vit-large"


class RefFidelity:
    def __init__(self, device="cuda", use_clip=True):
        self.device = device
        self.dino_proc = AutoImageProcessor.from_pretrained(DINO_PATH)
        self.dino = AutoModel.from_pretrained(DINO_PATH).to(device).eval()
        self.use_clip = use_clip
        if use_clip:
            self.clip_proc = CLIPProcessor.from_pretrained(CLIP_PATH)
            self.clip = CLIPModel.from_pretrained(CLIP_PATH).to(device).eval()

    @torch.no_grad()
    def _dino_emb(self, img):
        x = self.dino_proc(images=img, return_tensors="pt").to(self.device)
        out = self.dino(**x).last_hidden_state[:, 0]          # CLS token
        return torch.nn.functional.normalize(out, dim=-1)

    @torch.no_grad()
    def _clip_emb(self, img):
        x = self.clip_proc(images=img, return_tensors="pt").to(self.device)
        out = self.clip.get_image_features(**x)
        return torch.nn.functional.normalize(out, dim=-1)

    def score(self, ref_img, out_img, box=None):
        """box: 输出图上目标区域 (x0,y0,x1,y1); 给了就先裁剪再比。"""
        if box is not None:
            out_img = out_img.crop(tuple(int(v) for v in box))
        r = {"dino": float((self._dino_emb(ref_img) * self._dino_emb(out_img)).sum())}
        if self.use_clip:
            r["clip"] = float((self._clip_emb(ref_img) * self._clip_emb(out_img)).sum())
        return r


def bg_psnr(src_img, out_img, mask=None):
    """背景保持度。out 与 src 尺寸可能不同, 先 resize 到 src。"""
    a = np.asarray(src_img.convert("RGB"), dtype=np.float64)
    b = np.asarray(out_img.convert("RGB").resize(src_img.size), dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask.convert("L").resize(src_img.size)) < 128   # True = 背景
        a, b = a[m], b[m]
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
