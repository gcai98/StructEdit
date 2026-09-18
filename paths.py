"""把 OrionEdit 加入 import 路径。任何脚本开头 import paths 即可。"""
import sys, os
from pathlib import Path

ORION_ROOT = os.environ.get("ORION_ROOT")
if ORION_ROOT is None:
    if os.path.exists("/root/autodl-tmp/OrionEdit"):
        ORION_ROOT = "/root/autodl-tmp/OrionEdit"          # 服务器
    else:
        ORION_ROOT = str(Path(__file__).resolve().parent.parent / "OrionEdit")  # 本地

if not os.path.exists(ORION_ROOT):
    raise RuntimeError(f"找不到 OrionEdit: {ORION_ROOT}，设置环境变量 ORION_ROOT 指定路径")

if ORION_ROOT not in sys.path:
    sys.path.insert(0, ORION_ROOT)

WEIGHTS_BASE = "/root/autodl-tmp/weights/qwen-image-edit-2511"
WEIGHTS_LORA = "/root/autodl-tmp/weights/orionedit"

def load_orion_pipeline(dtype=None, device=None):
    """绕过 from_orion_pretrained 里的 hf_hub_download, 改用本地权重目录。"""
    import os, torch
    import models.pipeline_orion_edit as P

    _orig = P.hf_hub_download
    def _local(repo_id, filename, **kw):
        if os.path.isdir(repo_id):
            p = os.path.join(repo_id, filename)
            if os.path.exists(p):
                return p
            raise FileNotFoundError(p)
        return _orig(repo_id=repo_id, filename=filename, **kw)

    P.hf_hub_download = _local
    try:
        pipe = P.OrionEditPipeline.from_orion_pretrained(
            base_model=WEIGHTS_BASE, orion_repo=WEIGHTS_LORA,
            torch_dtype=dtype, device=device,
        )
    finally:
        P.hf_hub_download = _orig
    return pipe
