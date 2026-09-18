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
