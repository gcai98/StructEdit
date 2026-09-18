"""打印 build_image_stream_slices 的运行时 seg, 验证 ref token 区间。
只跑 1 步, 不出图。"""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import paths, torch
from PIL import Image
import models.pipeline_tools as PT

captured = []
_orig = PT.build_image_stream_slices
def spy(*a, **kw):
    r = _orig(*a, **kw)
    if not captured:
        captured.append({"args": [str(x) for x in a], "result": str(r)})
        print("=== build_image_stream_slices ===")
        print("args:", a, kw)
        print("result:", r, flush=True)
    return r
PT.build_image_stream_slices = spy
import models.transformer_orion as T
if hasattr(T, "build_image_stream_slices"):
    T.build_image_stream_slices = spy

E = paths.ORION_ROOT + "/examples"
refs = [Image.open(f"{E}/example3-ref{i}.png").convert("RGB") for i in (1, 2)]
src = Image.open(f"{E}/example3-source.png").convert("RGB")

pipe = paths.load_orion_pipeline(dtype=torch.bfloat16, device=torch.device("cuda"))
pipe.vae.enable_tiling(); pipe.vae.enable_slicing()
print("vae_scale_factor =", pipe.vae_scale_factor)

with torch.autocast("cuda", dtype=torch.bfloat16):
    pipe(prompt="two characters walking on the street in Figure 2, anime style.",
         reference_image=refs, source_image=src,
         num_inference_steps=1, true_cfg_scale=4.0,
         negative_prompt=" ", guidance_scale=1.0)

OUT = os.path.abspath(os.path.dirname(__file__) + "/../results")
json.dump(captured, open(f"{OUT}/seg_probe.json", "w"), indent=2)
print("saved results/seg_probe.json")
