"""抓 build_image_stream_slices 的运行时参数, 验证 ref token 区间。只跑 1 步。"""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import paths, torch
from PIL import Image
import models.transformer_orion as T

captured = []
_orig = T.build_image_stream_slices
def spy(L_syn, L_ref, L_src, start_offset=0):
    r = _orig(L_syn, L_ref, L_src, start_offset)
    if not captured:
        captured.append({"L_syn": L_syn, "L_ref": L_ref, "L_src": L_src,
                         "start_offset": start_offset,
                         "slices": {k: [v.start, v.stop] for k, v in r.items()}})
        print("=== build_image_stream_slices ===")
        print(f"L_syn={L_syn} L_ref={L_ref} L_src={L_src} offset={start_offset}")
        print("slices:", {k: (v.start, v.stop) for k, v in r.items()}, flush=True)
    return r
T.build_image_stream_slices = spy

E = paths.ORION_ROOT + "/examples"
refs = [Image.open(f"{E}/example3-ref{i}.png").convert("RGB") for i in (1, 2)]
src = Image.open(f"{E}/example3-source.png").convert("RGB")
print("ref sizes:", [r.size for r in refs], "src:", src.size)

pipe = paths.load_orion_pipeline(dtype=torch.bfloat16, device=torch.device("cuda"))
pipe.vae.enable_tiling(); pipe.vae.enable_slicing()
print("vae_scale_factor =", pipe.vae_scale_factor)

with torch.autocast("cuda", dtype=torch.bfloat16):
    pipe(prompt="two characters walking on the street in Figure 2, anime style.",
         reference_image=refs, source_image=src, num_inference_steps=1,
         true_cfg_scale=4.0, negative_prompt=" ", guidance_scale=1.0)

json.dump(captured, open("results/seg_probe.json", "w"), indent=2)
print("saved results/seg_probe.json")
