"""GPU 首次冒烟测试: 出一张图 + 打印所有待实测量。一次跑完, 不为看一个数字重开机。"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
import paths, torch, json, os
from PIL import Image
from models.pipeline_orion_edit import OrionEditPipeline

os.makedirs("results", exist_ok=True)
info = {}

print("=== env ===")
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
info["gpu"] = torch.cuda.get_device_name(0)

print("=== loading pipeline ===")
pipe = paths.load_orion_pipeline(dtype=torch.bfloat16, device=torch.device("cuda"))

info["vae_scale_factor"] = int(pipe.vae_scale_factor)
info["latent_channels"] = int(getattr(pipe, "latent_channels", -1))
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
print("vae tiling/slicing enabled")
print("vae_scale_factor =", info["vae_scale_factor"])
print("latent_channels  =", info["latent_channels"])
print("patch grid divisor (vae_scale_factor*2) =", info["vae_scale_factor"] * 2)

E = paths.ORION_ROOT + "/examples"
refs = [Image.open(f"{E}/example3-ref1.png").convert("RGB"),
        Image.open(f"{E}/example3-ref2.png").convert("RGB")]
src = Image.open(f"{E}/example3-source.png").convert("RGB")
info["ref_sizes"] = [r.size for r in refs]
info["src_size"] = src.size
print("ref sizes", info["ref_sizes"], "src", info["src_size"])

print("=== generating ===")
with torch.autocast("cuda", dtype=torch.bfloat16):
    img = pipe(
        prompt="replace the character in Figure 3 with the character in Figure 1.",
        reference_image=refs, source_image=src,
        num_inference_steps=30, true_cfg_scale=4.0,
        negative_prompt=" ", guidance_scale=1.0,
    ).images[0]

img.save("results/smoke_example3.png")
info["out_size"] = img.size
info["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
print("saved results/smoke_example3.png", img.size)
print("peak GPU mem:", info["peak_mem_GB"], "GB")

with open("results/smoke_info.json", "w") as f:
    json.dump(info, f, indent=2, default=str)
print(json.dumps(info, indent=2, default=str))



