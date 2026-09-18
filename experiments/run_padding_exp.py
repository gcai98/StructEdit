"""padding 实验: ref2 内容不变, 只改宽高比, 看两张 ref 的保真度如何变化。
核心待验证: ref1 内容完全没动, 保真度是否也随 ref2 的 padding 下降。"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
import paths, torch, json, os, sys
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from eval.metrics import RefFidelity, bg_psnr
from models.pipeline_orion_edit import OrionEditPipeline

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
D = f"{ROOT}/data/padding_exp"
OUT = f"{ROOT}/results/padding_exp"
os.makedirs(OUT, exist_ok=True)

cases = json.load(open(f"{D}/cases.json"))["cases"]
ref1 = Image.open(f"{D}/ref1.png").convert("RGB")
src = Image.open(f"{D}/source.png").convert("RGB")
ref2_orig = Image.open(f"{D}/ref2_pad0.0.png").convert("RGB")

PROMPTS = {
    "A_ref1_only": "replace the character in Figure 3 with the character in Figure 1.",
    "B_both": "replace the two characters in Figure 3 with the characters in Figure 1 and Figure 2.",
}

pipe = paths.load_orion_pipeline(dtype=torch.bfloat16, device=torch.device("cuda"))
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
print("vae_scale_factor =", pipe.vae_scale_factor)

fid = RefFidelity(device="cuda")
rows = []
for pname, prompt in PROMPTS.items():
    for c in cases:
        ref2 = Image.open(f"{D}/{c['ref2_file']}").convert("RGB")
        torch.manual_seed(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            img = pipe(prompt=prompt, reference_image=[ref1, ref2], source_image=src,
                       num_inference_steps=30, true_cfg_scale=4.0,
                       negative_prompt=" ", guidance_scale=1.0).images[0]
        tag = f"{pname}_pad{c['pad']}"
        img.save(f"{OUT}/{tag}.png")
        row = {
            "prompt": pname, "pad": c["pad"],
            "ref1_area": c["area_frac"][0], "ref2_area": c["area_frac"][1],
            "ref1_fid": fid.score(ref1, img),
            "ref2_fid": fid.score(ref2_orig, img),   # 始终比原图, 不比加了黑边的
            "bg_psnr": bg_psnr(src, img),
            "out_size": img.size,
        }
        rows.append(row)
        print(tag, json.dumps(row["ref1_fid"]), json.dumps(row["ref2_fid"]),
              f"bgPSNR={row['bg_psnr']:.2f}")

json.dump(rows, open(f"{OUT}/results.json", "w"), indent=2, default=str)

print("\n=== 汇总 ===")
print(f"{'prompt':>12} {'pad':>5} {'ref1面积':>8} {'ref1 DINO':>10} {'ref2 DINO':>10} {'bgPSNR':>8}")
for r in rows:
    print(f"{r['prompt']:>12} {r['pad']:>5} {r['ref1_area']:>8.4f} "
          f"{r['ref1_fid']['dino']:>10.4f} {r['ref2_fid']['dino']:>10.4f} {r['bg_psnr']:>8.2f}")



