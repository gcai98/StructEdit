"""padding 实验 x 5 seeds, 判断 A 组的单调下降是真信号还是 n=1 噪声。"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import paths, torch, json
from PIL import Image
from eval.metrics import RefFidelity, bg_psnr

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
D, OUT = f"{ROOT}/data/padding_exp", f"{ROOT}/results/padding_seeds"
os.makedirs(OUT, exist_ok=True)

cases = json.load(open(f"{D}/cases.json"))["cases"]
ref1 = Image.open(f"{D}/ref1.png").convert("RGB")
src = Image.open(f"{D}/source.png").convert("RGB")
ref2_orig = Image.open(f"{D}/ref2_pad0.0.png").convert("RGB")

PROMPT = ("characters in Figure 1 walking on the sunset street in Figure 2, "
          "with their backs facing the camera, anime style.")

pipe = paths.load_orion_pipeline(dtype=torch.bfloat16, device=torch.device("cuda"))
pipe.vae.enable_tiling(); pipe.vae.enable_slicing()
fid = RefFidelity(device="cuda")

rows = []
for seed in [0, 1, 2, 3, 4]:
    for c in cases:
        ref2 = Image.open(f"{D}/{c['ref2_file']}").convert("RGB")
        g = torch.Generator(device="cuda").manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            img = pipe(prompt=PROMPT, reference_image=[ref1, ref2], source_image=src,
                       num_inference_steps=30, true_cfg_scale=4.0, negative_prompt=" ",
                       guidance_scale=1.0, generator=g).images[0]
        img.save(f"{OUT}/s{seed}_pad{c['pad']}.png")
        r = {"seed": seed, "pad": c["pad"], "ref1_area": c["area_frac"][0],
             "ref1_dino": fid.score(ref1, img)["dino"],
             "ref2_dino": fid.score(ref2_orig, img)["dino"],
             "bg_psnr": bg_psnr(src, img)}
        rows.append(r)
        print(json.dumps(r), flush=True)
        json.dump(rows, open(f"{OUT}/results.json", "w"), indent=2)

print("\n=== mean over seeds ===")
for pad in sorted({r["pad"] for r in rows}):
    sub = [r for r in rows if r["pad"] == pad]
    m1 = sum(r["ref1_dino"] for r in sub) / len(sub)
    m2 = sum(r["ref2_dino"] for r in sub) / len(sub)
    s1 = (sum((r["ref1_dino"] - m1) ** 2 for r in sub) / len(sub)) ** 0.5
    print(f"pad={pad:<5} ref1 {m1:.4f} +-{s1:.4f}  ref2 {m2:.4f}  (n={len(sub)})")
