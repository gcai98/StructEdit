"""实验 0: 指令能否覆盖 canvas 的位置先验。
canvas 上 ref 顺序固定为 [猴, 女孩] -> 猴永远在左。
P1 要求 "猴左女右" (与 canvas 一致), P2 要求 "女左猴右" (与 canvas 相反)。
若 P2 输出仍是猴左女右, 说明位置由 canvas 决定, 指令无效。
判定靠肉眼, 不用任何模型。"""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import paths, torch
from PIL import Image

OUT = os.path.abspath(os.path.dirname(__file__) + "/../results/exp0")
os.makedirs(OUT, exist_ok=True)

E = paths.ORION_ROOT + "/examples"
ref_monkey = Image.open(f"{E}/example3-ref1.png").convert("RGB")
ref_girl   = Image.open(f"{E}/example3-ref2.png").convert("RGB")
src        = Image.open(f"{E}/example3-source.png").convert("RGB")

# ref 顺序固定, canvas 上猴在左、女孩在右
REFS = [ref_monkey, ref_girl]

BASE = ("{spec}, walking on the sunset street in Figure 2, "
        "with their backs facing the camera, anime style.")
PROMPTS = {
    "P1_monkey_left": BASE.format(
        spec="the monkey from Figure 1 on the left and the girl from Figure 1 on the right"),
    "P2_girl_left": BASE.format(
        spec="the girl from Figure 1 on the left and the monkey from Figure 1 on the right"),
}

pipe = paths.load_orion_pipeline(dtype=torch.bfloat16, device=torch.device("cuda"))
pipe.vae.enable_tiling(); pipe.vae.enable_slicing()

meta = {"canvas_order": "monkey_left_girl_right", "prompts": PROMPTS, "runs": []}
for pname, prompt in PROMPTS.items():
    for seed in range(5):
        g = torch.Generator(device="cuda").manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            img = pipe(prompt=prompt, reference_image=REFS, source_image=src,
                       num_inference_steps=30, true_cfg_scale=4.0,
                       negative_prompt=" ", guidance_scale=1.0, generator=g).images[0]
        fn = f"{pname}_s{seed}.png"
        img.save(f"{OUT}/{fn}")
        meta["runs"].append({"prompt": pname, "seed": seed, "file": fn,
                             "verdict": ""})   # 人工填: follow / violate / unclear
        print(fn, "saved", flush=True)

json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2, ensure_ascii=False)
print(f"\n完成。10 张图在 {OUT}")
print("判定: 看每张图左边是猴还是女孩, 填 meta.json 的 verdict 字段")
print("  P1 组: 猴在左 = follow")
print("  P2 组: 女孩在左 = follow, 猴在左 = violate")
