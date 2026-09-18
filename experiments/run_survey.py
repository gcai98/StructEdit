"""失败普查: 在 OrionEditBench 样本上跑 baseline, 不预设失败类型。
只负责出图和记录, 不做任何自动判定。"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import paths, torch
from PIL import Image

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default=f"{ROOT}/data/survey/cases.json")
ap.add_argument("--out", default=f"{ROOT}/results/survey")
ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个, 0=全部")
ap.add_argument("--steps", type=int, default=30)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
cases = json.load(open(args.cases, encoding="utf-8"))
if args.limit:
    cases = cases[:args.limit]
print(f"{len(cases)} cases")

pipe = paths.load_orion_pipeline(dtype=torch.bfloat16, device=torch.device("cuda"))
pipe.vae.enable_tiling(); pipe.vae.enable_slicing()

done_path = f"{args.out}/done.json"
done = json.load(open(done_path)) if os.path.exists(done_path) else {}

for i, c in enumerate(cases):
    cid = c["id"]
    if cid in done:
        print(f"[{i+1}/{len(cases)}] {cid} 跳过 (已完成)"); continue
    try:
        refs = [Image.open(p if os.path.isabs(p) else f"{ROOT}/{p}").convert("RGB")
                for p in c["refs"]]
        src = (Image.open(c["source"] if os.path.isabs(c["source"])
                          else f"{ROOT}/{c['source']}").convert("RGB")
               if c.get("source") else None)
        t0 = time.time()
        g = torch.Generator(device="cuda").manual_seed(c.get("seed", 0))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            img = pipe(prompt=c["prompt"], reference_image=refs, source_image=src,
                       num_inference_steps=args.steps, true_cfg_scale=4.0,
                       negative_prompt=" ", guidance_scale=1.0, generator=g).images[0]
        img.save(f"{args.out}/{cid}.png")
        done[cid] = {"ok": True, "sec": round(time.time()-t0, 1),
                     "out_size": list(img.size), "n_refs": len(refs)}
        print(f"[{i+1}/{len(cases)}] {cid} ok {done[cid]['sec']}s", flush=True)
    except Exception as e:
        done[cid] = {"ok": False, "err": f"{type(e).__name__}: {e}"}
        print(f"[{i+1}/{len(cases)}] {cid} FAILED {e}", flush=True)
    json.dump(done, open(done_path, "w"), indent=2, ensure_ascii=False)

ok = sum(1 for v in done.values() if v.get("ok"))
print(f"\n完成 {ok}/{len(cases)}, 输出在 {args.out}")
