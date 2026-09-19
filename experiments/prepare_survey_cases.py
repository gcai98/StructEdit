"""从 OrionEditBench metadata 筛普查样本。
分层: 25 随机(代表真实分布) + 25 双主体(假设最可能触发)。
注意: 每条样本的 reference_image 是**单张预合成 canvas**, 不是多张独立 ref。"""
import json, random, os, argparse

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
ap = argparse.ArgumentParser()
ap.add_argument("--meta", default="/root/autodl-tmp/data/orioneditbench/metadata/metadata.json")
ap.add_argument("--imgroot", default="/root/autodl-tmp/data/obench_ex")
ap.add_argument("--out", default=f"{ROOT}/data/survey/cases.json")
ap.add_argument("--n-random", type=int, default=25)
ap.add_argument("--n-multi", type=int, default=25)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

items = json.load(open(args.meta, encoding="utf-8"))
print(f"metadata 共 {len(items)} 条")

def exists(x):
    return (os.path.exists(os.path.join(args.imgroot, x["source_image"])) and
            os.path.exists(os.path.join(args.imgroot, x["reference_image"])))

avail = [x for x in items if exists(x)]
print(f"图片齐全的: {len(avail)}")
if not avail:
    raise SystemExit("没有可用样本, 检查 --imgroot 下的目录结构")

MULTI = "two most prominent"
multi = [x for x in avail if MULTI in x["edit_prompt"].lower()]
single = [x for x in avail if MULTI not in x["edit_prompt"].lower()]
print(f"双主体 {len(multi)}, 单主体 {len(single)}")

random.seed(args.seed)
picked = ([("random", x) for x in random.sample(avail, min(args.n_random, len(avail)))] +
          [("multi",  x) for x in random.sample(multi, min(args.n_multi, len(multi)))])

seen, cases = set(), []
for group, x in picked:
    if x["img_ids"] in seen: continue
    seen.add(x["img_ids"])
    cases.append({
        "id": f'{group}_{x["img_ids"][:60]}',
        "group": group,
        "refs": [os.path.join(args.imgroot, x["reference_image"])],
        "source": os.path.join(args.imgroot, x["source_image"]),
        "prompt": x["edit_prompt"],
        "gt": os.path.join(args.imgroot, x["output_image"]),   # 可能不存在, 仅记录
        "wh": [x["width"], x["height"]],
        "seed": 0,
    })

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(cases, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
from collections import Counter
print(f"\n写入 {len(cases)} 条 -> {args.out}")
print("分组:", Counter(c["group"] for c in cases))
