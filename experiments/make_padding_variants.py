"""生成 padding 变体: 内容不变, 只改 ref2 的宽高比, 测容量分配对 identity fidelity 的影响。
纯 PIL + numpy, 本地即可跑。"""
import os, json, sys
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from ref_slices import ref_boxes_on_canvas, box_to_token_rows

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(HERE + "/..")
SRC  = os.path.abspath(ROOT + "/../OrionEdit/examples")
OUT  = ROOT + "/data/padding_exp"
os.makedirs(OUT, exist_ok=True)

ref1 = Image.open(f"{SRC}/example3-ref1.png").convert("RGB")
ref2 = Image.open(f"{SRC}/example3-ref2.png").convert("RGB")
src  = Image.open(f"{SRC}/example3-source.png").convert("RGB")
ref1.save(f"{OUT}/ref1.png")
src.save(f"{OUT}/source.png")

cases = []
for pad in [0.0, 0.25, 0.5, 1.0]:
    w, h = ref2.size
    nh = int(round(h * (1 + 2 * pad)))
    canvas = Image.new("RGB", (w, nh), (0, 0, 0))
    canvas.paste(ref2, (0, (nh - h) // 2))
    name = f"ref2_pad{pad}.png"
    canvas.save(f"{OUT}/{name}")

    sizes = [ref1.size, canvas.size]
    boxes = ref_boxes_on_canvas(sizes, src.size)
    areas = [(b[2]-b[0]) * (b[3]-b[1]) / (src.size[0]*src.size[1]) for b in boxes]
    # ref2 框内真正有内容的部分 (去掉自身黑边)
    content_frac = h / nh
    cases.append({
        "pad": pad,
        "ref2_file": name,
        "ref2_size": list(canvas.size),
        "canvas_boxes": [[round(v, 1) for v in b] for b in boxes],
        "area_frac": [round(a, 4) for a in areas],
        "ref2_content_frac_of_own_box": round(content_frac, 4),
        "ref2_effective_area": round(areas[1] * content_frac, 4),
    })

json.dump({"source": "example3", "ref1": "ref1.png", "cases": cases},
          open(f"{OUT}/cases.json", "w"), indent=2)

print(f"{'pad':>5} {'ref2尺寸':>12} {'ref1占比':>9} {'ref2占比':>9} {'ref2有效':>9}")
for c in cases:
    print(f"{c['pad']:>5} {str(tuple(c['ref2_size'])):>12} "
          f"{c['area_frac'][0]:>9.4f} {c['area_frac'][1]:>9.4f} "
          f"{c['ref2_effective_area']:>9.4f}")
