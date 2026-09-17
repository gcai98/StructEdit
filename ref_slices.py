import numpy as np

def ref_boxes_on_canvas(ref_sizes, canvas_size):
    if len(ref_sizes) == 1:
        W, H = ref_sizes[0]
        return [(0, 0, W, H)]
    strip_w = sum(w for w, h in ref_sizes)
    strip_h = max(h for w, h in ref_sizes)
    cw, ch = canvas_size
    scale = min(cw / strip_w, ch / strip_h)
    new_w = max(1, int(round(strip_w * scale)))
    new_h = max(1, int(round(strip_h * scale)))
    px, py = (cw - new_w) // 2, (ch - new_h) // 2
    boxes, x = [], 0
    for w, h in ref_sizes:
        boxes.append((px + x * scale, py, px + (x + w) * scale, py + h * scale))
        x += w
    return boxes

def box_to_token_rows(box, img_w, img_h, grid_w, grid_h, offset=0):
    x0, y0, x1, y1 = box
    c0 = max(0, int(np.floor(x0 / img_w * grid_w)))
    c1 = min(grid_w, int(np.ceil (x1 / img_w * grid_w)))
    r0 = max(0, int(np.floor(y0 / img_h * grid_h)))
    r1 = min(grid_h, int(np.ceil (y1 / img_h * grid_h)))
    return [(offset + r * grid_w + c0, offset + r * grid_w + c1) for r in range(r0, r1)]

if __name__ == "__main__":
    for refs in [[(512,768),(640,640)], [(512,512),(512,512),(512,512)], [(768,1024),(512,512)]]:
        canvas = (1344, 768)
        bs = ref_boxes_on_canvas(refs, canvas)
        ratios = [(b[2]-b[0])*(b[3]-b[1])/(canvas[0]*canvas[1]) for b in bs]
        print(f"\nrefs={refs} canvas={canvas}")
        print("  面积占比:", [f"{r:.3f}" for r in ratios])
        gw, gh = canvas[0]//16, canvas[1]//16
        for i, b in enumerate(bs):
            seg = box_to_token_rows(b, canvas[0], canvas[1], gw, gh)
            n = sum(s-t for t,s in seg)
            print(f"  ref{i}: {len(seg)} 行, {n}/{gw*gh} tokens ({n/(gw*gh):.3f})")
