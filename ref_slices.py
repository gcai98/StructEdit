"""OrionEdit 多 reference 预合成的坐标映射。纯标准库。

重要: latent 网格不能用 原图尺寸//16 推算。pipeline 内部用
calculate_dimensions(VAE_IMAGE_SIZE, ar) 把所有图重采样到统一像素预算,
保持宽高比。实测 example3 (src 1024x768) -> L_ref=4144 = 74x56。
grid_w/grid_h 必须由调用方从运行时传入。
"""
import math


def ref_boxes_on_canvas(ref_sizes, canvas_size):
    """复刻 _build_composited_reference_canvas (pipeline_orion_edit.py L203-232)。
    ref_sizes: [(w,h),...] 原始尺寸; canvas_size: (W,H) = source.size
    返回每张 ref 在 canvas 上的像素框 [(x0,y0,x1,y1),...]"""
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


def grid_from_L(L, aspect_ratio):
    """由 token 数和宽高比反推网格 (gw, gh)。gw*gh == L, gw/gh ≈ aspect_ratio。
    实测 L=4144, ar=1024/768 -> (74, 56)。"""
    best = None
    for gh in range(1, int(math.isqrt(L)) + 1):
        if L % gh:
            continue
        gw = L // gh
        err = abs(gw / gh - aspect_ratio)
        if best is None or err < best[0]:
            best = (err, gw, gh)
        # 也考虑反过来的因数对
        gw2, gh2 = gh, gw
        err2 = abs(gw2 / gh2 - aspect_ratio)
        if err2 < best[0]:
            best = (err2, gw2, gh2)
    return best[1], best[2]


def box_to_token_rows(box, canvas_w, canvas_h, grid_w, grid_h, offset=0):
    """像素框 -> 行优先 token 索引区间 [(start,stop),...]。
    _pack_latents: 2x2 patchify + 行优先 flatten, t = offset + row*grid_w + col。
    grid_w/grid_h 必须是运行时的实际网格 (见模块 docstring)。"""
    x0, y0, x1, y1 = box
    c0 = max(0, int(math.floor(x0 / canvas_w * grid_w)))
    c1 = min(grid_w, int(math.ceil(x1 / canvas_w * grid_w)))
    r0 = max(0, int(math.floor(y0 / canvas_h * grid_h)))
    r1 = min(grid_h, int(math.ceil(y1 / canvas_h * grid_h)))
    return [(offset + r * grid_w + c0, offset + r * grid_w + c1) for r in range(r0, r1)]


if __name__ == "__main__":
    # 用实测值验证: example3, ref 576x1024 x2, src 1024x768, L_ref=4144, ref 段起点 4144
    refs, canvas, L_ref, ref_offset = [(576, 1024)] * 2, (1024, 768), 4144, 4144
    gw, gh = grid_from_L(L_ref, canvas[0] / canvas[1])
    print(f"L_ref={L_ref} ar={canvas[0]/canvas[1]:.3f} -> grid {gw}x{gh} (期望 74x56)")

    boxes = ref_boxes_on_canvas(refs, canvas)
    print("canvas boxes:", [tuple(round(v, 1) for v in b) for b in boxes])
    for i, b in enumerate(boxes):
        seg = box_to_token_rows(b, canvas[0], canvas[1], gw, gh, offset=ref_offset)
        n = sum(s - t for t, s in seg)
        print(f"  ref{i}: {len(seg)} 行, {n}/{L_ref} tokens ({n/L_ref:.3f}), "
              f"首段 {seg[0]}, 末段 {seg[-1]}")
