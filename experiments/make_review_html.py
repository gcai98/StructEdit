"""生成审阅页面: ref/source/output 并排, 点按钮打标签, 导出 JSON。
先看 30 张再归类, 所以标签留空, 在页面上自由输入。"""
import os, json, base64, argparse

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
ap = argparse.ArgumentParser()
ap.add_argument("--cases", default=f"{ROOT}/data/survey/cases.json")
ap.add_argument("--results", default=f"{ROOT}/results/survey")
ap.add_argument("--out", default=f"{ROOT}/results/survey/review.html")
ap.add_argument("--max-h", type=int, default=260)
args = ap.parse_args()

def b64(p):
    if not p or not os.path.exists(p): return ""
    ext = os.path.splitext(p)[1].lstrip(".").lower() or "png"
    return f"data:image/{ext};base64," + base64.b64encode(open(p,"rb").read()).decode()

cases = json.load(open(args.cases, encoding="utf-8"))
rows = []
for c in cases:
    out_p = f"{args.results}/{c['id']}.png"
    if not os.path.exists(out_p): continue
    imgs = [(f"ref{i+1}", b64(p if os.path.isabs(p) else f"{ROOT}/{p}"))
            for i, p in enumerate(c["refs"])]
    if c.get("source"):
        imgs.append(("source", b64(c["source"] if os.path.isabs(c["source"])
                                   else f"{ROOT}/{c['source']}")))
    imgs.append(("OUTPUT", b64(out_p)))
    rows.append({"id": c["id"], "prompt": c["prompt"], "imgs": imgs})

cards = []
for r in rows:
    tiles = "".join(
        f'<figure class="{"out" if n=="OUTPUT" else ""}">'
        f'<img src="{d}"><figcaption>{n}</figcaption></figure>' for n, d in r["imgs"])
    cards.append(f'''<section class="card" data-id="{r["id"]}">
  <h3>{r["id"]}</h3>
  <p class="prompt">{r["prompt"]}</p>
  <div class="strip">{tiles}</div>
  <div class="tag">
    <input type="text" placeholder="失败标签(逗号分隔), 空=正常" data-id="{r["id"]}">
    <span class="quick">
      <button>source残留</button><button>ref细节丢失</button>
      <button>身份错配</button><button>主体缺失</button>
      <button>背景改变</button><button>画质差</button>
    </span>
  </div>
  <textarea placeholder="备注" data-id="{r["id"]}"></textarea>
</section>''')

html = f'''<!doctype html><meta charset="utf-8"><title>Survey Review</title>
<style>
body{{font-family:system-ui,"Microsoft YaHei";margin:0;padding:16px;background:#f7f7f8}}
.bar{{position:sticky;top:0;background:#fff;padding:10px 14px;border:1px solid #ddd;
  border-radius:8px;margin-bottom:14px;z-index:9;display:flex;gap:12px;align-items:center}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin-bottom:14px}}
h3{{margin:0 0 4px;font-size:15px}}
.prompt{{margin:0 0 8px;color:#555;font-size:13px;line-height:1.4}}
.strip{{display:flex;gap:8px;overflow-x:auto;align-items:flex-end}}
figure{{margin:0;text-align:center}}
figure img{{max-height:{args.max_h}px;border:1px solid #ccc;border-radius:4px;display:block}}
figure.out img{{border:2px solid #d33}}
figcaption{{font-size:11px;color:#666;margin-top:2px}}
.tag{{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
.tag input{{flex:1;min-width:260px;padding:6px 8px;border:1px solid #bbb;border-radius:4px}}
.quick button{{margin-right:4px;padding:4px 8px;font-size:12px;cursor:pointer;
  border:1px solid #bbb;background:#fafafa;border-radius:4px}}
textarea{{width:100%;margin-top:6px;min-height:36px;padding:6px;border:1px solid #ddd;
  border-radius:4px;font-family:inherit;font-size:13px}}
button.export{{padding:8px 16px;font-size:14px;cursor:pointer}}
</style>
<div class="bar">
  <strong>失败普查审阅</strong>
  <span>共 {len(rows)} 条</span>
  <span style="color:#888;font-size:13px">先看 30 张再归类; 标签可自由输入</span>
  <button class="export" onclick="exportJSON()">导出 JSON</button>
</div>
{"".join(cards)}
<script>
document.querySelectorAll(".quick button").forEach(b=>b.onclick=()=>{{
  const inp=b.closest(".tag").querySelector("input");
  const cur=inp.value.split(",").map(s=>s.trim()).filter(Boolean);
  const t=b.textContent;
  inp.value=(cur.includes(t)?cur.filter(x=>x!==t):cur.concat(t)).join(", ");
}});
function exportJSON(){{
  const out=[...document.querySelectorAll(".card")].map(c=>({{
    id:c.dataset.id,
    tags:c.querySelector("input").value.split(",").map(s=>s.trim()).filter(Boolean),
    note:c.querySelector("textarea").value.trim()
  }}));
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{{type:"application/json"}}));
  a.download="survey_labels.json"; a.click();
}}
</script>'''

open(args.out, "w", encoding="utf-8").write(html)
print("saved", args.out, f"({len(rows)} cards, {os.path.getsize(args.out)/1e6:.1f} MB)")
