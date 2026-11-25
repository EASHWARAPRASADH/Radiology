"""Live training monitor: ``uv run radio monitor RUN_DIR --log TRAIN_LOG``.

Serves a local page that polls the run every few seconds:
* per-epoch curves parsed from the training log lines ``[epoch N] {...}`` (and ``progress.jsonl`` step losses
  when the trainer writes them),
* elapsed time / ETA from ``config.yaml`` (epochs) and observed epoch durations,
* on demand, a CPU preview of the current best detector checkpoint on validation images
  (green = radiologist boxes, red = predicted boxes at score >= 0.3).
Read-only: never touches the training process.
"""

from __future__ import annotations

import ast
import io
import json
import re
import subprocess
import time
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response

_EPOCH = re.compile(r"^\[epoch (\d+)\] (\{.*\})\s*$")
_DONE = re.compile(r"^\[done\] (.*)$")
_STEP = re.compile(r"epoch (\d+) step (\d+)/(\d+)")  # classifier trainer: "  epoch E step S/T loss L"


def parse_log(text: str) -> tuple[list[dict], str | None, str | None]:
    epochs, done, error = [], None, None
    for line in text.splitlines():
        m = _EPOCH.match(line.strip())
        if m:
            try:
                epochs.append(ast.literal_eval(m.group(2)))
            except (ValueError, SyntaxError):
                pass
        d = _DONE.match(line.strip())
        if d:
            done = d.group(1)
        if "Traceback" in line or line.startswith("Error") or "Error:" in line:
            error = line.strip()
    return epochs, done, error


def _running(pattern: str) -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5).stdout
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return False


def status(run_dir: Path, log: Path | None) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text()) if (run_dir / "config.yaml").exists() else {}
    total_epochs = int(cfg.get("epochs") or cfg.get("optim", {}).get("epochs") or 0)
    text = log.read_text() if log and log.exists() else ""
    epochs, done, error = parse_log(text)
    start = (run_dir / "config.yaml").stat().st_mtime if (run_dir / "config.yaml").exists() else time.time()
    now = time.time()
    elapsed = now - start
    durations = [e.get("seconds", 0) for e in epochs]
    avg = sum(durations) / len(durations) if durations else None
    finished_secs = sum(durations)
    eta = None
    current_frac = None
    if avg and total_epochs and not done:
        in_epoch = max(0.0, elapsed - finished_secs)
        current_frac = min(0.99, in_epoch / avg)
        eta = max(0.0, (total_epochs - len(epochs)) * avg - in_epoch)
    # Step-level progress (classifier logs print "epoch E step S/T"): finer progress and ETA within an epoch.
    last_step = None
    for m in _STEP.finditer(text[-20000:]):
        last_step = (int(m.group(2)), int(m.group(3)))
    if last_step and not done:
        s_done, s_total = last_step
        frac_total = s_done / max(1, s_total)
        if total_epochs:
            current_frac = min(0.99, frac_total * total_epochs - len(epochs))
        if frac_total > 0.01:
            eta = max(0.0, elapsed * (1 - frac_total) / frac_total)
    steps = []
    prog = run_dir / "progress.jsonl"
    if prog.exists():
        steps = [json.loads(x) for x in prog.read_text().splitlines()[-2000:] if x.strip()]
    ckpt = run_dir / "model.pt"
    return {
        "run": run_dir.name, "config": cfg, "total_epochs": total_epochs, "epochs": epochs, "steps": steps,
        "elapsed_s": elapsed, "eta_s": eta, "current_epoch_fraction": current_frac, "done": done, "error": error,
        # Alive if a trainer process is running, or the log was written in the last 2 minutes.
        "process_alive": _running("radio train") or bool(log and log.exists() and now - log.stat().st_mtime < 120),
        "checkpoint_mtime": ckpt.stat().st_mtime if ckpt.exists() else None,
    }


def _preview_png(run_dir: Path, n: int = 4, score_thr: float = 0.3) -> bytes:
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    from radio.data.chestx_det import DET_CONCEPTS, load_split, read_gray, split_train_val
    from radio.models.detector import DetectorConfig, build_detector, run_detector

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    torch.set_num_threads(2)  # stay light while training runs
    dcfg = DetectorConfig(concepts=tuple(DET_CONCEPTS), pretrained_backbone=False, min_size=cfg["min_size"],
                          max_size=cfg["min_size"])
    model = build_detector(dcfg)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    _, val = split_train_val(load_split(cfg["root"], "train"), cfg["val_frac"])
    recs = [r for r in val if len(r.boxes)][:n]
    tiles = []
    for r in recs:
        img = read_gray(r.path)
        dets = run_detector(model, [img], DET_CONCEPTS, torch.device("cpu"))[0]
        rgb = Image.fromarray((np.stack([img] * 3, -1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(rgb)
        for b, lab in zip(r.boxes, r.labels, strict=True):
            d.rectangle(list(map(float, b)), outline=(40, 220, 90), width=6)
            d.text((float(b[0]) + 6, float(b[1]) + 4), lab, fill=(40, 220, 90))
        for det in dets:
            if det.score >= score_thr:
                d.rectangle(list(det.box_xyxy), outline=(235, 60, 60), width=5)
                d.text((det.box_xyxy[0] + 6, det.box_xyxy[3] - 14), f"{det.concept} {det.score:.2f}",
                       fill=(235, 60, 60))
        tiles.append(rgb.resize((420, 420)))
    grid = Image.new("RGB", (840, 420 * ((len(tiles) + 1) // 2)), (20, 20, 20))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % 2) * 420, (i // 2) * 420))
    buf = io.BytesIO()
    grid.save(buf, format="PNG")
    return buf.getvalue()


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Radio training monitor</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--fg:#e6e8eb;--muted:#9aa3ad;--accent:#5aa9ff;--ok:#3ecf8e;--warn:#f5b84c;--bad:#ef5b5b}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:18px 24px;border-bottom:1px solid #232733;display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:18px;margin:0}.muted{color:var(--muted)}.pill{padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600}
main{padding:20px 24px;display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.card{background:var(--card);border:1px solid #232733;border-radius:12px;padding:16px}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.big{font-size:28px;font-weight:650}.bar{height:10px;background:#232733;border-radius:6px;overflow:hidden;margin-top:8px}
.bar>div{height:100%;background:var(--accent);transition:width .5s}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:5px 6px;border-bottom:1px solid #232733;text-align:right}
th:first-child,td:first-child{text-align:left}svg{width:100%;height:190px}
button{background:var(--accent);color:#081320;border:0;border-radius:8px;padding:8px 14px;font-weight:650;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}.wide{grid-column:1/-1}img{max-width:100%;border-radius:8px;margin-top:10px}
.legend span{margin-right:14px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
</style></head><body>
<header><h1>Radio · training monitor</h1><span id="run" class="muted"></span><span id="state" class="pill"></span>
<span class="muted" id="updated"></span></header>
<main>
<section class="card"><h2>Progress</h2><div class="big" id="epochs">–</div>
<div class="bar"><div id="pbar" style="width:0%"></div></div>
<p class="muted" id="times"></p></section>
<section class="card"><h2>Latest epoch</h2><div class="big" id="map">–</div><p class="muted" id="latest"></p></section>
<section class="card"><h2 id="c1title">Validation metric</h2><svg id="c1"></svg>
<div class="legend muted" id="c1legend"></div></section>
<section class="card"><h2>Training loss per epoch</h2><svg id="c2"></svg></section>
<section class="card wide"><h2>Epochs</h2><table id="tbl"></table></section>
<section class="card wide"><h2>Live predictions on validation images (best checkpoint, CPU)</h2>
<p class="muted">Green = radiologist boxes · red = detector boxes with score ≥ 0.30. Takes ~20–40 s; runs on CPU so training is not slowed much.</p>
<button id="pv" onclick="preview()">Render predictions</button><span class="muted" id="pvmsg"></span><div id="pvimg"></div></section>
</main>
<script>
const fmt=s=>s==null?"–":(s>=3600?`${Math.floor(s/3600)}h ${Math.round(s%3600/60)}m`:`${Math.round(s/60)}m`);
function line(svg,series,colors,ymin,ymax){
  const W=svg.clientWidth||400,H=190,p=30;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  const n=Math.max(2,...series.map(s=>s.length));let g='';
  for(let i=0;i<=4;i++){const y=p+(H-2*p)*i/4,v=(ymax-(ymax-ymin)*i/4).toFixed(2);
    g+=`<line x1="${p}" x2="${W-8}" y1="${y}" y2="${y}" stroke="#232733"/><text x="2" y="${y+4}" fill="#9aa3ad" font-size="10">${v}</text>`;}
  series.forEach((s,k)=>{if(!s.length)return;const pts=s.map((v,i)=>[p+(W-p-8)*i/(n-1),p+(H-2*p)*(1-(v-ymin)/((ymax-ymin)||1))]);
    g+=`<polyline fill="none" stroke="${colors[k]}" stroke-width="2.5" points="${pts.map(q=>q.join(',')).join(' ')}"/>`;
    pts.forEach(q=>g+=`<circle cx="${q[0]}" cy="${q[1]}" r="3.5" fill="${colors[k]}"/>`);});
  for(let i=0;i<n;i++)g+=`<text x="${p+(W-p-8)*i/(n-1)-3}" y="${H-8}" fill="#9aa3ad" font-size="10">${i+1}</text>`;
  svg.innerHTML=g;}
async function tick(){
  const s=await (await fetch('api/status')).json();
  document.getElementById('run').textContent=s.run;
  const st=document.getElementById('state');
  if(s.error){st.textContent='error';st.style.background='var(--bad)';}
  else if(s.done){st.textContent='finished';st.style.background='var(--ok)';}
  else if(s.process_alive){st.textContent='training';st.style.background='var(--accent)';}
  else{st.textContent='not running';st.style.background='var(--warn)';}
  st.style.color='#081320';
  const e=s.epochs,done=e.length,tot=s.total_epochs||done;
  const frac=s.done?1:(done+(s.current_epoch_fraction||0))/(tot||1);
  document.getElementById('epochs').textContent=`Epoch ${Math.min(done+(s.done?0:1),tot)} / ${tot}`;
  document.getElementById('pbar').style.width=(100*frac).toFixed(1)+'%';
  document.getElementById('times').textContent=`Elapsed ${fmt(s.elapsed_s)} · ETA ${s.done?'done':fmt(s.eta_s)}`+
    (e.length?` · ~${fmt(e.reduce((a,x)=>a+x.seconds,0)/e.length)} per epoch`:'');
  if(e.length){const L=e[e.length-1];
    document.getElementById('map').textContent=(L.val_map25??L.val_mean_auroc??0).toFixed(3);
    document.getElementById('latest').textContent=('val_map25' in L)?`val mAP@0.25 · sens@1FP ${L['val_mean_sens@1fp'].toFixed(3)} · loss ${L.train_loss.toFixed(3)}`:`val mean AUROC · loss ${L.train_loss.toFixed(4)}`;}
  const det=e.length?('val_map25' in e[0]):!!(s.config&&s.config.root);
  document.getElementById('c1title').textContent=det?'Validation mAP@0.25 and sensitivity at 1 FP/image':'Validation mean AUROC';
  document.getElementById('c1legend').innerHTML=det?'<span><i class="sw" style="background:#5aa9ff"></i>mAP@0.25</span><span><i class="sw" style="background:#3ecf8e"></i>Sens@1FP</span>':'<span><i class="sw" style="background:#5aa9ff"></i>mean AUROC (radiologist labels where available)</span>';
  document.querySelector('#pv').closest('section').style.display=det?'':'none';
  const m=e.map(x=>x.val_map25??x.val_mean_auroc),sn=e.map(x=>x['val_mean_sens@1fp']).filter(v=>v!=null),lo=e.map(x=>x.train_loss);
  line(document.getElementById('c1'),[m,sn],['#5aa9ff','#3ecf8e'],0,Math.max(0.5,...m,...sn));
  line(document.getElementById('c2'),[lo],['#f5b84c'],Math.min(...lo,0)*0.9,Math.max(...lo,0.1)*1.1);
  const keys=e.length?Object.keys(e[0]):[];
  document.getElementById('tbl').innerHTML='<tr>'+keys.map(k=>`<th>${k}</th>`).join('')+'</tr>'+
    e.map(r=>'<tr>'+keys.map(k=>`<td>${typeof r[k]==='number'&&!Number.isInteger(r[k])?r[k].toFixed(3):r[k]}</td>`).join('')+'</tr>').join('');
  document.getElementById('updated').textContent='updated '+new Date().toLocaleTimeString();
  if(s.error)document.getElementById('latest').textContent=s.error;}
async function preview(){const b=document.getElementById('pv'),m=document.getElementById('pvmsg');
  b.disabled=true;m.textContent=' rendering on CPU…';
  const r=await fetch('api/preview.png?t='+Date.now());
  if(r.ok){const url=URL.createObjectURL(await r.blob());document.getElementById('pvimg').innerHTML=`<img src="${url}">`;m.textContent='';}
  else m.textContent=' '+(await r.text());b.disabled=false;}
tick();setInterval(tick,5000);
</script></body></html>"""


def create_monitor(run_dir: str | Path, log: str | Path | None) -> FastAPI:
    run = Path(run_dir)
    logp = Path(log) if log else None
    app = FastAPI(title="Radio training monitor")
    cache: dict[str, object] = {}

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE

    @app.get("/api/status")
    def api_status():
        return JSONResponse(status(run, logp))

    @app.get("/api/preview.png")
    def api_preview():
        ck = run / "model.pt"
        if not ck.exists() or not (run / "config.yaml").exists():
            return Response("no checkpoint yet (saved after the first epoch)", status_code=404)
        if "root" not in (yaml.safe_load((run / "config.yaml").read_text()) or {}):
            return Response("preview is available for detector runs only", status_code=400)
        key = ck.stat().st_mtime
        if cache.get("key") != key:
            cache["png"] = _preview_png(run)
            cache["key"] = key
        return Response(cache["png"], media_type="image/png")

    return app
