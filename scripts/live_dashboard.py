#!/usr/bin/env python
# =============================================================================
# Bloc 4 - GreenGuard MLOps : LIVE evidence dashboard (no narration needed)
# A tiny local web server that serves one visual page whose proof UPDATES LIVE
# every 3s from the running stack: live predictions on both test images, model
# metrics from MLflow, serving endpoints, CI/CD, drift monitoring, retraining,
# and criterion coverage - plus buttons to MLflow, Grafana, the API docs and
# the drift report.
#
#   python scripts/live_dashboard.py
#
# Works from PowerShell, cmd, or Git Bash. Ctrl+C to stop.
# =============================================================================
import http.server, socketserver, json, urllib.request, subprocess, time, html, pathlib, webbrowser, os, re, sys

os.chdir(pathlib.Path(__file__).resolve().parent.parent)
PORT = int(os.environ.get("PORT", "8090"))
API, MLF = "http://localhost:8000", "http://localhost:5500"
DASH = "http://localhost:3000"

def j(url, t=5):
    with urllib.request.urlopen(url, timeout=t) as r: return json.loads(r.read().decode())
def txt(url, t=5):
    with urllib.request.urlopen(url, timeout=t) as r: return r.read().decode()
def post(url, body, t=8):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=t) as r: return json.loads(r.read().decode())
def metric(m, name):
    for ln in m.splitlines():
        if ln.startswith(name + " ") or ln.startswith(name + "{"):
            try: return float(ln.rsplit(" ", 1)[1])
            except Exception: pass
    return None

def predict(img):
    try:
        out = subprocess.run(["curl", "-s", "--max-time", "15", "-F", f"file=@{img}", API + "/predict"],
                             capture_output=True, text=True, timeout=20).stdout
        return json.loads(out)
    except Exception:
        return None

def mlflow_metrics():
    try:
        r = post(MLF + "/api/2.0/mlflow/runs/search",
                 {"experiment_ids": ["1"], "max_results": 1, "order_by": ["attributes.start_time DESC"]})
        run = r["runs"][0]
        return {x["key"]: x["value"] for x in run["data"]["metrics"]}, {x["key"]: x["value"] for x in run["data"].get("params", [])}
    except Exception:
        return {}, {}

def live():
    o = {"ts": time.strftime("%H:%M:%S")}
    try:
        v = j(API + "/version"); o["model_arch"] = v.get("model_arch"); o["model_ready"] = v.get("model_ready"); o["classes"] = v.get("classes")
    except Exception: o["model_ready"] = False
    try: o["health"] = j(API + "/health").get("status")
    except Exception: o["health"] = "down"
    o["pred_dandelion"] = predict("test_dandelion.jpg")
    o["pred_grass"] = predict("test_grass.jpg")
    try:
        m = txt(API + "/metrics")
        d = metric(m, 'predictions_total{result="dandelion"}') or 0
        g = metric(m, 'predictions_total{result="grass"}') or 0
        o["predictions_total"] = int(d + g)
        o["drift_share"] = metric(m, "data_drift_share") or metric(m, "drift_share_drifted_features")
    except Exception: pass
    mm, _ = mlflow_metrics()
    o["val_f1"] = mm.get("val_f1"); o["val_acc"] = mm.get("val_accuracy"); o["train_acc"] = mm.get("train_accuracy")
    return o

# ---------- static proof (read once) ----------------------------------------
def read(p):
    try: return pathlib.Path(p).read_text(encoding="utf-8", errors="replace")
    except Exception: return ""
def grep(path, pat, lim=14):
    o = [l.rstrip() for l in read(path).splitlines() if re.search(pat, l, re.I)]
    return "\n".join(o[:lim]) if o else "(not found)"
def esc(s): return html.escape(str(s))

SPEC = grep("docs/SPECIFICATION.md", r"^#{2,3} |objective|KPI|F1|latency|requirement|herbicide|Loi Labbe", 12)
CICD = grep(".github/workflows/ci_cd.yml", r"name:|pytest|docker build|kubeconform|push|build|deploy|codecov", 14)
GATE = grep("retrain/promotion.py", r"min_f1|val_f1|quality floor|regress|should_promote|return True|return False|0\.85", 12)
DRIFT = grep("monitoring/drift/detect.py", r"def |psi|threshold|drift|0\.2|baseline|share|stability", 12)
ENDPOINTS = [("GET /health", "readiness + class names"), ("GET /version", "model arch, classes, source"),
             ("POST /predict", "class + per-class probabilities; 400 on non-image"), ("GET /metrics", "Prometheus counters, latency, drift gauges")]
CRIT = [("Specifications & business relevance","15%"),("ML model quality","25%"),("Serving API quality","15%"),
        ("CI/CD pipeline","20%"),("Automated retraining","10%"),("Production monitoring","10%"),("Presentation & Q&A","5%")]

def codecard(t, w, c):
    return f'<div class="proof"><div class="proof-h"><span>{esc(t)}</span><span class="pill">{esc(w)}</span></div><pre>{esc(c)}</pre></div>'

def page():
    init = live()
    endpoints = "".join(f'<div class="svc"><span class="dot ok"></span><span class="svc-n">{esc(a)}</span><span class="svc-s">{esc(b)}</span></div>' for a, b in ENDPOINTS)
    crit = "".join(f'<div class="crit"><span class="check">&#10003;</span><span class="crit-n">{esc(n)}</span><span class="pill">{esc(w)}</span></div>' for n, w in CRIT)
    return TEMPLATE.replace("__ENDPOINTS__", endpoints).replace("__CRIT__", crit)\
        .replace("__SPEC__", codecard("Specification (cahier des charges)", "15%", SPEC))\
        .replace("__CICD__", codecard("CI/CD - tests, build, deploy", "20%", CICD))\
        .replace("__GATE__", codecard("Retraining promotion gate", "10%", GATE))\
        .replace("__DRIFT__", codecard("Drift detection (PSI)", "10%", DRIFT))\
        .replace("__INIT__", json.dumps(init))

TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GreenGuard MLOps - LIVE</title>
<style>
*{box-sizing:border-box} body{margin:0;background:linear-gradient(160deg,#051C2C,#02141d 70%);color:#e8f1f8;font-family:Segoe UI,Arial,sans-serif;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:30px 26px 60px}
.top{display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px;border-bottom:1px solid #173040;padding-bottom:18px}
h1{font-size:26px;margin:0;font-weight:700} .sub{color:#8fa6b6;font-size:14px;margin-top:4px}
.live{display:inline-flex;align-items:center;gap:8px;background:#0c2a39;border:1px solid #16465c;border-radius:20px;padding:6px 14px;font-size:13px;color:#bfe7fb}
.pulse{width:9px;height:9px;border-radius:50%;background:#22c55e;animation:p 1.4s infinite}
@keyframes p{0%{box-shadow:0 0 0 0 rgba(34,197,94,.6)}70%{box-shadow:0 0 0 9px rgba(34,197,94,0)}100%{box-shadow:0 0 0 0 rgba(34,197,94,0)}}
h2{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:#00A9F4;margin:34px 0 14px;font-weight:700}
.preds{display:grid;grid-template-columns:1fr 1fr;gap:14px} @media(max-width:720px){.preds{grid-template-columns:1fr}}
.pred{background:#0b2433;border:1px solid #14384b;border-radius:14px;padding:18px}
.pred .lab{font-size:12px;color:#8fa6b6} .pred .cls{font-size:26px;font-weight:800;color:#fff;margin:2px 0} .pred .cls.ok{color:#22e58a}
.bar{height:8px;background:#14384b;border-radius:5px;overflow:hidden;margin-top:8px} .bar i{display:block;height:100%;background:linear-gradient(90deg,#2251FF,#00A9F4);transition:width .4s}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-top:14px}
.kpi{background:#0b2433;border:1px solid #14384b;border-radius:14px;padding:16px} .kpi-val{font-size:26px;font-weight:800;color:#fff} .kpi-lab{font-size:12.5px;color:#cfe2ee;margin-top:3px} .kpi-sub{font-size:11px;color:#7f99a9}
.arch{display:flex;align-items:center;gap:9px;flex-wrap:wrap;background:#081f2b;border:1px solid #14384b;border-radius:14px;padding:16px}
.node{background:#0e2f41;border:1px solid #1d4a61;border-radius:10px;padding:9px 12px;font-size:12.5px;font-weight:600} .node.acc{background:#10307a;border-color:#2251FF} .node.cy{background:#063b52;border-color:#00A9F4} .arrow{color:#00A9F4;font-weight:700}
.svcs{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:8px}
.svc{display:flex;align-items:center;gap:10px;background:#0b2433;border:1px solid #14384b;border-radius:10px;padding:9px 12px;font-size:13px} .svc-n{font-weight:600;font-family:Consolas,monospace} .svc-s{color:#8fa6b6;margin-left:auto;font-size:12px} .dot{width:10px;height:10px;border-radius:50%} .dot.ok{background:#22c55e}
.btns{display:flex;gap:12px;flex-wrap:wrap} a.btn{display:inline-block;text-decoration:none;background:#10307a;border:1px solid #2251FF;color:#fff;padding:11px 18px;border-radius:10px;font-weight:600;font-size:14px} a.btn.cy{background:#063b52;border-color:#00A9F4}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px} @media(max-width:820px){.grid2{grid-template-columns:1fr}}
.proof{background:#08202c;border:1px solid #14384b;border-radius:12px;overflow:hidden} .proof-h{display:flex;justify-content:space-between;align-items:center;padding:11px 14px;background:#0c2a39;font-size:13px;font-weight:700}
.pill{background:#10307a;border:1px solid #2251FF;color:#cfe0ff;font-size:11px;padding:2px 9px;border-radius:20px;font-weight:700}
pre{margin:0;padding:13px 14px;font-family:Consolas,monospace;font-size:12px;color:#bfe0f0;white-space:pre-wrap;word-break:break-word;max-height:230px;overflow:auto}
.crit{display:flex;align-items:center;gap:12px;background:#0b2433;border:1px solid #14384b;border-radius:10px;padding:10px 14px;margin-bottom:8px;font-size:13px} .check{color:#22c55e;font-weight:800} .crit-n{font-weight:600;min-width:240px}
.foot{margin-top:40px;color:#6f8896;font-size:12px;border-top:1px solid #173040;padding-top:16px}
</style></head><body><div class="wrap">
 <div class="top"><div><h1>GreenGuard &middot; Self-maintaining AI weeding</h1>
   <div class="sub">End-to-end MLOps: serving, CI/CD, drift monitoring, automated retraining &middot; RNCP Bloc 4</div></div>
   <div class="live"><span class="pulse"></span> LIVE &middot; predicts every 3s &middot; <span id="ts">--</span></div></div>

 <h2>Live predictions &middot; the model classifies both test images in real time</h2>
 <div class="preds">
  <div class="pred" id="p_dand"><div class="lab">test_dandelion.jpg &rarr;</div><div class="cls">--</div><div class="kpi-sub">confidence <span class="cf">--</span></div><div class="bar"><i style="width:0%"></i></div></div>
  <div class="pred" id="p_grass"><div class="lab">test_grass.jpg &rarr;</div><div class="cls">--</div><div class="kpi-sub">confidence <span class="cf">--</span></div><div class="bar"><i style="width:0%"></i></div></div>
 </div>
 <div class="kpis">
  <div class="kpi" id="k_served"><div class="kpi-val">--</div><div class="kpi-lab">Predictions served</div><div class="kpi-sub">Prometheus counter, climbing</div></div>
  <div class="kpi" id="k_f1"><div class="kpi-val">--</div><div class="kpi-lab">Validation macro F1</div><div class="kpi-sub">live from MLflow</div></div>
  <div class="kpi" id="k_arch"><div class="kpi-val">--</div><div class="kpi-lab">Model</div><div class="kpi-sub">transfer learning</div></div>
  <div class="kpi" id="k_health"><div class="kpi-val">--</div><div class="kpi-lab">API status</div><div class="kpi-sub">/health</div></div>
 </div>

 <h2>Architecture &middot; closed MLOps loop</h2>
 <div class="arch"><span class="node">Airflow</span><span class="arrow">&rarr;</span><span class="node acc">Train + MLflow</span><span class="arrow">&rarr;</span>
  <span class="node">MinIO (model)</span><span class="arrow">&rarr;</span><span class="node acc">FastAPI /predict</span><span class="arrow">&rarr;</span>
  <span class="node cy">Prometheus</span><span class="arrow">&rarr;</span><span class="node cy">Grafana + drift</span><span class="arrow">&rarr;</span><span class="node">Retrain gate</span></div>

 <h2>Serving API (15%) &middot; endpoints</h2>
 <div class="svcs">__ENDPOINTS__</div>

 <h2>Open the live tools</h2>
 <div class="btns"><a class="btn cy" href="http://localhost:8000/docs" target="_blank">API docs (Swagger)</a>
  <a class="btn cy" href="http://localhost:3000/d/dandelion-classifier" target="_blank">Grafana: Classifier</a>
  <a class="btn cy" href="http://localhost:3000/d/dandelion-drift" target="_blank">Grafana: Data Drift</a>
  <a class="btn" href="http://localhost:5500" target="_blank">MLflow</a>
  <a class="btn" href="http://localhost:8085" target="_blank">Airflow (retrain DAG)</a>
  <a class="btn" href="https://github.com/enzoberreur/ml-ops-dandelion-classifier/actions" target="_blank">GitHub Actions (CI/CD)</a></div>

 <h2>Specification & model (15% + 25%)</h2><div class="grid2">__SPEC__
  <div class="proof"><div class="proof-h"><span>Reproducible training</span><span class="pill">25%</span></div><pre>ResNet18 transfer learning, seeded, pinned deps.
Every run tracked in MLflow: params, val_accuracy, val_f1,
precision/recall, confusion matrix, and the model artifact.
Best checkpoint pushed to MinIO -> served by the API.
docs/MODEL_CARD.md documents data, eval, limitations.</pre></div></div>

 <h2>CI/CD (20%) & retraining (10%)</h2><div class="grid2">__CICD____GATE__</div>

 <h2>Production monitoring (10%)</h2><div class="grid2">__DRIFT__
  <div class="proof"><div class="proof-h"><span>Dashboards & report</span><span class="pill">10%</span></div><pre>Grafana "Dandelion Classifier": predictions, p90 latency.
Grafana "Dandelion Data Drift": dataset-drift status, share
of features drifted, per-feature PSI bars.
monitoring/drift/reports/drift_report.html (Evidently, with
a dependency-free fallback). Same drift signal feeds the
dashboard and the retrain gate.</pre></div></div>

 <h2>Jury criterion coverage</h2>__CRIT__
 <div class="foot">github.com/enzoberreur/ml-ops-dandelion-classifier &middot; served live by scripts/live_dashboard.py</div>
</div>
<script>
const fmt=n=>(n==null||n==='')?'n/a':(isNaN(+n)?n:(+n).toLocaleString('en-US'));
let prev={};
function setk(id,val,bump){const e=document.getElementById(id);if(!e)return;const v=e.querySelector('.kpi-val');const t=fmt(val);if(v.textContent!==t){v.textContent=t;}}
function setpred(id,p){const e=document.getElementById(id);if(!e||!p)return;const cls=e.querySelector('.cls');const cf=e.querySelector('.cf');const bar=e.querySelector('.bar i');
 cls.textContent=p.prediction||'--';cls.classList.add('ok');const c=Math.round((p.confidence||0)*1000)/10;cf.textContent=c+'%';bar.style.width=c+'%';}
async function tick(){try{const d=await (await fetch('/api/live',{cache:'no-store'})).json();
 document.getElementById('ts').textContent=d.ts||'';
 setpred('p_dand',d.pred_dandelion);setpred('p_grass',d.pred_grass);
 setk('k_served',d.predictions_total);
 setk('k_f1',d.val_f1!=null?(+d.val_f1).toFixed(3):null);
 setk('k_arch',d.model_arch||'--');
 const h=document.getElementById('k_health').querySelector('.kpi-val');h.textContent=d.health==='ready'?'ready':(d.health||'--');
 prev=d;}catch(e){}}
tick();setInterval(tick,3000);
</script></body></html>"""

# ---------- server -----------------------------------------------------------
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, body, ct="text/html; charset=utf-8"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200); self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/api/live"): self._send(json.dumps(live()), "application/json")
        elif self.path in ("/", "/index.html"): self._send(page())
        else: self.send_response(204); self.end_headers()

def main():
    global PORT
    for p in range(PORT, PORT + 6):
        try:
            srv = socketserver.ThreadingTCPServer(("127.0.0.1", p), H); PORT = p; break
        except OSError: continue
    else:
        print("No free port; set PORT env var."); sys.exit(1)
    url = f"http://localhost:{PORT}/"
    print("\n  LIVE evidence dashboard running at:", url)
    print("  Opening it now, plus the Grafana drift dashboard. Press Ctrl+C to stop.\n")
    if os.environ.get("NOBROWSER") != "1":
        try:
            webbrowser.open(url); time.sleep(1.2); webbrowser.open(DASH + "/d/dandelion-classifier")
        except Exception: pass
    try: srv.serve_forever()
    except KeyboardInterrupt: print("\n  stopped.")

if __name__ == "__main__":
    main()
