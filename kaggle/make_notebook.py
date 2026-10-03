"""Regenerates kaggle/fraudshield_laya_kaggle.ipynb: uv run python kaggle/make_notebook.py kaggle/fraudshield_laya_kaggle.ipynb"""
import json
import sys

OUT = sys.argv[1]
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# FraudShield: fine-tune Laya on Kaggle (GPU T4 x2)

What this notebook does, top to bottom (Save & Run All, about 3 to 5 hours on 2x T4):
1. Finds the `fraudshield-laya-bundle` dataset, copies it to `/kaggle/working/fs`, installs `laya==0.3.23`.
2. Rebuilds the training items from `train.jsonl` and re-checks the token budget (0 dropped, 0 truncated).
3. Pilot: 200 training steps per candidate setting (top 12 layers / micro-batch 16, then top 8 / 8), measures items/s
   and VRAM, and picks the setting with a safe rule (peak reserved VRAM under 85% of the card, 2 epochs under 6 h;
   otherwise 1 epoch).
4. Scores the STOCK Laya on the test set on the second GPU while training runs on the first.
5. Trains (checkpoint and resume state after every epoch; rerunning resumes).
6. Scores the fine-tuned model (test + calibration sets), fits `calibration.json`, writes `results_laya.md`.
7. Builds `laya_cache.json` for the demo bookings and measures latency.
8. Puts everything in `/kaggle/working/fraudshield_laya_output.zip` (download it from the Output tab).

Data: Olist (CC BY-NC-SA 4.0) plus synthetic injected fraud. Keep the dataset and this notebook PRIVATE.
""")

code(r'''
# 1. Locate the bundle and make a writable working copy
import os, sys, json, time, shutil, zipfile, subprocess
from pathlib import Path

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/fs")          # writable copy of the bundle (looks like the repo)
OUT = Path("/kaggle/working/output")
LA = WORK / "artifacts" / "laya"

if not (WORK / "scripts" / "laya_train.py").exists():
    hits = sorted(INPUT.rglob("scripts/laya_train.py"))
    if hits:                                   # Kaggle auto-extracted the uploaded zip
        shutil.copytree(hits[0].parent.parent, WORK, dirs_exist_ok=True)
    else:                                      # the zip itself is in the dataset
        zips = sorted(INPUT.rglob("fraudshield_laya_bundle.zip")) or sorted(INPUT.rglob("*.zip"))
        assert zips, "Bundle not found. Use 'Add Input' to attach your fraudshield-laya-bundle dataset."
        zipfile.ZipFile(zips[0]).extractall(WORK)
sys.path.insert(0, str(WORK))
os.chdir(WORK)
from fraudshield.models.laya_train.kaggle import find_bundle, choose_run_config
print("bundle found at:", find_bundle(INPUT))
print("working copy:", WORK, sorted(p.name for p in LA.iterdir()))


def run(args, log=None):
    """Run a python script from the bundle, stream its output here, return (exit code, last lines)."""
    print("$ python", " ".join(args), flush=True)
    p = subprocess.Popen([sys.executable] + args, cwd=WORK, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    tail = []
    fh = open(log, "a") if log else None
    for line in p.stdout:
        if fh:
            fh.write(line)
        if "Warning" not in line and "warnings.warn" not in line:
            print(line, end="", flush=True)
        tail = (tail + [line])[-40:]
    if fh:
        fh.close()
    return p.wait(), "".join(tail)


def must(args, log=None):
    rc, tail = run(args, log)
    if rc != 0:
        raise RuntimeError(f"step failed (exit {rc}):\n{tail}")


def start_bg(args, log):
    print("$ (background) python", " ".join(args), flush=True)
    return subprocess.Popen([sys.executable] + args, cwd=WORK, stdout=open(log, "a"), stderr=subprocess.STDOUT)
''')

code(r'''
# 2. Install (torch is Kaggle's own; laya only needs torch>=2.0)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(WORK / "requirements.txt")], check=True)
q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True).stdout.strip().splitlines()
GPUS = [(n.strip(), float(m) / 1024) for n, m in (l.split(",") for l in q)]
N_GPU, GPU_GB = len(GPUS), min(g for _, g in GPUS)
vcode = ("import json, sys, torch, transformers, laya; print(json.dumps({'python': sys.version.split()[0], "
         "'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__, "
         "'laya': laya.__version__}))")
vers = subprocess.run([sys.executable, "-c", vcode], capture_output=True, text=True).stdout.strip()
VERSIONS = {**json.loads(vers), "gpus": GPUS}
print(VERSIONS)
assert N_GPU >= 1, "No GPU. Settings > Accelerator > GPU T4 x2."
''')

code(r'''
# 3. Resume support: if a previous run of this notebook was attached as input, reuse its state
for name in ("resume_state.pt", "run_config.json", "train.log", "pilot.json"):
    prev = [p for p in INPUT.rglob(name) if "fs/artifacts/laya" in p.as_posix()]
    if prev and not (LA / name).exists():
        shutil.copy(prev[0], LA / name)
        print("restored", name, "from", prev[0])
for d in INPUT.rglob("ckpt_epoch*"):
    if d.is_dir() and not (LA / d.name).exists():
        shutil.copytree(d, LA / d.name)
        print("restored", d.name)
''')

code(r'''
# 4. Rebuild training items from train.jsonl (downloads the stock Laya once) and re-check the token budget
if not (LA / "train_items.pt").exists():
    must(["scripts/laya_prepare.py", "--from-jsonl", "artifacts/laya/train.jsonl"])
print(json.loads((LA / "prepare.json").read_text())["items"], "training items")
''')

code(r'''
# 5. Pilot (200 steps per candidate on GPU 0) and the run configuration
RC = LA / "run_config.json"
if RC.exists():
    cfg = json.loads(RC.read_text())                  # resuming: never change the configuration mid-run
else:
    pilots = []
    for k, micro in [(12, 16), (8, 8)]:
        n0 = len(json.loads((LA / "pilot.json").read_text())) if (LA / "pilot.json").exists() else 0
        rc, tail = run(["scripts/laya_train.py", "--pilot-steps", "200", "--k-top", str(k), "--micro", str(micro),
                        "--accum", str(max(1, 32 // micro)), "--device", "cuda:0"], log=LA / "pilot_stdout.txt")
        ps = json.loads((LA / "pilot.json").read_text()) if (LA / "pilot.json").exists() else []
        if rc == 0 and len(ps) > n0:
            pilots.append(ps[-1])
        else:
            pilots.append({"k_top": k, "micro": micro, "oom": True, "error": tail[-400:]})
        print("pilot:", pilots[-1])
        p = pilots[-1]
        if not p.get("oom") and p["peak_vram_reserved_gb"] <= 0.85 * GPU_GB:
            break                                     # preferred setting fits; no need to try a smaller one
    items = json.loads((LA / "prepare.json").read_text())["items"]
    cfg = choose_run_config(pilots, GPU_GB, items, budget_hours=6.0, epochs=2)
    cfg["pilots"] = pilots
    RC.write_text(json.dumps(cfg, indent=2))
print(json.dumps(cfg, indent=2))
''')

code(r'''
# 6. Stock Laya (zero-shot baseline B3) on the test set, in the background on GPU 1 while training runs
stock = None
if not (LA / "scores_stock_test.parquet").exists() and N_GPU >= 2:
    stock = start_bg(["scripts/laya_score.py", "--model", "stock", "--set", "test", "--device", "cuda:1",
                      "--batch", "32"], LA / "score_stock_test.log")
''')

code(r'''
# 7. Train on GPU 0 (resumes automatically from the last finished epoch)
ck = LA / "fraudshield-laya" / "rl_agent_config.json"
done = ck.exists() and json.loads(ck.read_text()).get("fraudshield", {}).get("epoch") == cfg["epochs"]
if not done:
    t0 = time.time()
    must(["scripts/laya_train.py", "--epochs", str(cfg["epochs"]), "--k-top", str(cfg["k_top"]),
          "--micro", str(cfg["micro"]), "--accum", str(cfg["accum"]), "--device", "cuda:0", "--resume"],
         log=LA / "train_stdout.txt")
    print(f"training took {(time.time() - t0) / 3600:.2f} h")
# the checkpoint must load with laya.load(<dir>) and answer
chk = ("import laya, json; a = laya.load('artifacts/laya/fraudshield-laya', device='cuda:0'); "
        "from fraudshield.contracts import QUESTIONS, SERVED_QUESTIONS; "
        "s = json.load(open('artifacts/laya/demo_states.json'))[0][1]; "
        "r = a.predict(s, {q: QUESTIONS[q] for q in SERVED_QUESTIONS}); "
        "print({q: r['answers'][q]['probabilities'] for q in SERVED_QUESTIONS})")
must(["-c", chk])
''')

code(r'''
# 8. Score: wait for stock; fine-tuned on test (GPU 0) and calibration (GPU 1) in parallel
if stock is not None:
    print("waiting for stock scoring ...", flush=True)
    print("stock scoring exit code", stock.wait())
    print(open(LA / "score_stock_test.log").read()[-1500:])
if not (LA / "scores_stock_test.parquet").exists():
    must(["scripts/laya_score.py", "--model", "stock", "--set", "test", "--device", "cuda:0", "--batch", "32"])
cal = None
if not (LA / "scores_ft_cal.parquet").exists():
    if N_GPU >= 2:
        cal = start_bg(["scripts/laya_score.py", "--model", "ft", "--set", "cal", "--device", "cuda:1",
                        "--batch", "32"], LA / "score_ft_cal.log")
if not (LA / "scores_ft_test.parquet").exists():
    must(["scripts/laya_score.py", "--model", "ft", "--set", "test", "--device", "cuda:0", "--batch", "32"])
if cal is not None:
    print("ft cal exit code", cal.wait())
if not (LA / "scores_ft_cal.parquet").exists():
    must(["scripts/laya_score.py", "--model", "ft", "--set", "cal", "--device", "cuda:0", "--batch", "32"])
''')

code(r'''
# 9. Calibration + evaluation (CPU): artifacts/calibration.json, artifacts/results_laya.md
must(["scripts/laya_eval.py"])
''')

code(r'''
# 10. Demo cache (the 5 demo bookings, every way the app builds their state) + latency on this GPU
must(["scripts/laya_cache.py", "--device", "cuda:0", "--states-json", "artifacts/laya/demo_states.json"])
''')

code(r'''
# 11. Collect outputs into /kaggle/working/fraudshield_laya_output.zip
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "laya").mkdir(parents=True)
shutil.copytree(LA / "fraudshield-laya", OUT / "fraudshield-laya")
for f in ("calibration.json", "results_laya.md", "results_laya.json", "laya_cache.json"):
    shutil.copy(WORK / "artifacts" / f, OUT / f)
for p in LA.iterdir():
    if p.is_file() and (p.suffix in (".parquet", ".log", ".txt") or p.name.endswith(".json")) \
            and not p.name.startswith("evalset_") and p.name != "train.jsonl":
        shutil.copy(p, OUT / "laya" / p.name)
(OUT / "versions.json").write_text(json.dumps(VERSIONS, indent=2))
zp = Path("/kaggle/working/fraudshield_laya_output.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT).as_posix())
print(f"{zp} {zp.stat().st_size / 1e9:.2f} GB")
# keep the saved output small: drop the resume state and per-epoch checkpoints (the zip has the final model)
for p in [LA / "resume_state.pt", *LA.glob("ckpt_epoch*"), LA / "fraudshield-laya", OUT]:
    if p.exists():
        shutil.rmtree(p) if p.is_dir() else p.unlink()
print(open(WORK / "artifacts" / "results_laya.md").read())
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", OUT, len(cells), "cells")
