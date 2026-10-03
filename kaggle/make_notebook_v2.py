"""Regenerates kaggle/fraudshield_laya_v2_kaggle.ipynb (Laya v2, "Laya decides"; docs/LAYA_V2.md):
uv run python kaggle/make_notebook_v2.py kaggle/fraudshield_laya_v2_kaggle.ipynb"""
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
# FraudShield: Laya v2 fine-tune ("Laya decides, LightGBM testifies") on Kaggle GPU T4 x2

About 4.5 hours with Save & Run All. Design and the pre-registered win condition: `docs/LAYA_V2.md` in the bundle.
1. Finds the `fraudshield-laya-v2-bundle` dataset, copies it to `/kaggle/working/fs`, installs `laya==0.3.23`.
2. Rebuilds the training items from `artifacts/laya_v2/train.jsonl` and re-checks the token budget.
3. Trains on GPU 0 with the setting the v1 run proved on a T4 (top 12 layers, micro-batch 16 x 2, fp16, 2 epochs).
   As soon as epoch 1 is saved, GPU 1 scores that checkpoint, so a usable result exists even if time runs short.
4. Scores the final model on GPU 0 and GPU 1 in parallel: test with the LightGBM line, test with it withheld,
   and the calibration set.
5. Puts the model, scores and logs in `/kaggle/working/fraudshield_laya_v2_output.zip` (Output tab).
Evaluation and calibration run on the laptop afterwards (CPU): `scripts/laya_v2_eval.py`.

Data: Olist (CC BY-NC-SA 4.0) plus synthetic injected fraud. Keep the dataset and this notebook PRIVATE.
""")

code(r'''
# 1. Locate the bundle and make a writable working copy
import os, sys, json, time, shutil, zipfile, subprocess
from pathlib import Path

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/fs")
OUT = Path("/kaggle/working/output_v2")
LA = WORK / "artifacts" / "laya_v2"
os.environ["FS_LAYA_ART"] = str(LA)            # every script reads and writes the v2 folder

if not (WORK / "scripts" / "laya_train.py").exists():
    hits = sorted(INPUT.rglob("scripts/laya_train.py"))
    if hits:
        shutil.copytree(hits[0].parent.parent, WORK, dirs_exist_ok=True)
    else:
        zips = sorted(INPUT.rglob("fraudshield_laya_v2_bundle.zip")) or sorted(INPUT.rglob("*.zip"))
        assert zips, "Bundle not found. Use 'Add Input' to attach your fraudshield-laya-v2-bundle dataset."
        zipfile.ZipFile(zips[0]).extractall(WORK)
sys.path.insert(0, str(WORK))
os.chdir(WORK)
assert (LA / "train.jsonl").exists(), "this is not the v2 bundle (artifacts/laya_v2/train.jsonl missing)"
print("working copy:", WORK, sorted(p.name for p in LA.iterdir()))


def run(args, log=None):
    print("$ python", " ".join(args), flush=True)
    p = subprocess.Popen([sys.executable] + args, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
# 2. Install (torch is Kaggle's own)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(WORK / "requirements.txt")], check=True)
q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True).stdout.strip().splitlines()
GPUS = [(n.strip(), float(m) / 1024) for n, m in (l.split(",") for l in q)]
N_GPU = len(GPUS)
vcode = ("import json, sys, torch, transformers, laya; print(json.dumps({'python': sys.version.split()[0], "
         "'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__, "
         "'laya': laya.__version__}))")
VERSIONS = {**json.loads(subprocess.run([sys.executable, "-c", vcode], capture_output=True, text=True).stdout.strip()),
            "gpus": GPUS}
print(VERSIONS)
assert N_GPU >= 1, "No GPU. Settings > Accelerator > GPU T4 x2."
''')

code(r'''
# 3. Resume support: if a previous run of this notebook was attached as input, reuse its state
for name in ("resume_state.pt", "train.log"):
    prev = [p for p in INPUT.rglob(name) if "artifacts/laya_v2" in p.as_posix()]
    if prev and not (LA / name).exists():
        shutil.copy(prev[0], LA / name)
        print("restored", name, "from", prev[0])
for d in INPUT.rglob("ckpt_epoch*"):
    if d.is_dir() and "laya_v2" in d.as_posix() and not (LA / d.name).exists():
        shutil.copytree(d, LA / d.name)
        print("restored", d.name)
''')

code(r'''
# 4. Rebuild training items from train.jsonl (downloads the stock Laya once) and re-check the token budget
if not (LA / "train_items.pt").exists():
    must(["scripts/laya_prepare.py", "--from-jsonl", "artifacts/laya_v2/train.jsonl"])
PREP = json.loads((LA / "prepare.json").read_text())
print(PREP["items"], "training items;", PREP["fraud_unique"], "distinct fraud;", PREP["bookings"], "bookings")
''')

code(r'''
# 5. Train on GPU 0. Fixed setting proven by the v1 run on a T4 (4.98 GB peak, 8.4 items/s): no pilot needed.
CFG = {"k_top": 12, "micro": 16, "accum": 2, "epochs": 2}
CFG["est_hours"] = round(PREP["items"] * CFG["epochs"] / 8.4 / 3600, 2)
(LA / "run_config.json").write_text(json.dumps({**CFG, "note": "fixed (v1 pilot on T4)"}, indent=2))
print(CFG)
ck = LA / "fraudshield-laya" / "rl_agent_config.json"
done = ck.exists() and json.loads(ck.read_text()).get("fraudshield", {}).get("epoch") == CFG["epochs"]
ep1_jobs = []
if not done:
    t0 = time.time()
    tr = start_bg(["scripts/laya_train.py", "--epochs", str(CFG["epochs"]), "--k-top", str(CFG["k_top"]),
                   "--micro", str(CFG["micro"]), "--accum", str(CFG["accum"]), "--device", "cuda:0", "--resume"],
                  LA / "train_stdout.txt")
    shown = 0
    while tr.poll() is None:
        time.sleep(60)
        lines = (LA / "train.log").read_text().splitlines() if (LA / "train.log").exists() else []
        for line in lines[shown:]:
            print(line, flush=True)
        shown = len(lines)
        # safety net: score the epoch-1 checkpoint on GPU 1 while epoch 2 trains
        e1 = LA / "ckpt_epoch1"
        if N_GPU >= 2 and not ep1_jobs and (e1 / "rl_agent_config.json").exists() and CFG["epochs"] > 1:
            time.sleep(20)  # let the checkpoint finish writing
            ep1_jobs.append(start_bg(["scripts/laya_score.py", "--model", "ft", "--set", "test", "--path", str(e1),
                                      "--tag", "_ep1", "--with-action", "--device", "cuda:1", "--batch", "32"],
                                     LA / "score_ep1_test.log"))
    print("training exit code", tr.returncode, f"after {(time.time() - t0) / 3600:.2f} h")
    print("\n".join((LA / "train_stdout.txt").read_text().splitlines()[-30:]))
    assert tr.returncode == 0, "training failed; see train_stdout.txt above"
chk = ("import laya, json, pandas as pd; a = laya.load('artifacts/laya_v2/fraudshield-laya', device='cuda:0'); "
       "from fraudshield.contracts import QUESTIONS, SERVED_QUESTIONS; "
       "s = pd.read_parquet('artifacts/laya_v2/evalset_test.parquet').state.iloc[0]; "
       "qs = {q: QUESTIONS[q] for q in SERVED_QUESTIONS + ('action',)}; r = a.predict(s, qs); "
       "print({q: r['answers'][q]['probabilities'] for q in qs})")
must(["-c", chk])
''')

code(r'''
# 6. Score the final model: GPU 0 = test (with the LightGBM line, + action); GPU 1 = calibration set, then
#    test with the LightGBM line withheld. Waits for the epoch-1 safety scoring first.
for j in ep1_jobs:
    print("epoch-1 scoring exit code", j.wait())
bg = None
if N_GPU >= 2 and not (LA / "scores_ft_nogbm_test.parquet").exists():
    cmd = ("import subprocess, sys; "
           "subprocess.run([sys.executable, 'scripts/laya_score.py', '--model', 'ft', '--set', 'cal', '--with-action', "
           "'--device', 'cuda:1', '--batch', '32'], check=True); "
           "subprocess.run([sys.executable, 'scripts/laya_score.py', '--model', 'ft', '--set', 'test', "
           "'--state-col', 'state_nogbm', '--tag', '_nogbm', '--device', 'cuda:1', '--batch', '32'], check=True)")
    bg = start_bg(["-c", cmd], LA / "score_gpu1.log")
if not (LA / "scores_ft_test.parquet").exists():
    must(["scripts/laya_score.py", "--model", "ft", "--set", "test", "--with-action", "--device", "cuda:0", "--batch", "32"])
if bg is not None:
    print("GPU 1 scoring exit code", bg.wait())
    print(open(LA / "score_gpu1.log").read()[-1500:])
if not (LA / "scores_ft_cal.parquet").exists():
    must(["scripts/laya_score.py", "--model", "ft", "--set", "cal", "--with-action", "--device", "cuda:0", "--batch", "32"])
if not (LA / "scores_ft_nogbm_test.parquet").exists():
    must(["scripts/laya_score.py", "--model", "ft", "--set", "test", "--state-col", "state_nogbm", "--tag", "_nogbm",
          "--device", "cuda:0", "--batch", "32"])
''')

code(r'''
# 7. Collect outputs into /kaggle/working/fraudshield_laya_v2_output.zip
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "laya_v2").mkdir(parents=True)
shutil.copytree(LA / "fraudshield-laya", OUT / "laya_v2" / "fraudshield-laya")
for p in LA.iterdir():
    if p.is_file() and (p.suffix in (".parquet", ".log", ".txt") or p.name.endswith(".json")) \
            and not p.name.startswith("evalset_") and p.name != "train.jsonl":
        shutil.copy(p, OUT / "laya_v2" / p.name)
(OUT / "laya_v2" / "versions.json").write_text(json.dumps(VERSIONS, indent=2))
zp = Path("/kaggle/working/fraudshield_laya_v2_output.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT).as_posix())
print(f"{zp} {zp.stat().st_size / 1e9:.2f} GB")
for p in [LA / "resume_state.pt", *LA.glob("ckpt_epoch*"), LA / "fraudshield-laya", OUT]:
    if p.exists():
        shutil.rmtree(p) if p.is_dir() else p.unlink()
print("DONE. Download fraudshield_laya_v2_output.zip from the Output tab.")
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", OUT, len(cells), "cells")
