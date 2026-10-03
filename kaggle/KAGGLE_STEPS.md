# Fine-tune Laya on Kaggle: step by step

You will upload one zip file, import one notebook, press one button, and download one zip about 3 to 5
hours later. Kaggle's free GPUs do the heavy work, so nothing heavy runs on your laptop.

**Before you start:** on your PC, in the `fraudshield` folder, make sure this file exists:
`kaggle/dist/fraudshield_laya_bundle.zip` (about 5 MB). If it does not, run:
```
uv run python kaggle/build_kaggle_bundle.py
```
You also need `kaggle/fraudshield_laya_kaggle.ipynb`.

**License:** the bundle contains data derived from the Olist dataset (CC BY-NC-SA 4.0). Keep the dataset
and the notebook PRIVATE. Never make them public.

## 1. Create and verify your Kaggle account
1. Go to https://www.kaggle.com and sign up (Google sign-in is fine).
2. Click your profile picture (top right), then **Settings**.
3. Find **Phone verification** and verify your phone number. Without it Kaggle will not give you a GPU
   or internet access in notebooks.

## 2. Upload the bundle as a private dataset
1. In the left menu click **Datasets**, then **+ New Dataset**.
2. Drag in `fraudshield_laya_bundle.zip`.
3. Title: `fraudshield-laya-bundle`.
4. Visibility: **Private**.
5. Click **Create**. Wait until it says the dataset is ready. Kaggle usually unpacks the zip by itself;
   that is fine, because the notebook handles both the zip and the unpacked folder.

## 3. Create the notebook
1. In the left menu click **Code**, then **+ New Notebook**.
2. In the notebook menu: **File > Import Notebook**, then upload `fraudshield_laya_kaggle.ipynb`.
3. In the right-hand panel click **+ Add Input**. Choose **Your Work** or **Datasets**, find
   `fraudshield-laya-bundle`, and click the plus sign to add it.
4. Still in the right panel, under **Session options** or **Settings**:
   - **Accelerator:** `GPU T4 x2`
   - **Internet:** On (the notebook downloads the base Laya model and installs `laya`)
   - **Persistence:** `Files only`, if the option is shown
5. Optional quick check: click into the first code cell and press Shift+Enter. It should print
   `bundle found at: ...`. If it says the bundle was not found, the dataset is not attached (step 3.3).
   Then stop the session (the power icon), so you do not burn GPU hours.

## 4. Run it in the background
1. Top right: **Save Version**.
2. Choose **Save & Run All (Commit)** and click **Save**.
3. You can close the browser. Kaggle runs the whole notebook on its own machine. It takes about 3 to 5
   hours: a 10-minute pilot picks the settings, training takes most of the time, and then scoring,
   calibration and the demo cache take about 30 more minutes. The run is planned to finish well
   inside Kaggle's 12-hour limit. If the pilot shows it would be too slow, it runs 1 epoch instead of 2
   and says so in the results.
4. Kaggle gives about 30 GPU hours per week; one run uses about 4 to 10 of them (2 GPUs count double).

## 5. Check progress
1. Go to **Code > Your Work**, open the notebook, and click the running version (in the **Versions** list).
2. Open **Logs**. Lines to look for, in order:
   - `pilot: {... "items_per_s": ...}`: the pilot speed and VRAM
   - `"k_top": ..., "epochs": ...`: the chosen setting
   - `ep 1/2 step 50/...`: training progress, every 50 steps, with items/s and minutes elapsed
   - `=== epoch 1 done`, then `saved ... and resume state`
   - `TRAIN DONE`, then the scoring lines, then the results table at the very end

## 6. Download the result
1. When the version shows **Complete**, open it and go to the **Output** tab (or the **Data** section of
   the version page).
2. Download `fraudshield_laya_output.zip` (about 0.8 GB).
3. On your PC, put it here: `fraudshield/kaggle/dist/fraudshield_laya_output.zip`. Do not commit it;
   that folder is git-ignored.
4. Tell the team. To import it yourself, run this on your PC (CPU only, no GPU used):
   ```
   uv run python kaggle/import_kaggle_output.py kaggle/dist/fraudshield_laya_output.zip
   ```
   It checks the files, copies the model to `artifacts/laya/fraudshield-laya/`, copies
   `calibration.json`, `laya_cache.json` and `results_laya.md` into `artifacts/`, and points
   `config/app.yaml` at the new model.

## If something goes wrong
- **"Bundle not found":** the dataset is not attached. Repeat step 3.3.
- **"No GPU":** set the accelerator to `GPU T4 x2` (step 3.4). Your phone must be verified.
- **The run failed or stopped part way:** open **Logs** and copy the last 30 lines for the team. After
  every finished epoch the notebook saves a checkpoint and a resume state under
  `/kaggle/working/fs/artifacts/laya/`. To continue rather than start over, open the notebook editor,
  **+ Add Input > Your Work > (this notebook's failed version output)**, and **Save & Run All** again;
  the third cell copies the resume state back and training continues from the last finished epoch.
  Kaggle does not always keep the files of a failed or timed-out version, so this is best effort.
