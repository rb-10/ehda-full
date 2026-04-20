"""
train_final_model.py

Trains the best configuration found during comparison:
  - Dataset:       optical_images2
  - Model:         squeezenet1_1
  - Preprocessing: no crop
  - Exports:       image_classification/final_model/export.pkl  (ready for deployment)
"""

from fastai.vision.all import *
from fastai.callback.tracker import SaveModelCallback
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import multiprocessing
import json
from pathlib import Path
import os

# ── Config ────────────────────────────────────────────────────────────
DATASET_PATH  = Path(r"C:\Users\HV\Desktop\bruno_work\EHDA Image Classificaton\ehda-image-classification\dataset\processed_images\training")
MODEL_ARCH   = squeezenet1_1
EPOCHS       = 10
BATCH_SIZE   = 64
VALID_SPLIT  = 0.2
NUM_WORKERS  = 0       # must be 0 on Windows
OUTPUT_DIR   = Path("image_classification/final_model")
 
 
if __name__ == "__main__":
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    # ── Data ──────────────────────────────────────────────────────────
    files   = get_image_files(DATASET_PATH)
    classes = sorted(set(parent_label(f) for f in files))
 
    print(f"[TRAIN] Dataset:  {DATASET_PATH}")
    print(f"[TRAIN] Classes:  {classes}")
    print(f"[TRAIN] Samples:  {len(files)}")
    print(f"[TRAIN] Epochs:   {EPOCHS}\n")
 
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=VALID_SPLIT, seed=42),
        get_y=parent_label,
        item_tfms=[]
    )
    dls = dblock.dataloaders(DATASET_PATH, bs=BATCH_SIZE, num_workers=NUM_WORKERS)
 
    # ── Train ──────────────────────────────────────────────────────────
    learn = vision_learner(dls, MODEL_ARCH, metrics=[accuracy], pretrained=True)
    learn = learn.to_fp16()
    learn.path = OUTPUT_DIR
 
    learn.fine_tune(
        EPOCHS,
        cbs=SaveModelCallback(monitor="valid_loss", fname="best_model")
    )
 
    learn.load("best_model")
    val_loss, acc = learn.validate()[:2]
    print(f"\n[TRAIN] Final accuracy: {acc:.4f}  val_loss={val_loss:.4f}")
 
    # ── Confusion matrix ───────────────────────────────────────────────
    preds, targs = learn.get_preds(dl=dls.valid)
    cm = confusion_matrix(targs.numpy(), preds.argmax(dim=1).numpy(), labels=list(range(len(classes))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation="vertical")
    plt.title(f"Final model  (acc={acc:.4f})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()
    print(f"[TRAIN] Confusion matrix saved → {OUTPUT_DIR}/confusion_matrix.png")
 
    # ── Save results ───────────────────────────────────────────────────
    results = {
        "model":       "squeezenet1_1",
        "dataset":     "optical_images2",
        "crop":        False,
        "epochs":      EPOCHS,
        "valid_split": VALID_SPLIT,
        "classes":     classes,
        "accuracy":    float(acc),
        "val_loss":    float(val_loss),
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=4)
 
    # ── Export ─────────────────────────────────────────────────────────
    learn.export("export.pkl")
    print(f"[EXPORT] Model saved → {OUTPUT_DIR}/export.pkl")
    print(f'\nSet in mapsetup.json:')
    print(f'  "image_model_path": "{OUTPUT_DIR}/export.pkl"')