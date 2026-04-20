from fastai.vision.all import *
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Prepare files and labels
path = Path("datasets/diffimg")
files = get_image_files(path)
labels = [parent_label(f) for f in files]
classes = sorted(list(set(labels)))

# Convert string labels to numeric indices
class2idx = {c: i for i, c in enumerate(classes)}
y_numeric = np.array([class2idx[l] for l in labels])

# Setup 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

all_preds = []
all_targs = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(files), 1):
    print(f"Fold {fold}")

    # Subset the files
    train_files = [files[i] for i in train_idx]
    valid_files = [files[i] for i in valid_idx]

    # DataBlock for this fold
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=lambda _: train_files + valid_files,
        splitter=IndexSplitter(valid_idx),
        get_y=parent_label,
        item_tfms=[]  # images already 128x128
    )

    dls = dblock.dataloaders(path, bs=32)

    # Create learner
    learn = vision_learner(dls, resnet18, metrics=accuracy, pretrained=True)

    # Fine-tune
    learn.fine_tune(5)

    # Get validation predictions
    preds, targs = learn.get_preds(dl=dls.valid)
    pred_labels = preds.argmax(dim=1).numpy()
    targ_labels = targs.numpy()

    # Collect predictions and targets
    all_preds.append(pred_labels)
    all_targs.append(targ_labels)

# Aggregate predictions across folds
all_preds = np.concatenate(all_preds)
all_targs = np.concatenate(all_targs)

# 4Confusion matrix
cm = confusion_matrix(all_targs, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("5-Fold Cross-Validation")
plt.tight_layout()
plt.show()