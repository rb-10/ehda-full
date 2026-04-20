from fastai.vision.all import *
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Path to your images
path = Path("datasets/diffimg")

# Create DataBlock without resizing
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),      # images → categories
    get_items=get_image_files,               # find image files
    splitter=RandomSplitter(valid_pct=0.2),
    get_y=parent_label,                      # label = folder name
    item_tfms=[]                             # no resizing
)

# Create dataloaders
dls = dblock.dataloaders(path, bs=32)

# Make a ResNet18 learner
learn = vision_learner(
    dls,
    resnet18,           # small ResNet suitable for 128x128 images
    metrics=accuracy,
    pretrained=True     # can also set False if you want to train from scratch
)

# Optional: fine-tune
learn.fine_tune(10)

interp = ClassificationInterpretation.from_learner(learn)

# Create figure
fig = plt.figure(figsize=(4,4))
interp.plot_confusion_matrix(figsize=(4,4))
plt.tight_layout()
plt.show()