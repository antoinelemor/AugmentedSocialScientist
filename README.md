# AugmentedSocialScientist enhancements

## About this fork

This repository is a thin fork of [rubingshen/AugmentedSocialScientist](https://github.com/rubingshen/AugmentedSocialScientist) with three main improvements:

- Adaptive training strategy for difficult labels (class 1 f1-score < 0.5)
- Automatic oversampling of the minority class in memory
- Native GPU detection on macOS (MPS) as well as CUDA

## New features

### Adaptive training for low-f1 tasks

If your previous model’s f1-score on the positive class is below 0.5, the training script will automatically:

1. Increase the batch size (64 instead of 32)
2. Lower the learning rate (1e-5 instead of 5e-5)
3. Apply a class-weighted loss (`pos_weight = n_neg / n_pos`)
4. Oversample the positive examples in memory until they roughly match the number of negatives
5. Train for an extended number of epochs (15 instead of your recorded best epoch)

If the f1-score is greater than or equal to 0.5, the original defaults are kept:

- Batch size = 32
- Learning rate = 5e-5
- No class weighting
- No oversampling
- Number of epochs = best epoch from your metrics CSV (or 10 if missing)

### Improved GPU detection

In `BertBase.__init__` (and similarly in `CamembertBase`), the device is now automatically detected:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

macOS users on Apple silicon will see:

mps is available. using the Apple Silicon GPU!

Installation

You can install directly from this repository:

pip install git+https://github.com/yourusername/AugmentedSocialScientist.git

Or clone the repository and install it in editable mode:

git clone https://github.com/yourusername/AugmentedSocialScientist.git
cd AugmentedSocialScientist
pip install -e .

Usage

from AugmentedSocialScientist.bert_base import BertBase
import torch

# Load your data
train_texts, train_labels = [...], [...]
val_texts, val_labels     = [...], [...]

# Instantiate the model (CUDA, MPS or CPU will be selected automatically)
model = BertBase(model_name="bert-base-cased")

# Decide hyperparameters based on the previous f1-score
prev_f1 = 0.42
if prev_f1 < 0.5:
    batch_size = 64
    lr         = 1e-5
    pos_weight = torch.tensor([n_neg / n_pos], device=model.device)
    n_epochs   = 15
else:
    batch_size = 32
    lr         = 5e-5
    pos_weight = None
    n_epochs   = your_best_epoch

# Prepare dataloaders
train_loader = model.encode(train_texts, train_labels, batch_size=batch_size)
val_loader   = model.encode(val_texts, val_labels, batch_size=batch_size)

# Train and save the model
scores = model.run_training(
    train_loader,
    val_loader,
    n_epochs=n_epochs,
    lr=lr,
    pos_weight=pos_weight,
    save_model_as="my_model_name"
)
print("Precision, recall, f1, support:", scores)

# After training, the model is saved under ./models/my_model_name/
# You can load and use it like this
model = BertBase(model_name="bert-base-cased")
trained_model = model.load_model("./models/my_model_name")
preds = model.predict_with_model(val_loader, "./models/my_model_name")

License

Same license as the original project (MIT).
