# AugmentedSocialScientist enhanced fork

> Fine-tuning BERT & friends for social-science projects, with robust tracking, smart model selection, and a reinforced learning safety-net.

---

## 1. What is this fork?

This repository is a **drop-in replacement** for the excellent original [rubingshen/AugmentedSocialScientist](https://github.com/rubingshen/AugmentedSocialScientist).  
All base classes (`BertBase`, `CamembertBase`, …) function identically but include additional features:

| Feature | Description |
|---------|-------------|
| **Full metric logging** | Each epoch's metrics are recorded in CSV format for easy tracking. |
| **Per-epoch checkpoints** | Every epoch generates a checkpoint, with only the best-performing checkpoint retained. |
| **Smart best-model picker** | Selects the best checkpoint using a combined metric (70% class 1 F1 + 30% macro F1 by default). |
| **Automatic reinforced training** | Optionally activates a reinforced training phase if class 1 F1 is below 0.60, using oversampling, larger batches, and weighted loss. |
| **Apple Silicon / MPS support** | Supports Mac M-series GPUs natively, as well as CUDA devices. |

All other functionalities remain unchanged.

---

## 2. Features in depth

### 2.1 Per-epoch metric tracking

Calling `run_training` generates `./training_logs/training_metrics.csv`, logging metrics per epoch:

```csv
epoch,train_loss,val_loss,precision_0,recall_0,f1_0,support_0,precision_1,recall_1,f1_1,support_1,macro_f1,reinforced_phase
1,0.4421,0.3875,0.79,0.83,0.81,412,0.55,0.38,0.45,88,0.63,normal
...
```

### 2.2 Best-model selection & checkpointing

- After each epoch, the model is evaluated with:  
  `combined_metric = 0.7 × F1(class 1) + 0.3 × macro_F1`.
- If this metric surpasses the previous best, the checkpoint is saved to:  
  `./models/<your_name>_epoch_<n>/`.
- Only the best model is kept, older checkpoints are deleted.
- New best models are logged in `training_logs/best_models.csv`.

### 2.3 Reinforced training safety-net

If the final model's **F1(class 1)** is **below 0.60** and `reinforced_learning=True`:

1. Creates a dataloader using `WeightedRandomSampler` (oversamples minority class).
2. Doubles batch size (64) and reduces learning rate (5e-6).
3. Applies weighted cross-entropy loss (`pos_weight=2.0`).
4. Runs additional epochs (`reinforced_n_epochs`, default=2), logs metrics to `training_metrics.csv`, and saves checkpoints under:  
   `./models/<name>_reinforced_epoch_<n>/`

### 2.4 Device auto-detection

Initialization (`BertBase.__init__`) prioritizes devices:

1. CUDA → `torch.device("cuda")`
2. Apple Silicon MPS → `torch.device("mps")`
3. CPU fallback

No additional configuration needed—import and start training.

---

## 3. Quick-start

```python
from augmented_social_scientist import BertBase

# 1. Tokenize and create data loaders
model = BertBase(model_name="bert-base-cased")
train_loader = model.encode(train_texts, train_labels)
val_loader = model.encode(val_texts, val_labels)

# 2. Train the model
model.run_training(
    train_loader,
    val_loader,
    n_epochs=10,
    save_model_as="my_policy_model",
    reinforced_learning=True
)

# 3. Load and predict later
best_model = model.load_model("./models/my_policy_model")
pred_proba = model.predict_with_model(val_loader, "./models/my_policy_model")
```

Console output example:

```
======== Epoch 4 / 10 ========
Training...
  Average training loss: 0.35
Running Validation...
New best model found at epoch 4 with combined metric=0.7123.
```

Filesystem structure:

```
models/
└── my_policy_model/          # final best model
training_logs/
├── training_metrics.csv
└── best_models.csv
```

---

## 4. Adjustable parameters

| Parameter | Default | Functionality |
|-----------|---------|---------------|
| `f1_class_1_weight` | `0.7` | Weight of class 1 F1 in best-model metric. |
| `metrics_output_dir` | `"./training_logs"` | CSV metrics output directory. |
| `pos_weight` | `None` | Class weighting during normal training. |
| `n_epochs` | `3` | Number of epochs before reinforced training. |
| `reinforced_learning` | `False` | Activate reinforced training if necessary. |
| `reinforced_n_epochs` | `2` | Number of reinforced training epochs. |

Adjust parameters directly when calling `run_training()`.

---

## 5. Installation

```bash
git clone https://github.com/<your-handle>/AugmentedSocialScientist.git
cd AugmentedSocialScientist
pip install -e .
```

Requires **Python 3.10+**, `torch >= 2.0`, and `transformers >= 4.40`.

---

## 6. License & citation

This fork remains under the original **MIT License**.  
If used academically, please cite the original paper: [rubingshen/AugmentedSocialScientist](https://github.com/rubingshen/AugmentedSocialScientist).

Happy fine-tuning!
