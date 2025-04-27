# AugmentedSocialScientist enhanced fork

> Fine-tuning BERT & friends for social-science projects, with robust tracking, smart model selection, and a reinforced learning safety-net.

---

## 1. What is this fork?

This repository is a **drop-in replacement** for the excellent original [rubingshen/AugmentedSocialScientist](https://github.com/rubingshen/AugmentedSocialScientist).  
All base classes (`BertBase`, `CamembertBase`, …) function identically but include additional features:

| Feature | Description |
|---------|-------------|
| **Comprehensive metric logging** | Each epoch is recorded in `training_logs/training_metrics.csv`, enabling full post‑hoc analysis. |
| **Per‑epoch checkpoints** | A checkpoint is written after every epoch; only the best checkpoint (chosen with a combined metric) is kept. |
| **Smart best‑model selection** | The checkpoint maximising `0.7 × F1₁ + 0.3 × macro‑F1` is retained by default; both the weights and the formula are configurable. |
| **Automatic reinforced training** | If the positive‑class F1 remains below 0.60, the library enters a short reinforced phase with oversampling, weighted loss, and adaptive hyper‑parameters. |
| **Apple Silicon / MPS support** | Native GPU acceleration on macOS (M‑series) sits alongside CUDA and CPU fall‑backs—no flags needed. |

---

## 2. Feature details

### 2.1 Metric tracking

* Calling `run_training` automatically creates `training_logs/training_metrics.csv` with per‑epoch loss and classification metrics for both classes plus macro F1.

### 2.2 Checkpointing and best‑model selection

* After each epoch the library computes a **combined metric** (see above).
* If the score improves, the model is saved to `models/<name>_epoch_<n>/` and the previous checkpoint is removed.
* A concise summary is appended to `training_logs/best_models.csv`.

### 2.3 Reinforced‑training safety net

When the best model after the main loop has **F1(class 1) < 0.60** *and* `reinforced_learning=True`, a reinforced phase is triggered:

1. Minority‑class oversampling (`WeightedRandomSampler`)
2. Larger batches (64) and a lower learning rate (5 e‑6)
3. Weighted cross‑entropy (`pos_weight = 2.0`)
4. Two extra epochs by default, fully logged in `reinforced_training_metrics.csv`.
5. If the reinforced checkpoint surpasses the previous best, it replaces it transparently.

### 2.4 Device auto‑detection

`BertBase.__init__()` now selects the execution device in this order:

1. CUDA
2. Apple Silicon MPS
3. CPU

The choice is printed at runtime.

---

## 3. Quick‑start

```python
from augmented_social_scientist import BertBase

# 1 – Prepare dataloaders
model = BertBase(model_name="bert-base-cased")
train_loader = model.encode(train_texts, train_labels)
val_loader   = model.encode(val_texts,   val_labels)

# 2 – Train and keep the best checkpoint
model.run_training(
    train_loader,
    val_loader,
    n_epochs=10,
    save_model_as="my_policy_model",
    reinforced_learning=True
)

# 3 – Reload and predict
best_model = model.load_model("./models/my_policy_model")
probas = model.predict_with_model(val_loader, "./models/my_policy_model")
```

During training you will see messages such as:

```
======== Epoch 4 / 10 ========
Training...
  Average training loss: 0.35
Running Validation...
New best model found at epoch 4 with combined metric = 0.7123
```

Directory structure after training:

```
models/
└── my_policy_model/                 # best checkpoint
training_logs/
├── training_metrics.csv
├── best_models.csv
└── reinforced_training_metrics.csv  # created only if reinforced phase ran
```

---

## 4. Configuration

| Argument | Default | Purpose |
|----------|---------|---------|
| `f1_class_1_weight` | `0.7` | Weight of positive‑class F1 in the combined metric. |
| `metrics_output_dir` | `"./training_logs"` | Directory where CSV logs are stored. |
| `pos_weight` | `None` | Class weights for the loss function during main training. |
| `n_epochs` | `3` | Epochs in the main training loop. |
| `n_epochs_reinforced` | `2` | Epochs in the reinforced phase. |
| Reinforced LR | `5e‑6` | Learning rate during the reinforced phase. |
| Reinforced batch | `64` | Batch size during the reinforced phase. |

Modify `run_training` or `reinforced_training` if you need finer‑grained control.

---

## 5. Installation

```bash
git clone https://github.com/<your‑handle>/AugmentedSocialScientist.git
cd AugmentedSocialScientist
pip install -e .
```

**Requirements** : Python 3.10+, `torch >= 2.0`, `transformers >= 4.40`.

---

## 6. License & citation

This fork remains under the original **MIT License**.  
If used academically, please cite the original paper: [rubingshen/AugmentedSocialScientist](https://github.com/rubingshen/AugmentedSocialScientist).

Happy fine-tuning!
