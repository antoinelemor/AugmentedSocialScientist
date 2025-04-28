# AugmentedSocialScientist enhanced fork

> Fine‑tuning BERT & friends for social‑science projects, with robust tracking, smart model selection, and a reinforced‑learning safety‑net.

---

## 1. Overview

This repository is a fork with new functionnalities of the original [rubingshen/AugmentedSocialScientist](https://github.com/rubingshen/AugmentedSocialScientist).  
All base classes (`BertBase`, `CamembertBase`, …) function identically while exposing the additional capabilities listed below.

---

## 2  Key capabilities

| Capability                        | Description                                                                                                                                                                                         |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Comprehensive metric logging**  | Every epoch is appended to `training_logs/training_metrics.csv` (and `reinforced_training_metrics.csv` if applicable) with losses, per‑class precision/recall/F1, and macro F1.                     |
| **Per‑epoch checkpoints**         | A lightweight checkpoint is written after each epoch; only the best checkpoint (see below) is retained to save disk space.                                                                          |
| **Smart best‑model selection**    | By default the model maximising `0.7 × F1₁ + 0.3 × macro‑F1` is kept. The weight and even the formula can be overridden.                                                                            |
| **Automatic reinforced training** | When the positive‑class F1 stays below 0.60, the library launches an adaptive reinforced phase with class‑weighted loss, oversampling, larger batches and a reduced learning rate.                  |
| **Rescue logic for class 1**      | If the best normal model achieved `F1₁ == 0`, reinforced training considers any epoch where `F1₁ > f1_1_rescue_threshold` (default 0) as an improvement
| **Apple Silicon / MPS support**   | Native GPU acceleration on macOS (M‑series) sits alongside CUDA and CPU fall‑backs—no flags required.                                                                                               |

---

## 3  Metric tracking

- Calling `run_training` automatically creates the CSV logs mentioned above.
- A concise summary of any newly selected checkpoint (normal or reinforced) is appended to `training_logs/best_models.csv`.

---

## 4  Checkpointing & model selection

- The combined metric is re‑evaluated after every epoch.
- When it improves, the corresponding checkpoint is moved to `models/<name>/` and the previous best is deleted.
- Upon completion, the folder `models/<save_model_as>` always contains the best checkpoint (whether it came from the main loop or the reinforced phase).

---

## 5  Reinforced‑training safety‑net

If `reinforced_learning=True` **and** `F1(class 1) < 0.60` at the end of the main loop, an additional cycle starts:

1. **Oversampling** of the minority class through `WeightedRandomSampler`.
2. **Batch size** doubled to 64 and **learning rate** reduced (default 5 e‑6).
3. **Weighted cross‑entropy** with `pos_weight = 2.0` emphasises the positive class.
4. Full logging to `reinforced_training_metrics.csv` and standard checkpoint selection.
5. Optional **rescue logic** (`rescue_low_class1_f1=True`) promotes any epoch where `F1₁` breaks the zero‑barrier (threshold configurable).

---

## 6  Device auto‑detection

`BertBase.__init__()` selects the computation device in this order:

1. CUDA
2. Apple Silicon MPS
3. CPU

A one‑line message confirms the choice at runtime.

---

## 7  Quick‑start

```python
from augmented_social_scientist import BertBase

# 1 – encode data
model = BertBase(model_name="bert-base-cased")
train_loader = model.encode(train_texts, train_labels)
val_loader   = model.encode(val_texts,   val_labels)

# 2 – train & keep the best checkpoint
after_training_scores = model.run_training(
    train_loader,
    val_loader,
    n_epochs=10,
    save_model_as="my_policy_model",
    reinforced_learning=True,
    rescue_low_class1_f1=True
)

# 3 – reload & predict
best_model = model.load_model("./models/my_policy_model")
probas = model.predict_with_model(val_loader, "./models/my_policy_model")
```

Typical console excerpt:

```
======== Epoch 4 / 10 ========
Training...
  Average training loss: 0.35
Running Validation...
New best model found at epoch 4 with combined metric = 0.7123
```

Resulting layout:

```
models/
└── my_policy_model/                 # final checkpoint
training_logs/
├── training_metrics.csv             # main loop
├── best_models.csv                  # checkpoint summary
└── reinforced_training_metrics.csv  # only if reinforced phase executed
```

---

## 8  Configuration reference

| Argument                | Default             | Purpose                                                   |
| ----------------------- | ------------------- | --------------------------------------------------------- |
| `n_epochs`              | `3`                 | Epochs in the main loop.                                  |
| `lr`                    | `5e‑5`              | Learning rate in the main loop.                           |
| `f1_class_1_weight`     | `0.7`               | Weight of `F1₁` in the combined metric.                   |
| `metrics_output_dir`    | `"./training_logs"` | Location of CSV logs.                                     |
| `pos_weight`            | `None`              | Class weights for the loss in the main loop.              |
| `reinforced_learning`   | `False`             | Enable the safety‑net phase.                              |
| `n_epochs_reinforced`   | `2`                 | Epochs in the reinforced phase.                           |
| `rescue_low_class1_f1`  | `False`             | Activate the rescue logic for stalled `F1₁`.              |
| `f1_1_rescue_threshold` | `0.0`               | Minimal `F1₁` improvement that triggers rescue promotion. |

Hyper‑parameters inside `reinforced_training` (batch size, LR, `pos_weight`) can be overridden by subclassing or editing the method.

---

## 9  Installation

```bash
git clone https://github.com/<your‑handle>/AugmentedSocialScientist.git
cd AugmentedSocialScientist
pip install -e .
```

**Requirements** : Python 3.10+, `torch >= 2.0`, `transformers >= 4.40`.

---


## 10 License & citation

This fork remains under the original **MIT License**.  
If used academically, please cite the upstream repository: [rubingshen/AugmentedSocialScientist](https://github.com/rubingshen/AugmentedSocialScientist), and this repo if you're cool.

Happy fine‑tuning!

