# AugmentedSocialScientist enhanced fork

> Fine-tuning BERT & friends for social-science projects, with robust tracking, smart model selection, and a safety-net if performance stalls.

---

## 1. What is this fork?

This repo is a **drop-in replacement** for the excellent original  
[rubingshen/AugmentedSocialScientist](https://github.com/rubingshen/AugmentedSocialScientist).  
All base classes (`BertBase`, `CamembertBase`, …) still work the same way but I added a few extras :

| What's new | What exactly |
|-------|-------------------|
| **Full metric logging** | Every epoch is logged to CSV—no more guessing what happened in the middle of the night. |
| **Per-epoch checkpoints** | A checkpoint is written each epoch, but only the *best* one is kept. |
| **Smart best-model picker** | Chooses the checkpoint that maximises a *combined metric* (70 % F1 on the positive class + 30 % macro F1 by default). |
| **Automatic reinforced training** | If class 1 F1 < 0.60 after training, the code launches a short extra pass with over-sampling, larger batches and weighted loss. |
| **Apple Silicon / MPS support** | Runs out-of-the-box on Mac M-series GPUs as well as CUDA. |

Everything else stays unchanged.

---

## 2. New features in depth

### 2.1 Per-epoch metric tracking

* When you call `run_training`, the class now creates `./training_logs/training_metrics.csv`
  with one line per epoch:

```csv
epoch,train_loss,val_loss,precision_0,recall_0,f1_0,support_0,precision_1,recall_1,f1_1,support_1,macro_f1
1,0.4421,0.3875,0.79,0.83,0.81,412,0.55,0.38,0.45,88,0.63
...
```

### 2.2 Best-model selection & checkpointing

* After each epoch the code computes  
  `combined_metric = 0.7 × F1(class 1) + 0.3 × macro_F1`.  
* If this score beats the previous best, the model is written to  
  `./models/<your_name>_epoch_<n>/`.
* The previous “best” folder is deleted, so you never keep more than one checkpoint per run.
* A summary of every new best model goes to `training_logs/best_models.csv`.

### 2.3 Reinforced training safety-net

If the **final** best model still has **F1(class 1) < 0.60**:

1. A new dataloader is built with `WeightedRandomSampler` (oversamples the minority class).  
2. Batch-size is doubled (64 by default) and learning-rate divided by 10.  
3. A weighted cross-entropy loss emphasises class 1 (`pos_weight=2.0` by default).  
4. Two quick extra epochs are run. Metrics land in `reinforced_training_metrics.csv`, and the model is saved under `./models/<name>_reinforced/`.

### 2.4 Better device auto-detection

`BertBase.__init__` now tries in this order:

1. CUDA → `torch.device("cuda")`  
2. Apple Silicon MPS → `torch.device("mps")`  
3. CPU fallback

No flags to set—just import and go.

---

## 3. Quick-start

```python
from augmented_social_scientist import BertBase

# 1. Tokenise / create loaders
model = BertBase(model_name="bert-base-cased")
train_loader = model.encode(train_texts, train_labels)
val_loader   = model.encode(val_texts,   val_labels)

# 2. Train for 10 epochs and keep the best checkpoint
model.run_training(
    train_loader,
    val_loader,
    n_epochs=10,
    save_model_as="my_policy_model"
)

# 3. Load the best model later
best_model = model.load_model("./models/my_policy_model")
pred_proba = model.predict_with_model(val_loader, "./models/my_policy_model")
```

During training you will see something like:

```
======== Epoch 4 / 10 ========
Training...
  Average training loss: 0.35
Running Validation...
New best model found at epoch 4 with combined metric=0.7123.
```

And your disk will look like:

```
models/
└── my_policy_model/           # final best checkpoint
training_logs/
├── training_metrics.csv
├── best_models.csv
└── reinforced_training_metrics.csv   # only if triggered
```

---

## 4. Adjusting the knobs

| Argument | Default | What it does |
|----------|---------|--------------|
| `f1_class_1_weight` | `0.7` | Weight for class 1 F1 in best-model metric. |
| `metrics_output_dir` | `"./training_logs"` | Where all CSV logs are written. |
| `pos_weight` | `None` | Tensor to re-weight classes *during normal* training. |
| `n_epochs` | `3` | Training epochs before any reinforced pass. |
| Reinforced LR / batch / epochs | hard-coded (`5e-6`, `64`, `2`) | Edit `reinforced_training()` if you need different values. |

---

## 5. Example notebooks

See `notebooks/` for minimal end-to-end demos:

* **binary_sentiment.ipynb** – fine-tune on IMDB, inspect CSV logs.  
* **policy_frames_fr_en.ipynb** – multilingual training with forced reinforced pass.

---

## 6. Installation

```bash
git clone https://github.com/<your-handle>/AugmentedSocialScientist.git
cd AugmentedSocialScientist
pip install -e .
```

Requires **Python 3.10+**, `torch >= 2.0`, `transformers >= 4.40`.

---

## 7. License & citation

This fork remains under the original **MIT License**.  
If you use it in academic work, please cite the upstream paper [rubingshen/AugmentedSocialScientist](https://github.com/rubingshen/AugmentedSocialScientist).

Happy fine-tuning!
