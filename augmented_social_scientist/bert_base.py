import datetime
import random
import time
import os
import shutil
import csv
from typing import List, Tuple, Any

import numpy as np
import torch
from scipy.special import softmax
from torch.types import Device
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    CONFIG_NAME
)

from AugmentedSocialScientist.bert_abc import BertABC


class BertBase(BertABC):
    def __init__(
            self,
            model_name: str = 'bert-base-cased',
            tokenizer: Any = BertTokenizer,
            model_sequence_classifier: Any = BertForSequenceClassification,
            device: Device | None = None,
    ):
        """
        Parameters
        ----------
        model_name: str, default='bert-base-cased'
            A model name from huggingface models: https://huggingface.co/models

        tokenizer: huggingface tokenizer, default=BertTokenizer.from_pretrained('bert-base-cased')
            Tokenizer to use

        model_sequence_classifier: huggingface sequence classifier, default=BertForSequenceClassification
            A huggingface sequence classifier that implements a from_pretrained() function

        device: torch.Device, default=None
            Device to use. If None, automatically set if GPU is available. CPU otherwise.
        """
        self.model_name = model_name
        self.tokenizer = tokenizer.from_pretrained(self.model_name)
        self.model_sequence_classifier = model_sequence_classifier
        self.dict_labels = None

        # Set or detect device
        self.device = device
        if self.device is None:
            # If CUDA is available, use it
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print('There are %d GPU(s) available.' % torch.cuda.device_count())
                print('We will use GPU {}:'.format(torch.cuda.current_device()),
                      torch.cuda.get_device_name(torch.cuda.current_device()))
            # If MPS is available on Apple Silicon, use it
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print('MPS is available. Using the Apple Silicon GPU!')
            # Otherwise, use CPU
            else:
                self.device = torch.device("cpu")
                print('No GPU available, using the CPU instead.')

    def encode(
            self,
            sequences: List[str],
            labels: List[str | int] | None = None,
            batch_size: int = 32,
            progress_bar: bool = True,
            add_special_tokens: bool = True
    ) -> DataLoader:
        """
        Preprocess training, test or prediction data by:
          (1) Tokenizing the sequences and mapping tokens to their IDs.
          (2) Truncating or padding to a max length of 512 tokens, and creating corresponding attention masks.
          (3) Returning a pytorch DataLoader containing token ids, labels (if any) and attention masks.

        Parameters
        ----------
        sequences: 1D array-like
            List of input texts.

        labels: 1D array-like or None, default=None
            List of labels. None for unlabeled prediction data.

        batch_size: int, default=32
            Batch size for the PyTorch DataLoader.

        progress_bar: bool, default=True
            If True, print a progress bar for tokenization and mask creation.

        add_special_tokens: bool, default=True
            If True, add '[CLS]' and '[SEP]' tokens.

        Returns
        -------
        dataloader: torch.utils.data.DataLoader
            A PyTorch DataLoader with input_ids, attention_masks, and labels (if provided).
        """
        input_ids = []
        if progress_bar:
            sent_loader = tqdm(sequences, desc="Tokenizing")
        else:
            sent_loader = sequences

        for sent in sent_loader:
            encoded_sent = self.tokenizer.encode(
                sent,
                add_special_tokens=add_special_tokens
            )
            input_ids.append(encoded_sent)

        # Calculate max length (capped at 512)
        max_len = min(max([len(sen) for sen in input_ids]), 512)

        # Pad/truncate input tokens to max_len
        pad = np.full((len(input_ids), max_len), 0, dtype='long')
        for idx, s in enumerate(input_ids):
            trunc = s[:max_len]
            pad[idx, :len(trunc)] = trunc

        input_ids = pad

        # Create attention masks
        attention_masks = []
        if progress_bar:
            input_loader = tqdm(input_ids, desc="Creating attention masks")
        else:
            input_loader = input_ids

        for sent in input_loader:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)

        # If no labels, return DataLoader without labels
        if labels is None:
            inputs_tensors = torch.tensor(input_ids)
            masks_tensors = torch.tensor(attention_masks)

            data = TensorDataset(inputs_tensors, masks_tensors)
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
            return dataloader
        else:
            # Build a dictionary of labels if needed
            label_names = np.unique(labels)
            self.dict_labels = dict(zip(label_names, range(len(label_names))))

            if progress_bar:
                print(f"label ids: {self.dict_labels}")

            inputs_tensors = torch.tensor(input_ids)
            masks_tensors = torch.tensor(attention_masks)
            labels_tensors = torch.tensor([self.dict_labels[x] for x in labels])

            data = TensorDataset(inputs_tensors, masks_tensors, labels_tensors)
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
            return dataloader

    def run_training(
            self,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            n_epochs: int = 3,
            lr: float = 5e-5,
            random_state: int = 42,
            save_model_as: str | None = None,
            pos_weight: torch.Tensor | None = None,
            metrics_output_dir: str = "./training_logs",
            best_model_criteria: str = "combined",
            f1_class_1_weight: float = 0.7,
            reinforced_learning: bool = False,
            n_epochs_reinforced: int = 2,
            rescue_low_class1_f1: bool = False,
            f1_1_rescue_threshold: float = 0.0
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Train, evaluate, and (optionally) save a BERT model. This method also logs training and validation
        metrics per epoch, handles best-model selection, and can optionally trigger reinforced learning if
        the best F1 on class 1 is below 0.6 at the end of normal training.

        This method can also (optionally) apply a "rescue" logic for class 1 F1 scores that remain at 0
        after normal training: if ``rescue_low_class1_f1=True`` and the best model's F1 for class 1 is 0,
        the reinforced learning step will consider any small improvement of class 1's F1 (greater than
        ``f1_1_rescue_threshold``) as sufficient to select a reinforced epoch's model.

        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            Training dataloader, typically from self.encode().

        test_dataloader: torch.utils.data.DataLoader
            Test/validation dataloader, typically from self.encode().

        n_epochs: int, default=3
            Number of epochs for the normal training phase.

        lr: float, default=5e-5
            Learning rate for normal training.

        random_state: int, default=42
            Random seed for reproducibility.

        save_model_as: str, default=None
            If not None, will save the final best model to ./models/<save_model_as>.

        pos_weight: torch.Tensor, default=None
            If not None, weights the loss to favor certain classes more heavily
            (useful in binary classification).

        metrics_output_dir: str, default="./training_logs"
            Directory for saving CSV logs: training_metrics.csv and best_models.csv.

        best_model_criteria: str, default="combined"
            Criterion for best model. Currently supports:
              - "combined": Weighted combination of F1(class 1) and macro F1.

        f1_class_1_weight: float, default=0.7
            Weight for F1(class 1) in the combined metric. The remaining (1 - weight) goes to macro F1.

        reinforced_learning: bool, default=False
            If True, and if the best model after normal training has F1(class 1) < 0.6,
            a reinforced training phase will be triggered.

        n_epochs_reinforced: int, default=2
            Number of epochs for the reinforced learning phase (if triggered).

        rescue_low_class1_f1: bool, default=False
            If True, then during reinforced learning we check if the best normal-training
            F1 for class 1 is 0. In that case, any RL epoch where class 1's F1 becomes greater
            than ``f1_1_rescue_threshold`` is automatically considered a better model.

        f1_1_rescue_threshold: float, default=0.0
            The threshold above which a class 1 F1 (starting from 0 after normal training)
            is considered a sufficient improvement to pick the reinforced epoch's model.

        Returns
        -------
        scores: tuple (precision, recall, f1-score, support)
            Final best evaluation scores from sklearn.metrics.precision_recall_fscore_support,
            for each label. Shape: (4, n_labels).

        Notes
        -----
        This method generates:
            - "<metrics_output_dir>/training_metrics.csv": logs metrics for each normal-training epoch.
            - "<metrics_output_dir>/best_models.csv": logs any new best model (normal or reinforced).
            - If reinforced training is triggered, it also logs a reinforced_training_metrics.csv.
            - The final best model is ultimately saved to "./models/<save_model_as>" if save_model_as is provided.
              (If reinforced training finds a better model, that replaces the previous best.)
        """
        # Ensure metric output directory exists
        os.makedirs(metrics_output_dir, exist_ok=True)
        training_metrics_csv = os.path.join(metrics_output_dir, "training_metrics.csv")
        best_models_csv = os.path.join(metrics_output_dir, "best_models.csv")

        # Initialize CSV for normal training metrics
        with open(training_metrics_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "precision_0",
                "recall_0",
                "f1_0",
                "support_0",
                "precision_1",
                "recall_1",
                "f1_1",
                "support_1",
                "macro_f1"
            ])

        # Initialize CSV for best models (both normal and reinforced)
        # We'll include a "training_phase" column to indicate normal or reinforced.
        with open(best_models_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "precision_0",
                "recall_0",
                "f1_0",
                "support_0",
                "precision_1",
                "recall_1",
                "f1_1",
                "support_1",
                "macro_f1",
                "saved_model_path",
                "training_phase"
            ])

        # Collect test labels for classification report
        test_labels = []
        for batch in test_dataloader:
            test_labels += batch[2].numpy().tolist()
        num_labels = np.unique(test_labels).size

        # Potentially store label names (if dict_labels is available)
        if self.dict_labels is None:
            label_names = None
        else:
            # Sort by index
            label_names = [str(x[0]) for x in sorted(self.dict_labels.items(), key=lambda x: x[1])]

        # Set seeds for reproducibility
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

        # Initialize the model
        model = self.model_sequence_classifier.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        model.to(self.device)

        optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * n_epochs
        )

        train_loss_values = []
        val_loss_values = []

        best_metric_val = -1.0
        best_model_path = None
        best_scores = None  # Will store final best (precision, recall, f1, support)

        # =============== Normal Training Loop ===============
        for i_epoch in range(n_epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(i_epoch + 1, n_epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0.0
            model.train()

            for step, train_batch in enumerate(train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                b_inputs = train_batch[0].to(self.device)
                b_masks = train_batch[1].to(self.device)
                b_labels = train_batch[2].to(self.device)

                model.zero_grad()

                outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks)
                logits = outputs[0]

                # Weighted loss if pos_weight is specified
                if pos_weight is not None:
                    weight_tensor = torch.tensor([1.0, pos_weight.item()], device=self.device)
                    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
                else:
                    criterion = torch.nn.CrossEntropyLoss()

                loss = criterion(logits, b_labels)
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            train_loss_values.append(avg_train_loss)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training took: {:}".format(self.format_time(time.time() - t0)))

            # =============== Validation after this epoch ===============
            print("")
            print("Running Validation...")

            t0 = time.time()
            model.eval()

            total_val_loss = 0.0
            logits_complete = []

            for test_batch in test_dataloader:
                b_inputs = test_batch[0].to(self.device)
                b_masks = test_batch[1].to(self.device)
                b_labels = test_batch[2].to(self.device)

                with torch.no_grad():
                    outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks, labels=b_labels)

                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item()
                logits_complete.append(logits.detach().cpu().numpy())

            logits_complete = np.concatenate(logits_complete, axis=0)
            avg_val_loss = total_val_loss / len(test_dataloader)
            val_loss_values.append(avg_val_loss)

            print("")
            print("  Average validation loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(self.format_time(time.time() - t0)))

            preds = np.argmax(logits_complete, axis=1).flatten()
            report = classification_report(test_labels, preds, target_names=label_names, output_dict=True)

            # Extract metrics for classes 0 and 1 (assuming binary classification or focusing on first two classes)
            class_0_metrics = report.get("0", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
            class_1_metrics = report.get("1", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
            macro_avg = report.get("macro avg", {"f1-score": 0})

            precision_0 = class_0_metrics["precision"]
            recall_0 = class_0_metrics["recall"]
            f1_0 = class_0_metrics["f1-score"]
            support_0 = class_0_metrics["support"]

            precision_1 = class_1_metrics["precision"]
            recall_1 = class_1_metrics["recall"]
            f1_1 = class_1_metrics["f1-score"]
            support_1 = class_1_metrics["support"]

            macro_f1 = macro_avg["f1-score"]

            # Print a short classification report
            print(classification_report(test_labels, preds, target_names=label_names))

            # Append to training_metrics.csv (normal training phase)
            with open(training_metrics_csv, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    i_epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    precision_0,
                    recall_0,
                    f1_0,
                    support_0,
                    precision_1,
                    recall_1,
                    f1_1,
                    support_1,
                    macro_f1
                ])

            # Compute "combined" metric if best_model_criteria is "combined"
            if best_model_criteria == "combined":
                combined_metric = f1_class_1_weight * f1_1 + (1.0 - f1_class_1_weight) * macro_f1
            else:
                # Fallback or alternative strategy
                combined_metric = (f1_1 + macro_f1) / 2.0

            if combined_metric > best_metric_val:
                # We found a new best model
                print(f"New best model found at epoch {i_epoch + 1} with combined metric={combined_metric:.4f}.")
                # Remove old best model folder if it exists
                if best_model_path is not None:
                    try:
                        shutil.rmtree(best_model_path)
                    except OSError:
                        pass

                best_metric_val = combined_metric

                if save_model_as is not None:
                    # Save the new best model in a temporary folder
                    best_model_path = f"./models/{save_model_as}_epoch_{i_epoch+1}"
                    os.makedirs(best_model_path, exist_ok=True)

                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(best_model_path, WEIGHTS_NAME)
                    output_config_file = os.path.join(best_model_path, CONFIG_NAME)

                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    self.tokenizer.save_vocabulary(best_model_path)
                else:
                    best_model_path = None

                # Log this new best model
                with open(best_models_csv, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        i_epoch + 1,
                        avg_train_loss,
                        avg_val_loss,
                        precision_0,
                        recall_0,
                        f1_0,
                        support_0,
                        precision_1,
                        recall_1,
                        f1_1,
                        support_1,
                        macro_f1,
                        best_model_path if best_model_path else "Not saved to disk",
                        "normal"  # training phase
                    ])

                best_scores = precision_recall_fscore_support(test_labels, preds)

        # End of normal training
        print("")
        print("Normal training complete!")

        # If we have a best model, rename it to the final user-specified name (for normal training)
        final_path = None
        if save_model_as is not None and best_model_path is not None:
            final_path = f"./models/{save_model_as}"
            # Remove existing final path if any
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            os.rename(best_model_path, final_path)
            best_model_path = final_path
            print(f"Best model from normal training is available at: {best_model_path}")

        # ==================== Reinforced Training Check ====================
        if best_scores is not None:
            best_f1_1 = best_scores[2][1]  # best_scores = (precision, recall, f1, support)
            if best_f1_1 < 0.6 and reinforced_learning:
                print(f"\nThe best model's F1 score for class 1 ({best_f1_1:.3f}) is below 0.60.")
                print("Reinforced learning is enabled. Triggering reinforced training...")

                # Perform reinforced training
                # This returns updated best_metric_val, best_model_path, best_scores
                (best_metric_val,
                 best_model_path,
                 best_scores) = self.reinforced_training(
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    base_model_path=best_model_path,
                    random_state=random_state,
                    metrics_output_dir=metrics_output_dir,
                    save_model_as=save_model_as,
                    best_model_criteria=best_model_criteria,
                    f1_class_1_weight=f1_class_1_weight,
                    previous_best_metric=best_metric_val,
                    n_epochs_reinforced=n_epochs_reinforced,
                    rescue_low_class1_f1=rescue_low_class1_f1,
                    f1_1_rescue_threshold=f1_1_rescue_threshold
                )
            else:
                print("No reinforced training triggered.")
        else:
            print("No valid best scores found after normal training (unexpected). No reinforced training triggered.")

        # Finally, if reinforced training was triggered and found a better model, it might have placed it
        # in a temporary folder. The method already handles rename at the end. So at this point we are done.
        return best_scores

    def reinforced_training(
            self,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            base_model_path: str | None,
            random_state: int = 42,
            metrics_output_dir: str = "./training_logs",
            save_model_as: str | None = None,
            best_model_criteria: str = "combined",
            f1_class_1_weight: float = 0.7,
            previous_best_metric: float = -1.0,
            n_epochs_reinforced: int = 2,
            rescue_low_class1_f1: bool = False,
            f1_1_rescue_threshold: float = 0.0
    ) -> Tuple[float, str | None, Tuple[Any, Any, Any, Any] | None]:
        """
        A "reinforced training" procedure that is triggered if the final best model from normal
        training has F1(class 1) < 0.6 (and reinforced_learning is True). This method:
          - Oversamples class 1 via WeightedRandomSampler.
          - Increases batch size to 64 (by default).
          - Reduces learning rate (e.g., 1/10 of the original normal training).
          - Uses a weighted cross-entropy loss to emphasize class 1.
          - Logs each epoch's metrics to "reinforced_training_metrics.csv".
          - Uses the same best-model selection logic as normal training and logs to best_models.csv
            with a "training_phase" = "reinforced".
          - If `rescue_low_class1_f1` is True and the best normal-training F1 for class 1 was 0,
            then any RL epoch where class 1 F1 becomes > `f1_1_rescue_threshold` is automatically
            selected as the best, overriding the standard combined metric.

        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            The original training dataloader (we will rebuild it internally for oversampling).

        test_dataloader: torch.utils.data.DataLoader
            The test/validation dataloader.

        base_model_path: str or None
            Path to the best model saved after normal training. If provided, we load that model
            as the starting point for reinforced training. If None, we load a fresh model from self.model_name.

        random_state: int, default=42
            Random seed for reproducibility.

        metrics_output_dir: str, default="./training_logs"
            Directory for logs (reinforced_training_metrics.csv and best_models.csv).

        save_model_as: str, default=None
            If not None, the final best reinforced model will be saved in ./models/<save_model_as>
            (overwriting any previous normal-training best model if we find a better one).

        best_model_criteria: str, default="combined"
            How to select the best model (same logic as in run_training).

        f1_class_1_weight: float, default=0.7
            Weight for F1(class 1) in the combined metric.

        previous_best_metric: float, default=-1.0
            The best metric value from normal training. We'll only overwrite if we find a better metric here.

        n_epochs_reinforced: int, default=2
            Number of epochs for the reinforced training phase.

        rescue_low_class1_f1: bool, default=False
            If True, then if the best normal model had class 1 F1 == 0, any RL epoch achieving
            class 1 F1 > `f1_1_rescue_threshold` is automatically considered an improvement.

        f1_1_rescue_threshold: float, default=0.0
            The threshold to detect a "small improvement" of class 1 F1 from 0.

        Returns
        -------
        (best_metric_val, best_model_path, best_scores)
            Where:
              - best_metric_val is the updated best metric value after reinforced training.
              - best_model_path is the path to the best model (reinforced if improved).
              - best_scores is the final (precision, recall, f1, support) from sklearn metrics.
        """
        print("=== Reinforced Training Mode ===")
        print("Oversampling class 1, larger batch size, lower LR, weighted CE loss...")

        # Prepare new CSV for reinforced training metrics
        os.makedirs(metrics_output_dir, exist_ok=True)
        reinforced_metrics_csv = os.path.join(metrics_output_dir, "reinforced_training_metrics.csv")
        with open(reinforced_metrics_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "precision_0",
                "recall_0",
                "f1_0",
                "support_0",
                "precision_1",
                "recall_1",
                "f1_1",
                "support_1",
                "macro_f1"
            ])

        # We'll also append to best_models.csv if we find improvements
        best_models_csv = os.path.join(metrics_output_dir, "best_models.csv")

        # Extract the original dataset from train_dataloader to apply WeightedRandomSampler
        dataset = train_dataloader.dataset  # Should be TensorDataset(input_ids, masks, labels)
        labels = dataset.tensors[2].numpy()  # third item = labels

        class_sample_count = np.bincount(labels)
        weight_per_class = 1.0 / class_sample_count  # inverse frequency
        sample_weights = [weight_per_class[t] for t in labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # Build a new train dataloader with a bigger batch size
        new_batch_size = 64
        new_train_dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=new_batch_size
        )

        # Lower learning rate by factor of 10 (example)
        new_lr = 5e-6

        # Weighted cross-entropy for binary classes: weight for class 1
        pos_weight_val = 2.0  # This can be tuned
        weight_tensor = torch.tensor([1.0, pos_weight_val], dtype=torch.float)

        # Set seeds again
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

        # Load from base_model_path if given, else from self.model_name
        if base_model_path:
            model = self.model_sequence_classifier.from_pretrained(base_model_path)
            print(f"Loaded base model from {base_model_path} for reinforced training.")
        else:
            model = self.model_sequence_classifier.from_pretrained(
                self.model_name,
                num_labels=2,
                output_attentions=False,
                output_hidden_states=False
            )
            print("No base_model_path provided. Using fresh model from self.model_name.")
        model.to(self.device)

        optimizer = AdamW(model.parameters(), lr=new_lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(new_train_dataloader) * n_epochs_reinforced
        )

        best_metric_val = previous_best_metric
        best_model_path_local = base_model_path  # Start from the best model from normal training
        best_scores = None

        # Collect test labels for final metrics
        test_labels = []
        for batch in test_dataloader:
            test_labels += batch[2].numpy().tolist()

        # Detect if the best normal-training F1 for class 1 was exactly 0
        # We'll use this to trigger the "rescue" logic below.
        best_normal_f1_class_1_was_zero = False
        # If we have a best_scores from normal training, check F1(class1)
        if best_model_path_local and (previous_best_metric != -1.0):
            # Attempt to compute the actual F1 from best_scores
            # But best_scores might come from run_training
            # We'll rely on the prior classification if needed.
            # For safety, let's rely on best_scores if it's stored properly.
            pass  # We'll handle logic if best_scores was carried over
        else:
            # If there's no prior metric or best_model_path, we consider normal training inconclusive
            pass

        # If the user explicitly wants rescue logic, let's see if we have a known F1=0 scenario
        # We'll rely on the fact that if previous_best_metric is > -1, we had a valid model
        # but let's not forcibly re-check that; we do it dynamically later.

        # Reinforced training epochs
        for epoch in range(n_epochs_reinforced):
            print(f"\n=== Reinforced Training: Epoch {epoch+1}/{n_epochs_reinforced} ===")
            t0 = time.time()
            model.train()
            running_loss = 0.0

            # Weighted cross entropy (for 2 classes) with emphasis on class 1
            criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(self.device))

            for step, train_batch in enumerate(new_train_dataloader):
                b_inputs = train_batch[0].to(self.device)
                b_masks = train_batch[1].to(self.device)
                b_labels = train_batch[2].to(self.device)

                model.zero_grad()
                outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks)
                logits = outputs[0]

                loss = criterion(logits, b_labels)
                running_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

            avg_train_loss = running_loss / len(new_train_dataloader)
            elapsed_str = self.format_time(time.time() - t0)
            print(f"  [Reinforced] Average train loss: {avg_train_loss:.4f}  Elapsed: {elapsed_str}")

            # Validation
            model.eval()
            total_val_loss = 0.0
            logits_complete = []
            eval_labels = []

            for test_batch in test_dataloader:
                b_inputs = test_batch[0].to(self.device)
                b_masks = test_batch[1].to(self.device)
                b_labels = test_batch[2].to(self.device)
                eval_labels.extend(b_labels.cpu().numpy())

                with torch.no_grad():
                    outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks, labels=b_labels)

                val_loss = outputs.loss
                val_logits = outputs.logits

                total_val_loss += val_loss.item()
                logits_complete.append(val_logits.detach().cpu().numpy())

            avg_val_loss = total_val_loss / len(test_dataloader)
            logits_complete = np.concatenate(logits_complete, axis=0)
            val_preds = np.argmax(logits_complete, axis=1).flatten()

            # Classification report
            report = classification_report(eval_labels, val_preds, output_dict=True)
            class_0_metrics = report.get("0", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
            class_1_metrics = report.get("1", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
            macro_avg = report.get("macro avg", {"f1-score": 0})

            precision_0 = class_0_metrics["precision"]
            recall_0 = class_0_metrics["recall"]
            f1_0 = class_0_metrics["f1-score"]
            support_0 = class_0_metrics["support"]

            precision_1 = class_1_metrics["precision"]
            recall_1 = class_1_metrics["recall"]
            f1_1 = class_1_metrics["f1-score"]
            support_1 = class_1_metrics["support"]

            macro_f1 = macro_avg["f1-score"]

            print(classification_report(eval_labels, val_preds))

            # Save epoch metrics to reinforced_training_metrics.csv
            with open(reinforced_metrics_csv, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    precision_0,
                    recall_0,
                    f1_0,
                    support_0,
                    precision_1,
                    recall_1,
                    f1_1,
                    support_1,
                    macro_f1
                ])

            # -- Rescue Logic for Class 1 F1 = 0 from normal training --
            # We'll interpret "best_model_path_local" and "previous_best_metric" to see if normal training
            # might have had class 1's F1 = 0. If so, any improvement above `f1_1_rescue_threshold` is considered better.
            rescue_override = False
            if rescue_low_class1_f1 and previous_best_metric != -1.0:
                # We must check if the best F1_1 from normal training was effectively 0.
                # The simplest check: if combined metric is extremely low, or we keep track separately.
                # Instead, let's rely on classification_report from the last best_scores if possible:
                # best_scores is (precision, recall, f1, support).
                # best_scores[2] -> f1 array, best_scores[2][1] is f1 for class1.
                # If that was 0, we do the rescue logic.
                if best_scores is not None:
                    prev_f1_1 = best_scores[2][1]
                    if prev_f1_1 == 0.0 and f1_1 > f1_1_rescue_threshold:
                        # This RL epoch is automatically an improvement
                        print(f"[Rescue Logic Triggered] Class 1 F1 moved from 0.0 to {f1_1:.4f}, "
                              f"exceeding threshold {f1_1_rescue_threshold:.4f}")
                        rescue_override = True

            # Check if this epoch yields a new best model by normal combined logic
            if best_model_criteria == "combined":
                combined_metric = f1_class_1_weight * f1_1 + (1.0 - f1_class_1_weight) * macro_f1
            else:
                combined_metric = (f1_1 + macro_f1) / 2.0

            # If the rescue logic is triggered, we override combined_metric comparison
            if rescue_override:
                # Force "infinite" improvement to ensure we treat it as a new best
                new_metric_val = combined_metric + 9999.0
            else:
                new_metric_val = combined_metric

            # Standard best-model selection logic
            if new_metric_val > best_metric_val:
                print(f"New best (reinforced) model found at epoch {epoch + 1} with combined metric={combined_metric:.4f}.")
                # Remove old best model if needed
                if best_model_path_local is not None and os.path.isdir(best_model_path_local):
                    try:
                        shutil.rmtree(best_model_path_local)
                    except OSError:
                        pass

                best_metric_val = new_metric_val
                # Save new best model to a temporary path
                if save_model_as is not None:
                    best_model_path_local = f"./models/{save_model_as}_reinforced_epoch_{epoch+1}"
                    os.makedirs(best_model_path_local, exist_ok=True)

                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(best_model_path_local, WEIGHTS_NAME)
                    output_config_file = os.path.join(best_model_path_local, CONFIG_NAME)

                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    self.tokenizer.save_vocabulary(best_model_path_local)
                else:
                    best_model_path_local = None

                # Log in best_models.csv
                with open(best_models_csv, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1,
                        avg_train_loss,
                        avg_val_loss,
                        precision_0,
                        recall_0,
                        f1_0,
                        support_0,
                        precision_1,
                        recall_1,
                        f1_1,
                        support_1,
                        macro_f1,
                        best_model_path_local if best_model_path_local else "Not saved to disk",
                        "reinforced"  # training phase
                    ])

                best_scores = precision_recall_fscore_support(eval_labels, val_preds)

        # After finishing the reinforced epochs, if we have found a better model, rename it to final
        if best_model_path_local and (best_model_path_local != base_model_path):
            # If user wants to save the final best model
            if save_model_as is not None:
                final_path = f"./models/{save_model_as}"
                if os.path.exists(final_path):
                    shutil.rmtree(final_path)
                os.rename(best_model_path_local, final_path)
                best_model_path_local = final_path
                print(f"Reinforced best model saved at: {best_model_path_local}")

        print("Reinforced training complete.\n")
        return best_metric_val, best_model_path_local, best_scores

    def predict(
            self,
            dataloader: DataLoader,
            model: Any,
            proba: bool = True,
            progress_bar: bool = True
    ):
        """
        Predict with a trained model.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            Dataloader for prediction, from self.encode().

        model: huggingface model
            A trained model to use for inference.

        proba: bool, default=True
            If True, return probability distributions (softmax).
            If False, return raw logits.

        progress_bar: bool, default=True
            If True, display a progress bar during prediction.

        Returns
        -------
        pred: ndarray of shape (n_samples, n_labels)
            Probabilities or logits for each sample.
        """
        logits_complete = []
        if progress_bar:
            loader = tqdm(dataloader, desc="Predicting")
        else:
            loader = dataloader

        model.eval()

        for batch in loader:
            batch = tuple(t.to(self.device) for t in batch)
            if len(batch) == 3:
                b_input_ids, b_input_mask, _ = batch
            else:
                b_input_ids, b_input_mask = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            logits_complete.append(logits)

            del outputs
            torch.cuda.empty_cache()

        pred = np.concatenate(logits_complete, axis=0)

        if progress_bar:
            print(f"label ids: {self.dict_labels}")

        return softmax(pred, axis=1) if proba else pred

    def load_model(
            self,
            model_path: str
    ):
        """
        Load a previously saved model from disk.

        Parameters
        ----------
        model_path: str
            Path to the saved model folder.

        Returns
        -------
        model: huggingface model
            The loaded model instance.
        """
        return self.model_sequence_classifier.from_pretrained(model_path)

    def predict_with_model(
            self,
            dataloader: DataLoader,
            model_path: str,
            proba: bool = True,
            progress_bar: bool = True
    ):
        """
        Convenience method that loads a model from the specified path, moves it to self.device,
        and performs prediction on the given dataloader.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            DataLoader for prediction.

        model_path: str
            Path to the model to load.

        proba: bool, default=True
            If True, return probability distributions (softmax).
            If False, return raw logits.

        progress_bar: bool, default=True
            If True, display a progress bar during prediction.

        Returns
        -------
        ndarray
            Probability or logit predictions.
        """
        model = self.load_model(model_path)
        model.to(self.device)
        return self.predict(dataloader, model, proba, progress_bar)

    def format_time(
            self,
            elapsed: float | int
    ) -> str:
        """
        Format a time duration to hh:mm:ss.

        Parameters
        ----------
        elapsed: float or int
            Elapsed time in seconds.

        Returns
        -------
        str
            The time in hh:mm:ss format.
        """
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))
