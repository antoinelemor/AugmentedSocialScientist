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
            a model name from huggingface models: https://huggingface.co/models

        tokenizer: huggingface tokenizer, default=BertTokenizer.from_pretrained('bert-base-cased')
            tokenizer to use

        model_sequence_classifier: huggingface sequence classifier, default=BertForSequenceClassification
            a huggingface sequence classifier that implements a from_pretrained() function

        device: torch.Device, default=None
            device to use. If None, automatically set if presence of GPU is detected. CPU otherwise.
        """
        self.model_name = model_name
        self.tokenizer = tokenizer.from_pretrained(self.model_name)
        self.model_sequence_classifier = model_sequence_classifier
        self.dict_labels = None

        # Users can set their device (flexibility),
        # but device is set by default in case users are not familiar with PyTorch
        self.device = device
        if self.device is None:
            # If CUDA is available, use it
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print('There are %d GPU(s) available.' % torch.cuda.device_count())
                print('We will use GPU {}:'.format(torch.cuda.current_device()),
                      torch.cuda.get_device_name(torch.cuda.current_device()))
            # If MPS is available, use it (Apple Silicon GPU)
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print('MPS is available. Using the Apple Silicon GPU!')
            else:
                # Otherwise, fall back to CPU
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
        Preprocessing of the training, test or prediction data.
        The function will:
            (1) tokenize the sequences and map tokens to their IDs;
            (2) truncate or pad to 512 tokens (limit for BERT), create corresponding attention masks;
            (3) return a PyTorch dataloader object containing token ids, labels and attention masks.

        Parameters
        ----------
        sequences: 1D array-like
            list of texts

        labels: 1D array-like or None, default=None
            list of labels. None for unlabelled prediction data

        batch_size: int, default=32
            batch size for PyTorch DataLoader

        progress_bar: bool, default=True
            if True, print progress bar for the processing

        add_special_tokens: bool, default=True
            if True, add '[CLS]' and '[SEP]' tokens

        Returns
        -------
        dataloader: torch.utils.data.DataLoader
            PyTorch DataLoader object containing token ids, labels and attention masks
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

        max_len = min(max(len(sen) for sen in input_ids), 512)

        # Pad the input tokens with value 0 and truncate to max_len
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

        if labels is None:
            # Convert to PyTorch tensors
            inputs_tensors = torch.tensor(input_ids)
            masks_tensors = torch.tensor(attention_masks)

            data = TensorDataset(inputs_tensors, masks_tensors)
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
            return dataloader
        else:
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
            reinforced_n_epochs: int = 2
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Train, evaluate and optionally save a BERT model. 
        Additionally, this method:
          - Logs training and validation metrics per epoch in a CSV file.
          - Saves a model checkpoint each epoch if it outperforms the previous best model.
          - Keeps only the best model (removes the previous best) based on a selection metric
            that favors class 1's F1 score and also considers the macro F1 (you can tune the weights).
          - Logs all kept models in a separate CSV with their exact metrics.
          - At the end, if the final best model's F1 on class 1 is below 0.6 **and** `reinforced_learning=True`,
            triggers a reinforced training pass for `reinforced_n_epochs` epochs with the same best-model selection
            approach, oversampling, weighted loss, etc.

        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            training dataloader obtained with self.encode()

        test_dataloader: torch.utils.data.DataLoader
            test dataloader obtained with self.encode()

        n_epochs: int, default=3
            number of epochs for the main training

        lr: float, default=5e-5
            learning rate for the main training

        random_state: int, default=42
            random state (for reproducibility)

        save_model_as: str, default=None
            name of the model to save. The final best model will be saved under ./models/<model_name>.
            If None, models are not saved to disk (though metrics CSVs will still be generated).

        pos_weight: torch.Tensor, default=None
            if not None, weights the loss to favor certain classes more heavily 
            (commonly used in binary classification).

        metrics_output_dir: str, default="./training_logs"
            directory where CSV files of metrics will be stored.

        best_model_criteria: str, default="combined"
            selection criterion for the best model. Currently only "combined" is implemented,
            which uses a weighted combination of class 1's F1 score and the macro F1.

        f1_class_1_weight: float, default=0.7
            weight given to class 1's F1 in the combined metric. The rest (1 - weight) is for the macro F1.

        reinforced_learning: bool, default=False
            if True, and if after the main training the best model's F1 on class 1 is < 0.6, 
            a reinforced training procedure will be triggered.

        reinforced_n_epochs: int, default=2
            number of epochs to perform during the reinforced learning phase if triggered.

        Returns
        -------
        scores: tuple (precision, recall, f1-score, support)
            final evaluation scores from sklearn.metrics.precision_recall_fscore_support, 
            for each label. shape: (4, n_labels)

        Notes
        -----
        The method also creates:
            - "<metrics_output_dir>/training_metrics.csv": logs all metrics for each epoch of the main training.
            - "<metrics_output_dir>/best_models.csv": logs the checkpoint of any new best model found during the main training.
            - If reinforced learning is triggered, additional epochs are appended to these files
              (with a "reinforced_phase" marker in best_models.csv and in the training metrics).
        """
        # Ensure output directory for metrics exists
        os.makedirs(metrics_output_dir, exist_ok=True)
        training_metrics_csv = os.path.join(metrics_output_dir, "training_metrics.csv")
        best_models_csv = os.path.join(metrics_output_dir, "best_models.csv")

        # Initialize CSV files with headers (if they don't exist yet)
        # We will append to them if they do exist (e.g., from a previous run).
        if not os.path.isfile(training_metrics_csv):
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
                    "macro_f1",
                    "reinforced_phase"
                ])
        if not os.path.isfile(best_models_csv):
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
                    "reinforced_phase"
                ])

        # Unpack all test labels for evaluation
        test_labels = []
        for batch in test_dataloader:
            test_labels += batch[2].numpy().tolist()
        num_labels = np.unique(test_labels).size

        if self.dict_labels is None:
            label_names = None
        else:
            # label_names sorted by the index in self.dict_labels
            label_names = [str(x[0]) for x in sorted(self.dict_labels.items(), key=lambda x: x[1])]

        # Set seed for reproducibility
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

        # Load the base model
        model = self.model_sequence_classifier.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        model.to(self.device)

        # Prepare the optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * n_epochs
        )

        best_metric_val = -1.0
        best_model_path = None
        best_scores = None  # Will store final best (prec, rec, f1, support)

        # =====================
        #   MAIN TRAINING
        # =====================
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

                # Weighted loss if pos_weight is specified (for CrossEntropyLoss in multi-class)
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
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training took: {:}".format(self.format_time(time.time() - t0)))

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
                logits = logits.detach().cpu().numpy()
                logits_complete.append(logits)

            logits_complete = np.concatenate(logits_complete, axis=0)
            avg_val_loss = total_val_loss / len(test_dataloader)
            print("")
            print("  Average validation loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(self.format_time(time.time() - t0)))

            preds = np.argmax(logits_complete, axis=1).flatten()
            report = classification_report(test_labels, preds, target_names=label_names, output_dict=True)

            # Extract class 0 and class 1 metrics
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

            # Print classification report to console
            print(classification_report(test_labels, preds, target_names=label_names))

            # Append epoch metrics to training_metrics.csv
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
                    macro_f1,
                    "normal"
                ])

            # Compute "combined" metric for best model selection
            if best_model_criteria == "combined":
                combined_metric = f1_class_1_weight * f1_1 + (1.0 - f1_class_1_weight) * macro_f1
            else:
                combined_metric = (f1_1 + macro_f1) / 2.0

            if combined_metric > best_metric_val:
                print(f"New best model found at epoch {i_epoch + 1} with combined metric={combined_metric:.4f}.")
                # Remove the old best model if it existed
                if best_model_path is not None:
                    try:
                        shutil.rmtree(best_model_path)
                    except OSError:
                        pass

                best_metric_val = combined_metric

                if save_model_as is not None:
                    # Save to epoch-specific folder
                    best_model_path = f"./models/{save_model_as}_epoch_{i_epoch + 1}"
                    os.makedirs(best_model_path, exist_ok=True)

                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(best_model_path, WEIGHTS_NAME)
                    output_config_file = os.path.join(best_model_path, CONFIG_NAME)

                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    self.tokenizer.save_vocabulary(best_model_path)
                else:
                    best_model_path = None  # Not saving to disk if user didn't specify

                # Log this new best model in best_models.csv
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
                        "normal"
                    ])

                # Store final best metrics for return
                best_scores = precision_recall_fscore_support(test_labels, preds)

        # End of main training
        print("")
        print("Main training complete!")

        # Rename best model path to final user path if save_model_as is provided
        if save_model_as is not None and best_model_path is not None:
            final_path = f"./models/{save_model_as}"
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            os.rename(best_model_path, final_path)
            best_model_path = final_path
            print(f"Best model is available at: {best_model_path}")

        # Check if we need to do reinforced learning
        # (Only if best_f1_1 < 0.6 and user has reinforced_learning=True)
        if best_scores is not None:
            best_f1_1 = best_scores[2][1]  # f1 array, index 1 for class '1'
            if best_f1_1 < 0.6 and reinforced_learning:
                print(f"\nThe best model's F1 score for class 1 ({best_f1_1:.3f}) is below 0.60.")
                print("Triggering reinforced training...")

                # Perform the reinforced training pass with the same best-model selection approach
                self.run_training_reinforced(
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    base_model_path=best_model_path if best_model_path else None,
                    random_state=random_state,
                    metrics_output_dir=metrics_output_dir,
                    save_model_as=save_model_as,
                    best_metric_val=best_metric_val,
                    best_scores=best_scores,
                    best_models_csv=best_models_csv,
                    training_metrics_csv=training_metrics_csv,
                    f1_class_1_weight=f1_class_1_weight,
                    best_model_criteria=best_model_criteria,
                    reinforced_n_epochs=reinforced_n_epochs
                )
        else:
            print("No best scores found (unexpected). No reinforced training triggered.")

        return best_scores

    def run_training_reinforced(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        base_model_path: str | None,
        random_state: int,
        metrics_output_dir: str,
        save_model_as: str | None,
        best_metric_val: float,
        best_scores: Tuple[Any, Any, Any, Any],
        best_models_csv: str,
        training_metrics_csv: str,
        f1_class_1_weight: float,
        best_model_criteria: str,
        reinforced_n_epochs: int
    ):
        """
        A "reinforced training" procedure that is triggered if the final best model
        has a poor F1 score (< 0.6) for class 1 and reinforced learning is activated.
        This method:
           - Loads the current best model.
           - Uses oversampling of class 1 (WeightedRandomSampler).
           - Increases batch size (default = 64).
           - Decreases the learning rate (e.g., 5e-6).
           - Uses a weighted cross-entropy loss, placing more weight on class 1.
           - Performs 'reinforced_n_epochs' epochs of training with the same best-model selection 
             logic as in the main training. Logs everything in the same CSV files, 
             with a 'reinforced_phase' marker.

        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            original training dataloader

        test_dataloader: torch.utils.data.DataLoader
            original test dataloader

        base_model_path: str or None
            path to the best model so far

        random_state: int
            random state for reproducibility

        metrics_output_dir: str
            directory to store CSV logs

        save_model_as: str or None
            name of the model to save under ./models/<model_name>

        best_metric_val: float
            the best combined metric value so far

        best_scores: tuple
            the best scores so far (precision, recall, f1, support)

        best_models_csv: str
            path to the CSV file that logs best models

        training_metrics_csv: str
            path to the CSV file that logs per-epoch metrics

        f1_class_1_weight: float
            weight for class 1's F1 when computing the combined metric

        best_model_criteria: str
            either "combined" or another strategy if implemented

        reinforced_n_epochs: int
            number of epochs to train in reinforced mode
        """
        # Re-seed everything for reproducibility
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

        # Load the best model so far (if path is provided), else load a fresh model
        if base_model_path:
            model = self.model_sequence_classifier.from_pretrained(base_model_path)
            print(f"Loaded base model from {base_model_path} for reinforced training.")
        else:
            # Fallback if no path is given
            model = self.model_sequence_classifier.from_pretrained(
                self.model_name,
                num_labels=2,
                output_attentions=False,
                output_hidden_states=False
            )
            print("No base_model_path provided. Using fresh model from self.model_name.")

        model.to(self.device)

        # Create a WeightedRandomSampler for oversampling
        dataset = train_dataloader.dataset
        labels = dataset.tensors[2].numpy()
        class_sample_count = np.bincount(labels)  # count of each class
        weight_per_class = 1.0 / class_sample_count
        sample_weights = [weight_per_class[t] for t in labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # Reinforced training hyperparams
        new_lr = 5e-6  # smaller LR
        new_batch_size = 64
        pos_weight_value = 2.0  # extra weight on class 1

        new_train_dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=new_batch_size
        )

        optimizer = AdamW(model.parameters(), lr=new_lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(new_train_dataloader) * reinforced_n_epochs
        )

        # We need the test labels again for final evaluation
        test_labels = []
        for batch in test_dataloader:
            test_labels += batch[2].numpy().tolist()

        print("=== Reinforced Training Mode ===")
        print("Oversampling class 1, bigger batch size, lower LR, weighted loss...")

        # Start the reinforced training loop
        for epoch in range(reinforced_n_epochs):
            print(f"\n=== Reinforced Epoch {epoch + 1}/{reinforced_n_epochs} ===")

            t0 = time.time()
            model.train()
            running_loss = 0.0

            # Weighted cross entropy for class imbalance
            weight_tensor = torch.tensor([1.0, pos_weight_value], device=self.device)
            criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)

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
            print(f"  [Reinforced] Average train loss: {avg_train_loss:.4f}  Elapsed: {self.format_time(time.time() - t0)}")

            # Validation
            model.eval()
            total_val_loss = 0.0
            logits_complete = []

            for test_batch in test_dataloader:
                b_inputs = test_batch[0].to(self.device)
                b_masks = test_batch[1].to(self.device)
                b_labels = test_batch[2].to(self.device)

                with torch.no_grad():
                    outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks, labels=b_labels)

                val_loss = outputs.loss
                val_logits = outputs.logits

                total_val_loss += val_loss.item()
                logits_complete.append(val_logits.detach().cpu().numpy())

            avg_val_loss = total_val_loss / len(test_dataloader)
            logits_complete = np.concatenate(logits_complete, axis=0)
            val_preds = np.argmax(logits_complete, axis=1).flatten()

            # Compute metrics
            report = classification_report(test_labels, val_preds, output_dict=True)
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
            print(classification_report(test_labels, val_preds))

            # Log metrics to the same training_metrics.csv, marked as reinforced
            with open(training_metrics_csv, mode='a', newline='', encoding='utf-8') as f:
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
                    "reinforced"
                ])

            # Best model selection logic (same as main training)
            if best_model_criteria == "combined":
                combined_metric = f1_class_1_weight * f1_1 + (1.0 - f1_class_1_weight) * macro_f1
            else:
                combined_metric = (f1_1 + macro_f1) / 2.0

            if combined_metric > best_metric_val:
                print(f"New best model found (reinforced) at epoch {epoch + 1} with combined metric={combined_metric:.4f}.")
                # Remove the old best model if it existed
                # (only if we had a path - might have been None if not saving previously)
                if base_model_path is not None and os.path.exists(base_model_path):
                    try:
                        shutil.rmtree(base_model_path)
                    except OSError:
                        pass

                best_metric_val = combined_metric

                # Save new best model
                if save_model_as is not None:
                    # Save to epoch-specific (reinforced) folder
                    best_model_path = f"./models/{save_model_as}_reinforced_epoch_{epoch + 1}"
                    os.makedirs(best_model_path, exist_ok=True)

                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(best_model_path, WEIGHTS_NAME)
                    output_config_file = os.path.join(best_model_path, CONFIG_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    self.tokenizer.save_vocabulary(best_model_path)
                else:
                    best_model_path = None

                # Log best model info
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
                        best_model_path if best_model_path else "Not saved to disk",
                        "reinforced"
                    ])

                best_scores = precision_recall_fscore_support(test_labels, val_preds)

        # After reinforced training
        print("\nReinforced training complete.")

        # If we ended up with a new best model path, rename it to the final user path
        if save_model_as is not None and best_model_path is not None:
            final_path = f"./models/{save_model_as}"
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            os.rename(best_model_path, final_path)
            print(f"[Reinforced] Best model is now available at: {final_path}")

    def predict(
            self,
            dataloader: DataLoader,
            model: Any,
            proba: bool = True,
            progress_bar: bool = True
    ):
        """
        Prediction with a trained model.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            prediction dataloader obtained with self.encode()

        model: huggingface model
            trained model

        proba: bool, default=True
            if True, return prediction probabilities; else, return logits

        progress_bar: bool, default=True
            if True, print progress bar of prediction

        Returns
        -------
        pred: ndarray of shape (n_samples, n_labels)
            probabilities for each sequence (row) of belonging to each category (column)
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
        Load a saved model.

        Parameters
        ----------
        model_path: str
            path to the saved model

        Returns
        -------
        model: huggingface model
            loaded model
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
        A convenience method that loads a model from a given path,
        moves it to the same device as self.device,
        and performs prediction on the provided dataloader.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            dataloader obtained with self.encode()

        model_path: str
            path to the saved model

        proba: bool, default=True
            if True, return prediction probabilities; else, return logits

        progress_bar: bool, default=True
            if True, show a progress bar for prediction

        Returns
        -------
        predictions: ndarray
            probabilities or logits for each sample in the dataloader
        """
        model = self.load_model(model_path)
        model.to(self.device)
        return self.predict(dataloader, model, proba, progress_bar)

    def format_time(
            self,
            elapsed: float | int
    ):
        """
        Format a time to hh:mm:ss.

        Parameters
        ----------
        elapsed: float or int
            elapsed time in seconds

        Returns
        -------
        str
            time in hh:mm:ss format
        """
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))
