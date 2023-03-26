"""Classification model for enron_spam"""
import os 
import pandas as pd
from transformers import AutoConfig, BertForSequenceClassification
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import BertConfig, AutoModelForSequenceClassification

import random
import numpy as np
import torch

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
      return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_model():
    tokenizer = AutoTokenizer.from_pretrained("./data/models/wiki_pl")
    model = AutoModelForSequenceClassification.from_pretrained(
        "./data/models/wiki_pl", num_labels=2,
        ignore_mismatched_sizes=True
    )

    test = pd.read_json(f"data/preprocessed/poleval/test.jsonl", lines=True)
    train = pd.read_json(f"data/preprocessed/poleval/train.jsonl", lines=True)
    y_test = [0 if y == "normal" else 1 for y in test["label"]]
    y_train = [0 if y == "normal" else 1 for y in train["label"]]
    x_test = test["text"].tolist()
    x_train = train["text"].tolist()

    train_encodings = tokenizer(
	x_train, truncation=True, padding=True, max_length=512
    )
    train_dataset = Dataset(train_encodings, y_train)

    test_encodings = tokenizer(
	x_test, truncation=True, padding=True, max_length=512
    )
    test_dataset = Dataset(test_encodings, y_test)

    training_args = TrainingArguments(
	output_dir='./tmp',
	num_train_epochs=100,
	warmup_steps=100,
	weight_decay=0.01,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
	logging_dir='./tmp/logs',
	logging_steps=500,
	save_steps=500,
        save_total_limit=10,
        learning_rate=1e-5,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
    )
    trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	compute_metrics=compute_metrics,
    )
    trainer.train()
    return model, tokenizer


def train_model_old():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    config = AutoConfig.from_pretrained(
        "mrm8488/bert-tiny-finetuned-enron-spam-detection"
    )
    config.update({"vocab_size": tokenizer.vocab_size})

    model = BertForSequenceClassification(config)
    test = pd.read_json(f"data/preprocessed/poleval/test.jsonl", lines=True)
    train = pd.read_json(f"data/preprocessed/poleval/train.jsonl", lines=True)
    y_test = [0 if y == "normal" else 1 for y in test["label"]]
    y_train = [0 if y == "normal" else 1 for y in train["label"]]
    x_test = test["text"].tolist()
    x_train = train["text"].tolist()

    train_encodings = tokenizer(
	x_train, truncation=True, padding=True, max_length=512
    )
    train_dataset = Dataset(train_encodings, y_train)

    test_encodings = tokenizer(
	x_test, truncation=True, padding=True, max_length=512
    )
    test_dataset = Dataset(test_encodings, y_test)

    training_args = TrainingArguments(
	output_dir='./tmp',
	num_train_epochs=250,
	warmup_steps=100,
	weight_decay=0.01,
	logging_dir='./tmp/logs',
	logging_steps=1000,
	save_steps=1000,
        save_total_limit=10,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
    )
    trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	compute_metrics=compute_metrics,
    )
    trainer.train()
    return model, tokenizer


def get_model_and_tokenizer():
    model_path = "./data/models/poleval/"
    if not os.path.exists(model_path + "config.json"):
        model, tokenizer = train_model()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.config.id2label = {0: "normal", 1: "hate"}
    return model, tokenizer


def get_classify_function(device="cpu"):
    model, tokenizer = get_model_and_tokenizer()
    model.eval()
    model = model.to(device)

    def fun(texts):
        logits = list()
        i = 0
        for chunk in tqdm(
            [texts[pos:pos + 128] for pos in range(0, len(texts), 128)]
        ):
            encoded_inputs = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            with torch.no_grad():
                logits.append(model(**encoded_inputs).logits.cpu())
        logits = torch.cat(logits, dim=0)
        pred_y = torch.argmax(logits, dim=1).tolist()
        pred_y = [model.config.id2label[p] for p in pred_y]
        return pred_y

    return fun
