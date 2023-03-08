"""Classification model for enron_spam"""
import os

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_model_and_tokenizer():
    model_path = "data/models/endron_spam"
    if not os.path.exists(model_path):
        model_path = "mrm8488/bert-tiny-finetuned-enron-spam-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.config.id2label = {0: "ham", 1: "spam"}
    return model, tokenizer


def get_classify_function():
    model, tokenizer = get_model_and_tokenizer()

    def fun(texts):
        encoded_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        logits = model(**encoded_inputs).logits
        pred_y = torch.argmax(logits, dim=1).tolist()
        pred_y = [model.config.id2label[p] for p in pred_y]
        return pred_y

    return fun
