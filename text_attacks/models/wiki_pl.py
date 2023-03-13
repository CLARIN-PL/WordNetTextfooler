"""Classification model for enron_spam"""
import os

import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_model_and_tokenizer():
    model_path = "./data/models/wiki_pl"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer


def get_classify_function(device="cpu"):
    model, tokenizer = get_model_and_tokenizer()
    model.eval()
    model = model.to(device)

    def fun(texts):
        logits = list()
        i = 0
        for chunk in tqdm(
            [texts[pos:pos + 256] for pos in range(0, len(texts), 256)]
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
