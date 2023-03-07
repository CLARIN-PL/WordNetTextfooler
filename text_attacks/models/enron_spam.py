"""Classification model for enron_spam"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "mrm8488/bert-tiny-finetuned-enron-spam-detection"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "mrm8488/bert-tiny-finetuned-enron-spam-detection"
    )
    model.config.id2label = {0: "ham", 1: "spam"}
    return model, tokenizer
