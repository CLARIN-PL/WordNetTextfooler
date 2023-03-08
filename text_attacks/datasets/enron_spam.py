"""Classification model for enron_spam"""
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def download_dataset():
    dataset = load_dataset("SetFit/enron_spam")
    train = pd.DataFrame(dataset["train"].to_dict())
    test = pd.DataFrame(dataset["test"].to_dict())

    train["label"] = train["label_text"]
    train = train.rename(columns={"message_id": "id"})
    train = train.drop(columns=["label_text", "subject", "message", "date"])

    test["label"] = test["label_text"]
    test = test.rename(columns={"message_id": "id"})
    test = test.drop(columns=["label_text", "subject", "message", "date"])
    adversarial, test = train_test_split(
        test,
        test_size=0.9,
        stratify=test["label"]
    )

    return train, test, adversarial
