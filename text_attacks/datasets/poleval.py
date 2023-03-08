"""Download and preprecess poleval"""
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def download_dataset():
    dataset = load_dataset("poleval2019_cyberbullying", "task01")
    train = pd.DataFrame(dataset["train"].to_dict())
    test = pd.DataFrame(dataset["test"].to_dict())

    train["id"] = list(range(len(train)))
    train["label"] = [
        "hate" if lab == 1 else "normal" for lab in train["label"]
    ]

    test["id"] = list(range(len(test)))
    test["label"] = [
        "hate" if lab == 1 else "normal" for lab in test["label"]
    ]
    adversarial, test = train_test_split(
        test,
        test_size=0.9,
        stratify=test["label"]
    )

    return train, test, adversarial
