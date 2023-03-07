"""Script for downloading and converting datasets."""
from pathlib import Path

import click
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def convert(dataset):
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


@click.command()
@click.option(
    "--dataset_name",
    help="Dataset name",
    type=str,
)
@click.option(
    "--output_dir",
    help="Path to output directory",
    type=click.Path(path_type=Path),
)
def main(
    dataset_name: str,
    output_dir: Path,
):
    """Downloads the dataset to the output directory."""
    dataset_mappings = {
        "enron_spam": "SetFit/enron_spam",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_mappings[dataset_name])
    train, test, adversarial = convert(dataset)
    train.to_json(output_dir / "train.jsonl", orient="records", lines=True)
    test.to_json(output_dir / "test.jsonl", orient="records", lines=True)
    adversarial.to_json(
        output_dir / "adversarial.jsonl",
        orient="records",
        lines=True
    )


if __name__ == "__main__":
    main()

