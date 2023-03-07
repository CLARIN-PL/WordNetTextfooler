"""Classification results."""
from pathlib import Path

import click
import pandas as pd
import torch
from sklearn.metrics import classification_report

from text_attacks.utils import get_model_and_tokenizer


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
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = get_model_and_tokenizer(
        dataset_name=dataset_name,
    )
    test = pd.read_json(f"data/datasets/{dataset_name}/test.jsonl", lines=True)
    test_x = test["text"].tolist()
    test_y = test["label"]
    encoded_inputs = tokenizer(
        test_x,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    logits = model(**encoded_inputs).logits
    pred_y = torch.argmax(logits, dim=1).tolist()
    pred_y = [model.config.id2label[p] for p in pred_y]

    with open(output_dir / "metrics.txt", mode="wt") as fd:
        fd.write(classification_report(test_y, pred_y))


if __name__ == "__main__":
    main()
