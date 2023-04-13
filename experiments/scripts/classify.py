"""Classification results."""
from pathlib import Path

import click
import pandas as pd
import torch
from sklearn.metrics import classification_report

from text_attacks.utils import get_classify_function


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
    """Classifies the test data and saves results to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    classify = get_classify_function(
        dataset_name=dataset_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    test = pd.read_json(f"data/reduced/{dataset_name}/test.jsonl", lines=True)
    test_x = test["text"].tolist()
    test_y = test["label"]
    pred_y = classify(test_x)

    with open(output_dir / "metrics.txt", mode="wt") as fd:
        fd.write(classification_report(test_y, pred_y))

    test["pred_label"] = pred_y
    test.to_json(output_dir / "test.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
