"""Script for downloading and converting datasets."""
from pathlib import Path

import click

from text_attacks.utils import download_dataset


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

    train, test, adversarial = download_dataset(dataset_name)
    train.to_json(output_dir / "train.jsonl", orient="records", lines=True)
    test.to_json(output_dir / "test.jsonl", orient="records", lines=True)
    adversarial.to_json(
        output_dir / "adversarial.jsonl",
        orient="records",
        lines=True
    )


if __name__ == "__main__":
    main()

