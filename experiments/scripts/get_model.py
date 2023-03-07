"""Downloads pretrained model from huggingface or trains new one."""
from pathlib import Path

import click

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
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()

