"""XAI results."""
import pickle
from pathlib import Path

import click
import pandas as pd
import shap
import torch

from text_attacks.utils import get_model_and_tokenizer


def build_predict_fun(model, tokenizer):
    def f(x):
        encoded_inputs = torch.tensor(
            [tokenizer.encode(
                v, padding='max_length', max_length=512, truncation=True
            ) for v in x])
        logits = model(encoded_inputs).logits
        return logits

    return f


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
    test = pd.read_json(f"data/preprocessed/{dataset_name}/adversarial.jsonl", lines=True)
    test_x = test["text"].tolist()

    predict = build_predict_fun(model, tokenizer)
    explainer = shap.Explainer(
        predict,
        masker=tokenizer,
        output_names=list(model.config.id2label.values())
    )
    shap_values = explainer(test_x)
    with open(output_dir / "shap_values.pickle", mode="wb") as fd:
        pickle.dump(shap_values, fd)


if __name__ == "__main__":
    main()
