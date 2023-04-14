"""XAI results."""
import os
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
import shap
import torch
from tqdm import tqdm

from text_attacks.utils import get_model_and_tokenizer


def build_predict_fun(model, tokenizer):
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model = model.to(device)

    def f(x):
        encoded_inputs = torch.tensor(
            [tokenizer.encode(
                v, padding="max_length", max_length=512, truncation=True
            ) for v in x])
        encoded_inputs = encoded_inputs.to(device)
        with torch.no_grad():
            logits = model(encoded_inputs).logits.cpu()
        return logits
    return f


def get_importance(shap_values):
    cohorts = {"": shap_values}
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            cohort_exps[i] = cohort_exps[i].abs.mean(0)
    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    values = np.array(
        [cohort_exps[i].values for i in range(len(cohort_exps))],
        dtype=object,
    )
    feature_importance = pd.DataFrame(
        list(
            zip(feature_names, sum(values))
        ),
        columns=['features', 'importance']
    )
    feature_importance.sort_values(
        by=['importance'], ascending=False, inplace=True, key=lambda x: abs(x)
    )
    return feature_importance


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
    test = pd.read_json(
        f"data/preprocessed/{dataset_name}/adversarial.jsonl", lines=True
    )
    test_x = test["text"].tolist()
    test_x = [
        tokenizer.decode(
            tokenizer.encode(
                t, padding="do_not_pad", max_length=512, truncation=True
            ),
            skip_special_tokens=True
        ) for t in test_x
    ]

    predict = build_predict_fun(model, tokenizer)
    explainer = shap.Explainer(
        predict,
        masker=tokenizer,
        output_names=list(model.config.id2label.values())
    )
    shap_values = explainer(test_x)
    with open(output_dir / "shap_values.pickle", mode="wb") as fd:
        pickle.dump(shap_values, fd)

    # GLOBAL IMPORTANCE:
    os.makedirs(output_dir / "global", exist_ok=True)
    for class_id, class_name in model.config.id2label.items():
        importance_df = get_importance(shap_values[:, :, class_id].mean(0))
        class_name = class_name.replace("/", "_")
        importance_df.to_json(
            output_dir / "global" / f"{class_name}__importance.json",
        )

    # LOCAL IMPORTANCE
    for class_id, class_name in model.config.id2label.items():
        class_name = class_name.replace("/", "_")
        sub_dir = output_dir / "local" / "adversarial" /class_name
        os.makedirs(sub_dir, exist_ok=True)
        for shap_id, text_id in enumerate(test["id"]):
            importance_df = get_importance(shap_values[shap_id, :, class_id])
            importance_df.to_json(
                sub_dir / f"{text_id}__importance.json",
            )
    
    # LOCAL IMPORTANCE (test set)
    test = pd.read_json(
        f"data/preprocessed/{dataset_name}/test.jsonl", lines=True
    )
    test_x = test["text"].tolist()
    test_x = [
        tokenizer.decode(
            tokenizer.encode(
                t, padding="do_not_pad", max_length=512, truncation=True
            ),
            skip_special_tokens=True
        ) for t in test_x
    ]

    predict = build_predict_fun(model, tokenizer)
    explainer = shap.Explainer(
        predict,
        masker=tokenizer,
        output_names=list(model.config.id2label.values())
    )
    for text_id, text in tqdm(
        zip(test["id"], test_x),
        total=len(test_x),
        desc="Shap for test DS",
    ):
        shap_values = explainer([text])
        for class_id, class_name in model.config.id2label.items():
            sub_dir = output_dir / "local" / "test" / class_name
            os.makedirs(sub_dir, exist_ok=True)
            importance_df = get_importance(shap_values[0, :, class_id])
            importance_df.to_json(
                sub_dir / f"{text_id}__importance.json",
            )

if __name__ == "__main__":
    main()
