"""Script for running attacks on datasets."""
import click
import pandas as pd
import os
from tqdm import tqdm
from text_attacks.utils import get_classify_function
from textfooler import Attack, TextFooler, BaseLine, process


TEXT = "text"
LEMMAS = "lemmas"
TAGS = "tags"
ORTHS = "orths"

ATTACK_SUMMARY = "attacks_summary"
ATTACK_SUCCEEDED = "attacks_succeeded"
SIMILARITY = "similarity"
CHANGED = "changed"
CHANGED_WORDS = "changed_words"
SUCCEEDED = "succeeded"
ALL = "all"
DIFF = "diff"
EXPECTED = "expected"
ACTUAL = "actual"
COSINE_SCORE = "cosine_score"
CLASS = "class"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_RES = {
    "spoiled": {
        "attacks_summary": {"succeeded": 0, "all": 1},
        "attacks_succeeded": [],
    }
}


@click.command()
@click.option(
    "--dataset_name",
    help="Dataset name",
    type=str,
)
def main(dataset_name: str):
    """Downloads the dataset to the output directory."""
    lang = {
        "enron_spam": "en",
        "poleval": "pl",
        "20_news": "en",
        "wiki_pl": "pl",
    }[dataset_name]
    output_dir = f"data/results/{dataset_name}"
    input_file = f"data/preprocessed/{dataset_name}/test.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test.jsonl")
    classify = get_classify_function(dataset_name=dataset_name)
    dataset_df = pd.read_json(input_file, lines=True)

    spoiled, results = [], []
    similarity, max_sub = 0.95, 1
    classes = classify(dataset_df[TEXT].tolist())
    attack = TextFooler(lang)

    for i, cols in tqdm(
        dataset_df[[TEXT, LEMMAS, TAGS, ORTHS]].iterrows(), total=len(dataset_df)
    ):
        sentence, lemmas, tags, orths = cols[0], cols[1], cols[2], cols[3]
        changed_sent = attack.spoil(sentence, [], lemmas, tags, orths, similarity, max_sub)
        if changed_sent:
            spoiled.append(process(changed_sent, classes[i], classify))

    with open(output_path, mode="wt") as fd:
        fd.write(
            pd.DataFrame({"spoiled": spoiled}).to_json(
                orient="records", lines=True
            )
        )


if __name__ == "__main__":
    main()
