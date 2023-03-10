"""Script for running attacks on datasets."""
import click
import pandas as pd
import json
import os
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
from text_attacks.utils import get_classify_function
from textfooler import Attack, TextFooler


TEXT = 'text'
LEMMAS = 'lemmas'
TAGS = 'tags'


def spoil_sentence(sentence, lemmas, tags, lang, classify_fun, similarity):
    attack = TextFooler(lang)
    return attack.process(sentence, lemmas, tags, classify_fun, similarity)


@click.command()
@click.option(
    "--dataset_name",
    help="Dataset name",
    type=str,
)
def main(dataset_name: str):
    """Downloads the dataset to the output directory."""
    lang = 'en' if dataset_name == 'enron_spam' else 'pl'
    output_dir = f"data/results/{dataset_name}"
    input_file = f"data/preprocessed/{dataset_name}/test.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'test.jsonl')
    classify = get_classify_function(
        dataset_name=dataset_name,
    )
    dataset_df = pd.read_json(input_file, lines=True)
    spoiled = []
    similarity = 0.95
    cpus = cpu_count()
    with Pool(processes=cpus) as pool:
        results = []
        for idx in tqdm(range(0, len(dataset_df), cpus)):
            end = min(idx+cpus, len(dataset_df) + 1)
            for sentence, lemmas, tags in dataset_df[[TEXT, LEMMAS, TAGS], idx:end]:
                results.append(pool.apply_async(spoil_sentence, args=[sentence, lemmas,
                                                                      tags, lang, classify, similarity]))
            for res in results:
                spoiled_sent = res.get()
                spoiled.append(spoiled_sent)
            results = []

    with open(output_path, mode="wt") as fd:
        fd.write(pd.DataFrame(
            {"spoiled": spoiled}).to_json(
            orient='records', lines=True))


if __name__ == "__main__":
    main()
