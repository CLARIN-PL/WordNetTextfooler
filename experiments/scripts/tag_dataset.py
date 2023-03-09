"""Script for running tagger on datasets."""
import click
import pandas as pd
from lpmn_client_biz import Connection, IOType, Task, download
import json
import os
from tqdm import tqdm
from multiprocessing import cpu_count, Pool

TOKENS = 'tokens'
ORTH = 'orth'
LEXEMES = 'lexemes'
LEMMA = 'lemma'
MSTAG = 'mstag'
TEXT = 'text'
LEMMAS = 'lemmas'
TAGS = 'tags'


def tag_sentence(connection: Connection, sentence: str, lang: str):
    task = Task([{'postagger': {'output_type': 'json', 'lang': lang}}],
                connection=connection)
    output_file_id = task.run(sentence, IOType.TEXT)
    tokens = []
    try:
        clarin_json = json.loads(download(connection, output_file_id, IOType.TEXT).decode("utf-8"))
        tokens = clarin_json[TOKENS]
    except json.decoder.JSONDecodeError:
        downloaded = download(connection, output_file_id, IOType.FILE)
        with open(downloaded, 'r') as file:
            lines = [json.loads(line) for line in file.readlines()]
            for line in lines:
                tokens.extend(line[TOKENS])
        os.remove(downloaded)
    lemmas, tags = [], []
    for token in tokens:
        lexeme = token['lexemes'][0]
        lemmas.append(lexeme['lemma'])
        tags.append(lexeme['mstag'])
    return lemmas, tags


@click.command()
@click.option(
    "--dataset_name",
    help="Dataset name",
    type=str,
)
def main(dataset_name: str):
    """Downloads the dataset to the output directory."""
    lang = 'en' if dataset_name == 'enron_spam' else 'pl'
    test = pd.read_json(f"data/datasets/{dataset_name}/test.jsonl", lines=True)
    test_with_tags = pd.DataFrame(test)
    conn = Connection(config_file="experiments/configs/config.yml")
    lemmas_col, tags_col = [], []
    cpus = cpu_count()
    with Pool(processes=cpus) as pool:
        results = []
        for idx in tqdm(range(0, len(test), cpus)):
            end = min(idx+cpus, len(test) + 1)
            for sentence in test[TEXT][idx:end]:
                results.append(pool.apply_async(tag_sentence, args=[conn,
                                                                    sentence,
                                                                    lang]))
            for res in results:
                lemmas, tags = res.get()
                lemmas_col.append(lemmas)
                tags_col.append(tags)
            results = []
    test_with_tags[LEMMAS] = lemmas_col
    test_with_tags[TAGS] = tags_col

    output_dir = f"data/preprocessed/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/test.jsonl", mode="wt") as fd:
        fd.write(test_with_tags.to_json(orient='records', lines=True))


if __name__ == "__main__":
    main()