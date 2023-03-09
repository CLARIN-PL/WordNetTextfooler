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


def process_file(dataset_df, connection, lang, output_path):
    test_with_tags = pd.DataFrame(dataset_df)
    lemmas_col, tags_col = [], []
    cpus = cpu_count()
    with Pool(processes=cpus) as pool:
        results = []
        for idx in tqdm(range(0, len(dataset_df), cpus)):
            end = min(idx+cpus, len(dataset_df) + 1)
            for sentence in dataset_df[TEXT][idx:end]:
                results.append(pool.apply_async(tag_sentence, args=(connection,
                                                                    sentence,
                                                                    lang,)))
            for res in results:
                lemmas, tags = res.get()
                lemmas_col.append(lemmas)
                tags_col.append(tags)
            results = []
    test_with_tags[LEMMAS] = lemmas_col
    test_with_tags[TAGS] = tags_col

    with open(output_path, mode="wt") as fd:
        fd.write(test_with_tags.to_json(orient='records', lines=True))


@click.command()
@click.option(
    "--dataset_name",
    help="Dataset name",
    type=str,
)
def main(dataset_name: str):
    """Downloads the dataset to the output directory."""
    lang = 'en' if dataset_name == 'enron_spam' else 'pl'
    conn = Connection(config_file="experiments/configs/config.yml")
    output_dir = f"data/preprocessed/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    input_dir = f"data/datasets/{dataset_name}"
    for file in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, file)):
            process_file(pd.read_json(os.path.join(input_dir, file), lines=True),
                         conn, lang, os.path.join(output_dir, file))


if __name__ == "__main__":
    main()