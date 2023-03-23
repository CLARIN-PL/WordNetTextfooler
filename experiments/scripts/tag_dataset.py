"""Script for running tagger on datasets."""
import click
import pandas as pd
from lpmn_client_biz import Connection, IOType, Task, download
import json
import os
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
import spacy


TOKENS = "tokens"
ORTH = "orth"
LEXEMES = "lexemes"
LEMMA = "lemma"
MSTAG = "mstag"
TEXT = "text"
LEMMAS = "lemmas"
TAGS = "tags"
ORTHS = "orths"


def tag_sentence(sentence: str, lang: str):
    connection = Connection(config_file="experiments/configs/config.yml")
    lpmn = [{"spacy": {"lang": "en"}}]
    if lang == "pl":
        lpmn = [
            "morphodita",
            {"posconverter": {"input_format": "ccl", "output_format": "json"}},
        ]

    task = Task(lpmn, connection=connection)
    output_file_id = task.run(str(sentence), IOType.TEXT)
    tokens = []
    try:
        clarin_json = json.loads(
            download(connection, output_file_id, IOType.TEXT).decode("utf-8")
        )
        tokens = clarin_json[TOKENS]
    except json.decoder.JSONDecodeError:
        downloaded = download(connection, output_file_id, IOType.FILE)
        with open(downloaded, "r") as file:
            lines = [json.loads(line) for line in file.readlines()]
            for line in lines:
                tokens.extend(line[TOKENS])
        os.remove(downloaded)
    lemmas, tags, orths = [], [], []
    for token in tokens:
        lexeme = token[LEXEMES][0]
        lemmas.append(lexeme[LEMMA])
        tags.append(lexeme[MSTAG])
        orths.append(token[ORTH])
    return lemmas, tags, orths


def process_file(dataset_df, lang, output_path):
    test_with_tags = pd.DataFrame(dataset_df)
    lemmas_col, tags_col, orth_col = [], [], []
    cpus = 2
    with Pool(processes=cpus) as pool:
        results = []
        for idx in tqdm(range(0, len(dataset_df), cpus)):
            end = min(idx + cpus, len(dataset_df) + 1)
            for sentence in dataset_df[TEXT][idx:end]:
                results.append(
                    pool.apply_async(tag_sentence, args=[sentence, lang])
                )
            for res in results:
                lemmas, tags, orths = res.get()
                lemmas_col.append(lemmas)
                tags_col.append(tags)
                orth_col.append(orths)
            results = []
    test_with_tags[LEMMAS] = lemmas_col
    test_with_tags[TAGS] = tags_col
    test_with_tags[ORTHS] = orth_col

    return test_with_tags


def add_ner(dataset_df, language):
    model = "en_core_web_trf" if language == "en" else "pl_core_news_lg"
    nlp = spacy.load(model)
    ner_data = list()

    for text in tqdm(dataset_df["text"]):
        doc = nlp(text)
        doc_ner = list()
        for ent in doc.ents:
            doc_ner.append({
                "text": ent.text,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "label": ent.label_,
            })
        ner_data.append(doc_ner)

    dataset_df["ner"] = ner_data
    return dataset_df


@click.command()
@click.option(
    "--dataset_name",
    help="Dataset name",
    type=str,
)
@click.option(
    "--output",
    help="Output directory",
    type=str,

)
def main(dataset_name: str, output: str):
    """Downloads the dataset to the output directory."""
    lang = {
        "enron_spam": "en",
        "poleval": "pl",
        "20_news": "en",
        "wiki_pl": "pl",
    }[dataset_name]
    output_dir = f"{output}/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    input_dir = f"data/datasets/{dataset_name}"
    for file in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, file)):
            if file in ["test.jsonl", "adversarial.jsonl"]:
                test_with_tags = process_file(
                    pd.read_json(os.path.join(input_dir, file), lines=True),
                    lang,
                    os.path.join(output_dir, file),
                )
                test_with_tags = add_ner(test_with_tags, lang)
            else:
                test_with_tags = pd.DataFrame(
                    pd.read_json(os.path.join(input_dir, file), lines=True)
                )
                empty_list = [[] for _ in range(len(test_with_tags))]
                test_with_tags[LEMMAS] = empty_list
                test_with_tags[TAGS] = empty_list
                test_with_tags[ORTHS] = empty_list
                test_with_tags["ner"] = empty_list
        with open(os.path.join(output_dir, file), mode="wt") as fd:
            fd.write(
                test_with_tags.to_json(orient="records", lines=True)
            )


if __name__ == "__main__":
    main()
