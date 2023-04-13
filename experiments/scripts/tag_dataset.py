"""Script for running tagger on datasets."""
import click
import pandas as pd
from lpmn_client_biz import Connection, IOType, Task, download, upload
import json
import os
from tqdm import tqdm
import spacy
import shutil
import uuid

TOKENS = "tokens"
ORTH = "orth"
LEXEMES = "lexemes"
LEMMA = "lemma"
MSTAG = "mstag"
TEXT = "text"
LEMMAS = "lemmas"
TAGS = "tags"
ORTHS = "orths"
NER = "ner"


def tag_sentences(sentences, lang: str):
    results = {}
    connection = Connection(config_file="experiments/configs/config.yml")
    lpmn = [[{"postagger": {"lang": lang}}], 'makezip']
    input_dir = str(uuid.uuid4())
    os.makedirs(input_dir)
    for idx, sentence in enumerate(sentences):
        with open(f'{input_dir}/file_{idx}',
                  'w', encoding='utf8') as fout:
            fout.write(sentence)

    uploaded = upload(connection, input_dir)
    task = Task(lpmn, connection)
    result = task.run(uploaded, IOType.FILE, verbose=True)
    archive_path = download(
        connection,
        result,
        IOType.FILE,
        filename=f'{uuid.uuid4()}.zip'
    )
    output_path = archive_path.replace('.zip', '')
    shutil.unpack_archive(archive_path, output_path)
    files = sorted(os.listdir(output_path), key=lambda x: int(x.split('_')[1]))
    for j, filename in enumerate(files):
        with open(f'{output_path}/{filename}', 'r') as file:
            lines = [json.loads(line) for line in file.readlines()]
            lemmas, tags, orths = [], [], []
            if len(lines) > 0:
                for idx, line in enumerate(lines):
                    tokens = line[TOKENS]
                    for token in tokens:
                        lexeme = token[LEXEMES][0]
                        lemmas.append(lexeme[LEMMA])
                        tags.append(lexeme[MSTAG])
                        orths.append(token[ORTH])
            else:
                tokens = lines[0][TOKENS]
                for token in tokens:
                    lexeme = token[LEXEMES][0]
                    lemmas.append(lexeme[LEMMA])
                    tags.append(lexeme[MSTAG])
                    orths.append(token[ORTH])
            results[int(filename.split('_')[1])] = {
                LEMMAS: lemmas,
                TAGS: tags,
                ORTHS: orths
            }
    shutil.rmtree(input_dir)
    os.remove(archive_path)
    shutil.rmtree(output_path)
    return results


def process_file(dataset_df, lang):
    test_with_tags = pd.DataFrame(dataset_df)
    lemmas_col, tags_col, orth_col = [], [], []

    tagged_sentences = tag_sentences(dataset_df[TEXT].tolist(), lang)
    for idx, tokens in tagged_sentences.items():
        lemmas_col.append(tokens[LEMMAS])
        tags_col.append(tokens[TAGS])
        orth_col.append(tokens[ORTHS])
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
def main(dataset_name: str):
    """Downloads the dataset to the output directory."""
    lang = {
        "enron_spam": "en",
        "poleval": "pl",
        "20_news": "en",
        "wiki_pl": "pl",
        "ag_news": "en",
    }[dataset_name]
    output_dir = f"data/preprocessed/{dataset_name}/"
    os.makedirs(output_dir, exist_ok=True)

    input_dir = f"data/datasets/{dataset_name}"
    for file in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, file)):
            if file in ["test.jsonl", "adversarial.jsonl"]:
                test_with_tags = process_file(
                    pd.read_json(os.path.join(input_dir, file), lines=True),
                    lang
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
