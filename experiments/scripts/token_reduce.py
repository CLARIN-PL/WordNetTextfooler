"""Reduce sample size to 512 tokens"""

from pathlib import Path
import click
import pandas as pd
import spacy
import uuid
import shutil
from tqdm import tqdm
import os
import json
from text_attacks.utils import get_model_and_tokenizer
from lpmn_client_biz import Connection, IOType, Task, download, upload

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
    for idx, sentence in sentences.items():
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


def add_ner(sentences, language):
    model = "en_core_web_trf" if language == "en" else "pl_core_news_lg"
    nlp = spacy.load(model)
    ner_data = {}

    for idx, text in tqdm(sentences.items()):
        doc = nlp(text)
        doc_ner = list()
        for ent in doc.ents:
            doc_ner.append({
                "text": ent.text,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "label": ent.label_,
            })
        ner_data[idx] = doc_ner
    return ner_data


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
    lang = {
        "enron_spam": "en",
        "poleval": "pl",
        "20_news": "en",
        "wiki_pl": "pl",
        "ag_news": "en",
    }[dataset_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = get_model_and_tokenizer(
        dataset_name=dataset_name
    )
    model.to("cpu")
    model.eval()
    test = pd.read_json(f"data/preprocessed/{dataset_name}/test.jsonl", lines=True)

    texts = test["text"].tolist()
    texts_reduced = {}
    for i, sentence in test["text"].items():
        encoded = tokenizer.encode(sentence, add_special_tokens=True, max_length=512, truncation=True)
        decod_res = tokenizer.decode(encoded, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        last_word = decod_res.split(" ")[-1]
        max_len = len(" ".join(sentence.split(" ")[:512]))
        idx = sentence.rfind(last_word, 0, max_len)
        if idx + len(last_word) < len(sentence) and idx > 0:
            texts_reduced[i] = sentence[:idx + len(last_word)]
    print("To reduce ", len(texts_reduced), " of ", len(texts))

    if len(texts_reduced) > 0:
        tagged_reduced = tag_sentences(texts_reduced, lang)
        ner_reduced = add_ner(texts_reduced, lang)
        for idx, sentence in texts_reduced.items():
            test.loc[idx, TEXT] = sentence
            test.at[idx, LEMMAS] = tagged_reduced[idx][LEMMAS]
            test.at[idx, TAGS] = tagged_reduced[idx][TAGS]
            test.at[idx, ORTHS] = tagged_reduced[idx][ORTHS]
            test.at[idx, NER] = ner_reduced[idx]
    test.to_json(output_dir / "test.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()