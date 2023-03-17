"""Script for running attacks on datasets."""
import json

import click
import pandas as pd
import os

import torch
from tqdm import tqdm
from text_attacks.utils import get_classify_function
from textfooler import Attack, TextFooler, Similarity, BaseLine, \
    process, run_queue, filter_similarity_queue, spoil_queue
from time import sleep
from multiprocessing import Process
from multiprocessing import Queue, Manager
import multiprocess
from threading import Thread

TEXT = "text"
LEMMAS = "lemmas"
TAGS = "tags"
ORTHS = "orths"
ID = "id"

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
QUEUE_SIZE = 1000

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_RES = {
    "spoiled": {
        "attacks_summary": {"succeeded": 0, "all": 1},
        "attacks_succeeded": [],
    }
}


def data_producer(queue_out, input_file):
    dataset_df = pd.read_json(input_file, lines=True)
    for i, cols in tqdm(
            dataset_df[[TEXT, ID, LEMMAS, TAGS, ORTHS]].iterrows(), total=len(dataset_df)
    ):
        sentence, sent_id, lemmas, tags, orths = cols[0], cols[1], \
                                                 cols[2], cols[3], cols[4]
        queue_out.put([sentence, orths, [], lemmas, tags, sent_id])
    queue_out.put(None)


def data_saver(queue_in, output_file):
    item = 1
    while item is not None:
        item = queue_in.get()
        if item is not None:
            with open(output_file, 'a') as file_out:
                json.dump(item, file_out, indent=2)


def classify_queue(queue_in, queue_out, queue_log, classify_fun):
    item = True
    while item is not None:
        item = queue_in.get()
        queue_log.put("Classify got from queue")
        if item is not None:
            sent_id, org_sentence, changed_sents = item
            sentences = [org_sentence]
            sentences.extend([sent[TEXT] for sent in changed_sents])
            queue_log.put(f"Classifying sentences {sentences[:100]}")
            classified = classify_fun(sentences)
            queue_out.put((sent_id, org_sentence, changed_sents, classified))
    queue_out.put(None)


def log_queues(queues):
    while True:
        sizes = [q.qsize() for q in queues]
        print(sizes, flush=True)
        sleep(2)


def log_info_queue(queue):
    print("Logging queue")
    while True:
        item = queue.get()
        print(item)


@click.command()
@click.option(
    "--dataset_name",
    help="Dataset name",
    type=str,
)
@click.option(
    "--attack_type",
    help="Attack type",
    type=str,
)
def main(dataset_name: str, attack_type: str):
    """Downloads the dataset to the output directory."""
    lang = {
        "enron_spam": "en",
        "poleval": "pl",
        "20_news": "en",
        "wiki_pl": "pl",
    }[dataset_name]
    params = {
        "attack_textfooler": [lang],
        "attack_basic": [lang, 0.5, 0.4, 0.3]
    }[attack_type]

    output_dir = f"data/results/{attack_type}/{dataset_name}/"
    input_file = f"data/preprocessed/{dataset_name}/test.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test.jsonl")
    classify = get_classify_function(dataset_name=dataset_name, device="cpu")
    max_sub = 1

    m = Manager()
    queues = [m.Queue(maxsize=QUEUE_SIZE) for _ in range(6)]
    sim = Similarity(queues[5], 0.95, "distiluse-base-multilingual-cased-v1")
    processes = [Process(target=data_producer, args=(queues[0], input_file,)),  # loading data file_in -> 0
                 Process(target=spoil_queue, args=(queues[0], queues[1], max_sub, attack_type, params)),  # spoiling 0 -> 1
                 Process(target=filter_similarity_queue, args=(queues[1], queues[2], queues[5], sim)),  # cosim 1 -> 2
                 multiprocess.Process(target=classify_queue, args=(queues[2], queues[3], queues[5], classify, )),  # classify changed 2 -> 3
                 Process(target=run_queue, args=(queues[3], queues[4], process,)),  # process 3 -> 4
                 Process(target=data_saver, args=(queues[4], output_path,))]  # saving 4 -> file_out
    [p.start() for p in processes]

    log_que = Thread(target=log_queues, args=(queues, ))
    log_que.daemon = True
    log_que.start()
    info_que = Thread(target=log_info_queue, args=(queues[5], ))
    info_que.daemon = True
    info_que.start()
    # wait for all processes to finish
    [p.join() for p in processes]
    log_que.join(timeout=0.5)
    info_que.join(timeout=0.5)


if __name__ == "__main__":
    main()
