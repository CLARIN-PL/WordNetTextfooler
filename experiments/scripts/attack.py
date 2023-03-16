"""Script for running attacks on datasets."""
import json

import click
import pandas as pd
import os
from tqdm import tqdm
from text_attacks.utils import get_classify_function
from textfooler import Attack, TextFooler, BaseLine, process, run_queue, filter_similarity_queue
from queue import Full, Empty
from time import sleep
from multiprocessing import Process
from multiprocessing import Queue, Manager
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
SLEEP_TIME = 0.01
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
        try:
            sentence, sent_id, lemmas, tags, orths = cols[0], cols[1], \
                                                     cols[2], cols[3], cols[4]
            queue_out.put([sentence, orths, [], lemmas, tags, sent_id])
        except Full:
            sleep(SLEEP_TIME)
    try:
        queue_out.put(None)
    except Full:
        sleep(SLEEP_TIME)


def data_saver(queue_in, output_file):
    item = 1
    while item is not None:
        try:
            item = queue_in.get(block=False)
        except Empty:
            sleep(SLEEP_TIME)
            continue
        if item is not None:
            with open(output_file, 'a') as file_out:
                json.dump(item, file_out, indent=2)


def classify_queue(queue_in, queue_out, classify_fun):
    item = True
    while item is not None:
        try:
            item = queue_in.get(block=False)
        except Empty:
            sleep(SLEEP_TIME)
            continue
        if item is not None:
            try:
                sent_id, org_sentence, changed_sents = item
                sentences = [org_sentence].extend([sent[TEXT] for sent in changed_sents])
                classified = classify_fun(sentences)
                queue_out.put((sent_id, org_sentence, changed_sents, classified))
            except Full:
                sleep(SLEEP_TIME)
                continue
    queue_out.put(None)


def log_queues(queues):
    while True:
        sizes = [q.qsize() for q in queues]
        print(sizes, flush=True)
        sleep(2)
        
    
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

    attack = {
        "attack_textfooler": TextFooler(lang),
        "attack_basic": BaseLine(lang, 0.5, 0.4, 0.3)
    }[attack_type]

    # sim = Similarity(0.95, "distiluse-base-multilingual-cased-v1")
    output_dir = f"data/results/{attack_type}/{dataset_name}/"
    input_file = f"data/preprocessed/{dataset_name}/test.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test.jsonl")
    classify = get_classify_function(dataset_name=dataset_name)
    max_sub = 1

    m = Manager()
    queues = [m.Queue(maxsize=QUEUE_SIZE) for _ in range(5)]
    processes = [Process(target=data_producer, args=(queues[0], input_file,)),  # loading data file_in -> 0
                 Process(target=attack.spoil_queue, args=(queues[0], queues[1], max_sub,)),  # spoiling 0 -> 1
                 Process(target=filter_similarity_queue, args=(queues[1], queues[2], 0.95,
                                                               "distiluse-base-multilingual-cased-v1",)),  # cosim 1 -> 2
                 Process(target=classify_queue, args=(queues[2], queues[3], classify, )),  # classify changed 2 -> 3
                 Process(target=run_queue, args=(queues[3], queues[4], process,)),  # process 3 -> 4
                 Process(target=data_saver, args=(queues[4], output_path,))]  # saving 4 -> file_out
    [p.start() for p in processes]

    log_que = Thread(target=log_queues, args=(queues, ))
    log_que.daemon = True
    log_que.start()
    # wait for all processes to finish
    [p.join() for p in processes]
    log_que.join(timeout=0.5)


if __name__ == "__main__":
    main()
