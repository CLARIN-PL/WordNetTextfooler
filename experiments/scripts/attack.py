"""Script for running attacks on datasets."""
import importlib
import json
from collections import defaultdict
import click
import pandas as pd
import os

import torch
from tqdm import tqdm
from textfooler import Attack, TextFooler, Similarity, BaseLine, \
    process, run_queue, filter_similarity_queue, spoil_queue
from time import sleep, time
from multiprocessing import Process
from multiprocessing import Queue, Manager
from threading import Thread
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

TEXT = "text"
LEMMAS = "lemmas"
TAGS = "tags"
ORTHS = "orths"
ID = "id"
PRED = "pred_label"

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
FEATURES = "features"
IMPORTANCE = "importance"
SYNONYM = "synonym"
DISCARD = "discard"
GLOBAL = "global"
LOCAL = "local"


os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_RES = {
    "spoiled": {
        "attacks_summary": {"succeeded": 0, "all": 1},
        "attacks_succeeded": [],
    }
}


def data_producer(queue_out, dataset_df):
    for i, cols in tqdm(
            dataset_df[[TEXT, ID, LEMMAS, TAGS, ORTHS, PRED]].iterrows(), total=len(dataset_df)
    ):
        sentence, sent_id, lemmas, tags, orths, y_pred = cols[0], cols[1], \
                                                         cols[2], cols[3], cols[4], cols[5]
        queue_out.put([sentence, orths, [], lemmas, tags, sent_id, y_pred])


def data_saver(queue_in, queue_log, output_file, output_dir, cases_nbr, queues_kill, to_kill_nbr):
    processed_nbr, start = 0, time()
    item = 1
    test_y, pred_y = [], []
    spoiled_sents = []
    ch_suc, ch_all = 0, 0
    end_time = time()
    while item is not None:
        item = queue_in.get()
        if item is not None:
            processed_nbr += 1
            spoiled, class_test, class_pred = item
            test_y.append(class_test)
            pred_y.append(class_pred)
            queue_log.put(f"Processed and saved {processed_nbr} in {time() - start} s")
            ch_suc += spoiled[ATTACK_SUMMARY][SUCCEEDED]
            ch_all += spoiled[ATTACK_SUMMARY][ALL]
            spoiled_sents.append(spoiled)
        if processed_nbr == cases_nbr:
            for que_kill in queues_kill:
                [que_kill.put(None) for _ in range(to_kill_nbr)]
        if processed_nbr == cases_nbr - 10:
            end_time = time()
        if processed_nbr >= cases_nbr - 10:
            if sum([q.qsize() for q in queues_kill]) == 0 and (time() - end_time) > 3600:
                for que_kill in queues_kill:
                    [que_kill.put(None) for _ in range(to_kill_nbr)]
    with open(output_file, 'wt') as fd:
        fd.write(pd.DataFrame(spoiled_sents).to_json(
            orient="records", lines=True))
    np.savetxt(f"{output_dir}/metrics.txt", confusion_matrix(test_y, pred_y))
    with open(f"{output_dir}/metrics.txt", mode="at") as fd:
        fd.write('\n')
        fd.write(classification_report(test_y, pred_y))
        fd.write('\n')
        fd.write(f"succeeded {ch_suc} all {ch_all}")


def classify_queue(queue_in, queue_out, queue_log, dataset_name, cuda_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    fun = getattr(
        importlib.import_module(f"text_attacks.models.{dataset_name}"),
        "get_classify_function",
    )
    classify_fun = fun(device="cuda" if torch.cuda.is_available() else "cpu")
    queue_log.put(f"Classify device {'cuda' if torch.cuda.is_available() else 'cpu'}")
    item = True
    while item is not None:
        item = queue_in.get()
        queue_log.put("Classify got from queue")
        if item is not None:
            sent_id, org_sentence, y_pred, changed_sents = item
            sentences = [sent[TEXT] for sent in changed_sents]
            queue_log.put(f"Classifying sentences {len(sentences)}, id {sent_id}")
            classified = classify_fun(sentences) if sentences else []
            queue_out.put((sent_id, org_sentence, changed_sents, y_pred, classified))
            queue_log.put(f"Classified sentences {sent_id}")


def log_queues(queues):
    while True:
        sizes = [q.qsize() for q in queues]
        print(sizes, flush=True)
        sleep(10)


def log_info_queue(queue):
    print("Logging queue")
    while True:
        item = queue.get()
        print(item)


def load_dir_files(dir_path):
    result = {}
    for filename in os.listdir(dir_path):
        with open(os.path.join(dir_path, filename), 'r') as fin:
            importance = json.load(fin)
            result[filename.split("__")[0]] = {
                word: importance[IMPORTANCE][idx]
                for idx, word in importance[FEATURES].items()
                if word
            }
    return result


def load_xai_importance(input_dir):
    global_xai_dir = os.path.join(input_dir, "global")
    local_xai_dir = os.path.join(input_dir, "local", "test")
    local_dirs = os.listdir(local_xai_dir)
    local_class_to_file = {dir_name: load_dir_files(os.path.join(local_xai_dir, dir_name))
                           for dir_name in local_dirs}
    local_file_to_class = defaultdict(dict)
    for c_name in local_dirs:
        for f_name, value_df in local_class_to_file[c_name].items():
            local_file_to_class[f_name][c_name] = value_df
    return load_dir_files(global_xai_dir), local_file_to_class


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
    xai_global, xai_local = {}, {}
    if "attack_xai" in attack_type:
        importance = load_xai_importance(f"data/explanations/{dataset_name}")
        xai_global, xai_local = importance[0], importance[1]
    xai_sub = 5
    params = {
        "attack_textfooler": [lang, SYNONYM],
        "attack_textfooler_discard": [lang, DISCARD],
        "attack_basic": [lang, 0.5, 0.4, 0.3],  # prawopodobieństwa spacji  > usunięcia znaku > usunięcia słowa
        "attack_xai": [lang, xai_global, xai_local, GLOBAL, SYNONYM, xai_sub],
        "attack_xai_discard": [lang, xai_global, xai_local, GLOBAL, DISCARD, xai_sub],
        "attack_xai_local": [lang, xai_global, xai_local, LOCAL, SYNONYM, xai_sub],
        "attack_xai_discard_local": [lang, xai_global, xai_local, LOCAL, DISCARD, xai_sub]
    }[attack_type]

    output_dir = f"data/results/{attack_type}/{dataset_name}/"
    input_file = f"data/classification/{dataset_name}/test.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test.jsonl")
    dataset_df = pd.read_json(input_file, lines=True)

    max_sub = 1

    m = Manager()
    queues = [m.Queue(maxsize=QUEUE_SIZE) for _ in range(6)]
    sim = Similarity(queues[5], 0.95, "distiluse-base-multilingual-cased-v1")
    processes = [
        Process(target=data_producer, args=(queues[0], dataset_df,)),  # loading data file_in -> 0
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        # Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        # Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        # Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        # Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        # Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        Process(target=spoil_queue, args=(queues[0], queues[1], queues[5], max_sub, attack_type, params)),
        # spoiling 0 -> 1
        Process(target=filter_similarity_queue, args=(queues[1], queues[2], queues[5], sim)),
        Process(target=filter_similarity_queue, args=(queues[1], queues[2], queues[5], sim)),  # cosim 1 -> 2
        Process(target=classify_queue, args=(queues[2], queues[3], queues[5], dataset_name, "6")),
        Process(target=classify_queue, args=(queues[2], queues[3], queues[5], dataset_name, "4")),
        # classify changed 2 -> 3
        Process(target=run_queue, args=(queues[3], queues[4], queues[5], process,)),  # process 3 -> 4
        Process(target=data_saver, args=(queues[4], queues[5], output_path, output_dir, len(dataset_df), queues, 30))
        # saving 4 -> file_out
    ]
    [p.start() for p in processes]

    log_que = Thread(target=log_queues, args=(queues[:5],))
    log_que.daemon = True
    log_que.start()
    info_que = Thread(target=log_info_queue, args=(queues[5],))
    info_que.daemon = True
    info_que.start()
    # wait for all processes to finish
    [p.join() for p in processes]
    log_que.join(timeout=0.5)
    info_que.join(timeout=0.5)


if __name__ == "__main__":
    main()
