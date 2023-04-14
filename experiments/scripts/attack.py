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
from string import punctuation


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
QUEUE_SIZE = 60
FEATURES = "features"
IMPORTANCE = "importance"
SYNONYM = "synonym"
DISCARD = "discard"
GLOBAL = "global"
LOCAL = "local"
CHAR_DISCARD = "char_discard"


os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_RES = {
    "spoiled": {
        "attacks_summary": {"succeeded": 0, "all": 1},
        "attacks_succeeded": [],
    }
}


def join_punct(words):
    punc = set(punctuation)
    return "".join(w if set(w) <= punc else " " + w for w in words).lstrip()


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
    ch_suc, ch_all, synonyms_nbr = 0, 0, 0
    samples, samples_succ = 0, 0
    count_tokens, sum_tokens = 0, 0

    end_time = time()
    while item is not None:
        item = queue_in.get()
        if item is not None:
            processed_nbr += 1
            spoiled, class_test, class_pred, synonym_nbr = process(*item)
            test_y.append(class_test)
            pred_y.append(class_pred)
            queue_log.put(f"Processed and saved {processed_nbr} in {time() - start} s")
            samples_succ = samples_succ + 1 if spoiled[ATTACK_SUMMARY][SUCCEEDED] > 0 else samples_succ
            samples += 1
            for success in spoiled[ATTACK_SUCCEEDED]:
                if CHANGED_WORDS in success:
                    count_tokens += len(success[CHANGED_WORDS])
                    sum_tokens += 1
            ch_suc += spoiled[ATTACK_SUMMARY][SUCCEEDED]
            ch_all += spoiled[ATTACK_SUMMARY][ALL]
            synonyms_nbr += synonym_nbr
            with open(output_file, 'at') as fd:
                fd.write(pd.DataFrame([spoiled]).to_json(orient="records", lines=True))
            spoiled  = None
        if processed_nbr == cases_nbr:
            for que_kill in queues_kill:
                [que_kill.put(None) for _ in range(to_kill_nbr)]
        if processed_nbr == cases_nbr - 10:
            end_time = time()
        if processed_nbr >= cases_nbr - 10:
            if sum([q.qsize() for q in queues_kill]) == 0 and (time() - end_time) > 3600:
                for que_kill in queues_kill:
                    [que_kill.put(None) for _ in range(to_kill_nbr)]

    metrics = {
        "confusion_matrix": confusion_matrix(test_y, pred_y).tolist(),
        "classification_report": classification_report(test_y, pred_y, output_dict=True),
        "attacks_succeeded": ch_suc,
        "attacks_all": ch_all,
        "synonyms_nbr": synonyms_nbr,
        "success_rate": ch_suc / ch_all,
        "success_rate_per_synonym": ch_suc / synonyms_nbr,
        "time": time() - start,
        "samples": samples,
        "samples_succ": samples_succ,
        "count_tokens": count_tokens,
        "sum_tokens": sum_tokens,
        "%F": (samples - samples_succ) / samples if samples > 0 else 0,
        "%C": count_tokens / sum_tokens if sum_tokens > 0 else 0,
        "BLEU": 0,
        "P": 0
    }
    with open(f"{output_dir}/metrics.json", mode="w") as fd:
        json.dump(metrics, fd)


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
            sent_id, org_sentence, y_pred, changed, synonyms_nbr, sent_words = item
            sentences = []
            for subst, _ in changed:
                sent_words_copy = [*sent_words]
                for idx, word_change in subst.items():
                    sent_words_copy[idx] = word_change['word']
                sentences.append(join_punct(sent_words_copy))

            queue_log.put(f"Classifying sentences {synonyms_nbr}, id {sent_id}")
            classified = classify_fun(sentences) if sentences else []
            queue_out.put((sent_id, org_sentence, changed, y_pred, classified, synonyms_nbr, sent_words))
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
        if item is not None:
            print(item)
    print("Logging queue finished")


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
    return load_dir_files(global_xai_dir), dict(local_file_to_class)


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
        "ag_news": "en",
    }[dataset_name]
    xai_global, xai_local = {}, {}
    if "attack_xai" in attack_type:
        importance = load_xai_importance(f"data/explanations/{dataset_name}")
        xai_global, xai_local = importance[0], importance[1]
    xai_sub = 0.15
    max_sub = 3
    char_delete_size = 0.4
    similarity_bound = 0.3

    params = {
        "attack_textfooler": [lang, SYNONYM],
        "attack_textfooler_discard": [lang, DISCARD],
        "attack_basic": [lang, 0.5, 0.4, 0.3],  # prawopodobieństwa spacji  > usunięcia znaku > usunięcia słowa
        "attack_xai": [lang, xai_global, xai_local, GLOBAL, SYNONYM, xai_sub],
        "attack_xai_discard": [lang, xai_global, xai_local, GLOBAL, DISCARD, xai_sub],
        "attack_xai_local": [lang, xai_global, xai_local, LOCAL, SYNONYM, xai_sub],
        "attack_xai_discard_local": [lang, xai_global, xai_local, LOCAL, DISCARD, xai_sub],
        "attack_xai_char_discard": [lang, xai_global, xai_local, GLOBAL, CHAR_DISCARD, xai_sub, char_delete_size],
        "attack_xai_char_discard_local": [lang, xai_global, xai_local, LOCAL, CHAR_DISCARD, xai_sub, char_delete_size]
    }[attack_type]
    output_dir = f"data/results/{attack_type}/{dataset_name}/"
    input_file = f"data/classification/{dataset_name}/test.jsonl"

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test.jsonl")
    dataset_df = pd.read_json(input_file, lines=True)


    test_sent_ids = ["Komputery_199721.txt", "Zydzi_976178.txt", "Kotowate_2015873.txt", "Zydzi_1602490.txt",
                     "Pilka-nozna_2899267.txt", "Optyka_1926807.txt", "Zydzi_929483.txt",
                     "Niemieccy-wojskowi_2410107.txt"]

    # dataset_df = dataset_df[dataset_df['id'].isin(test_sent_ids)]
    # dataset_df = dataset_df.reset_index(drop=True)

    dataset_df = dataset_df[:20]

    m = Manager()
    queues = [m.Queue(maxsize=QUEUE_SIZE) for _ in range(5)]

    log_que = Thread(target=log_queues, args=(queues[:4],))
    log_que.daemon = True
    log_que.start()
    info_que = Thread(target=log_info_queue, args=(queues[4],))
    info_que.daemon = True
    info_que.start()

    processes_nbr = 15
    sim = Similarity(queues[4], similarity_bound, "distiluse-base-multilingual-cased-v1")
    processes = [Process(target=data_producer, args=(queues[0], dataset_df,))]  # loading data file_in -> 0

    processes.extend([Process(target=spoil_queue, args=(queues[0], queues[1], queues[4], max_sub, attack_type, params))
                     for _ in range(processes_nbr)])  # spoiling 0 -> 1

    processes.extend([Process(target=filter_similarity_queue, args=(queues[1], queues[2], queues[4], sim)),
                      Process(target=filter_similarity_queue, args=(queues[1], queues[2], queues[4], sim)),  # cosim 1 -> 2
                      Process(target=classify_queue, args=(queues[2], queues[3], queues[4], dataset_name, "3")),
                      Process(target=classify_queue, args=(queues[2], queues[3], queues[4], dataset_name, "3")),
                      # classify changed 2 -> 3
                      # Process(target=run_queue, args=(queues[3], queues[4], queues[5], process,)),  # process 3 -> 4
                      Process(target=data_saver, args=(queues[3], queues[4], output_path, output_dir, len(dataset_df), queues, processes_nbr+6)) # saving 3 -> file_out
                     ])

    [p.start() for p in processes]

    # wait for all processes to finish
    [p.join() for p in processes]
    log_que.join(timeout=0.5)
    info_que.join(timeout=0.5)


if __name__ == "__main__":
    main()
