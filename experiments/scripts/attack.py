"""Script for running attacks on datasets."""
import click
import pandas as pd
import os
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
from text_attacks.utils import get_classify_function
from textfooler import Attack, TextFooler, BaseLine, process


TEXT = "text"
LEMMAS = "lemmas"
TAGS = "tags"

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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_RES = {
    "spoiled": {
        "attacks_summary": {"succeeded": 0, "all": 1},
        "attacks_succeeded": [],
    }
}


def spoil_sentence(sentence, lemmas, tags, lang, similarity, max_sub):
    attack = TextFooler(lang)
    # attack = BaseLine(lang, 0.5, 0.4, 0.3)
    return attack.spoil(sentence, [], lemmas, tags, similarity, max_sub)


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
    }[dataset_name]
    output_dir = f"data/results/{dataset_name}"
    input_file = f"data/preprocessed/{dataset_name}/test.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test.jsonl")
    classify = get_classify_function(dataset_name=dataset_name)
    dataset_df = pd.read_json(input_file, lines=True)
    # dataset_df = dataset_df[:10]

    spoiled, results = [], []
    similarity, max_sub = 0.95, 1
    cpus = cpu_count()
    classes = classify(dataset_df[TEXT].tolist())
    # used_id = 0
    # sent_nbr = len(dataset_df[TEXT])
    # with Pool(processes=cpus) as pool:
    #     for idx in range(0, min(cpus, sent_nbr)):
    #         sentence, lemmas, tags = dataset_df[TEXT][idx], \
    #                                  dataset_df[LEMMAS][idx], \
    #                                  dataset_df[TAGS][idx]
    #
    lang = "en" if dataset_name == "enron_spam" else "pl"
    #         results.append(pool.apply_async(spoil_sentence, args=[sentence,
    #                                                               lemmas,
    #                                                               tags,
    #                                                               lang,
    #                                                               similarity,
    #                                                               max_sub]))
    #         used_id = idx
    #     count = len(results)
    #     while count and used_id < sent_nbr:
    #         ready = 0
    #         to_rm = []
    #         for r in results:
    #             if r.ready():
    #                 ready += 1
    #                 changed_sent = r.get()
    #                 if changed_sent:
    #                     spoiled.append(process(changed_sent, classes[i], classify))
    #                 to_rm.append(r)
    #         count = len(results) - ready
    #         results = [res for res in results if res not in to_rm]
    #         h_bound = min(used_id + cpus - len(results), sent_nbr)
    #         for i in range(used_id + 1, h_bound):
    #             used_id += 1
    #             sentence, lemmas, tags = dataset_df[TEXT][idx], \
    #                                      dataset_df[LEMMAS][idx], \
    #                                      dataset_df[TAGS][idx]
    #
    #             results.append(pool.apply_async(spoil_sentence, args=[sentence,
    #                                                                   lemmas,
    #                                                                   tags,
    #                                                                   lang,
    #                                                                   similarity,
    #                                                                   max_sub]))

    for i, cols in tqdm(
        dataset_df[[TEXT, LEMMAS, TAGS]].iterrows(), total=len(dataset_df)
    ):
        sentence, lemmas, tags = cols[0], cols[1], cols[2]
        changed_sent = spoil_sentence(
            sentence, lemmas, tags, lang, similarity, max_sub
        )
        if changed_sent:
            spoiled.append(process(changed_sent, classes[i], classify))

    with open(output_path, mode="wt") as fd:
        fd.write(
            pd.DataFrame({"spoiled": spoiled}).to_json(
                orient="records", lines=True
            )
        )


if __name__ == "__main__":
    main()
