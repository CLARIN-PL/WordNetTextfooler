"""Utility functions."""
import importlib


def get_model_and_tokenizer(dataset_name):
    """Return get_model_and_tokenizer for a specific dataset."""
    fun = getattr(
        importlib.import_module(f"text_attacks.models.{dataset_name}"),
        "get_model_and_tokenizer",
    )
    return fun()


def get_classify_function(dataset_name):
    """Return get_model_and_tokenizer for a specific dataset."""
    fun = getattr(
        importlib.import_module(f"text_attacks.models.{dataset_name}"),
        "get_classify_function",
    )
    return fun()


def download_dataset(dataset_name):
    fun = getattr(
        importlib.import_module(f"text_attacks.datasets.{dataset_name}"),
        "download_dataset",
    )
    return fun()
