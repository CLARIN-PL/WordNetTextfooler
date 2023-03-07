"""Utility functions."""
import importlib


def get_model_and_tokenizer(dataset_name):
    """Return get_model_and_tokenizer for a specific dataset."""
    fun = getattr(
        importlib.import_module(f"text_attacks.models.{dataset_name}"),
        "get_model_and_tokenizer",
    )
    return fun()

    
