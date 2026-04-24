from src.utils import safe_filename


def test_safe_filename_removes_unsafe_characters() -> None:
    filename = safe_filename("Retrieval-Augmented Generation: What/Why? * 2024")

    assert filename == "Retrieval-Augmented_Generation_WhatWhy_2024"


def test_safe_filename_falls_back_for_empty_values() -> None:
    assert safe_filename("/// ***") == "untitled"
