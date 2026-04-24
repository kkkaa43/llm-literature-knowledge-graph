from src.text_cleaner import clean_text


def test_clean_text_preserves_section_headings() -> None:
    raw_text = """
    1 Introduction

    Graph learning is useful.

    Method

    The model uses message passing.
    """

    cleaned = clean_text(raw_text)

    assert "1 INTRODUCTION" in cleaned
    assert "METHOD" in cleaned
    assert "Graph learning is useful." in cleaned


def test_clean_text_repairs_hyphenated_line_breaks() -> None:
    cleaned = clean_text("The represen-\ntation is learned from graph data.")

    assert "representation" in cleaned
    assert "represen-" not in cleaned
