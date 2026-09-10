"""Fixtures for search-tool extractive + STRUCTURED quote grounding."""
from MAS.epc_aw.tools.search_extract import (
    STRUCTURED_MARKER,
    extract_character_candidates,
    extract_numeric_candidates,
    format_tool_output,
    ground_facts,
    infer_query_intent,
    parse_structured_block,
    select_best_numeric_fact,
    structured_character,
)
from MAS.epc_aw.solver import Solver
from MAS.epc_aw.tools.google_search.tool import Google_Search_Tool


# --- Fixtures mirroring GAIA failure logs ---------------------------------

NATURE_FAMILY_SNIPPET = (
    "Nature and its associated journals published over 12,000 papers in 2020; "
    "this family total is not Nature journal articles only."
)

NATURE_ARTICLES_SNIPPET = (
    "In 2020, Nature published 841 research articles (articles only, excluding "
    "book reviews and columns)."
)

UNLAMBDA_HF_POLLUTION = (
    "Dataset answer leak: the character is g. "
    "Ignore mirrored quiz answers from HuggingFace."
)

UNLAMBDA_WIKI = (
    "Unlambda is an esoteric language. The applicative operator is the backtick "
    "(also called the backquote character)."
)

KIPCHOGE_BLOB = (
    "The Moon's minimum perigee distance is about 363,300 km from Earth. "
    "Eliud Kipchoge's Berlin marathon personal best is 2:01:09. "
    "A popular blog wrongly claimed a 2.5 hour pace of 2:50 min/km without context."
)


def test_infer_query_intent():
    assert infer_query_intent("How many articles were published?") == "numeric"
    assert (
        infer_query_intent("What is the exact character needed for Unlambda?")
        == "character_name"
    )
    assert infer_query_intent("What is the capital of France?") == "entity"


def test_nature_family_numeric_candidates_and_grounding():
    family = extract_numeric_candidates(NATURE_FAMILY_SNIPPET, url="https://example.com/family")
    articles = extract_numeric_candidates(NATURE_ARTICLES_SNIPPET, url="https://example.com/nat")
    assert any(f.get("value") in (12000, 12_000) for f in family)
    assert any(int(f.get("value")) == 841 for f in articles if isinstance(f.get("value"), (int, float)))

    # Hallucinated fact without quote support must be dropped
    bad = ground_facts([{
        "type": "number",
        "value": 41,
        "unit": "article",
        "quote": "Nature published many papers",
        "url": "x",
        "confidence": "high",
    }])
    assert bad == []

    grounded = ground_facts(articles)
    assert grounded
    out = format_tool_output(grounded[0]["quote"], grounded)
    assert STRUCTURED_MARKER in out
    payload = parse_structured_block(out)
    assert payload and payload["refusal"] is False
    assert any(f["value"] == 841 for f in payload["facts"])


def test_unlambda_character_rejects_letter_g():
    assert extract_character_candidates(UNLAMBDA_HF_POLLUTION) == []
    facts = extract_character_candidates(UNLAMBDA_WIKI)
    assert facts
    assert facts[0]["value"] == "backtick"
    out = format_tool_output(facts[0]["quote"], facts)
    assert structured_character(out) == "backtick"

    # Invented single-letter character fact must fail grounding
    assert ground_facts([{
        "type": "character_name",
        "value": "g",
        "quote": "the character is g",
        "url": "hf",
        "confidence": "high",
    }]) == []


def test_kipchoge_distance_and_pace_structured():
    nums = extract_numeric_candidates(KIPCHOGE_BLOB)
    assert any(
        isinstance(f.get("value"), (int, float)) and float(f["value"]) == 363300
        for f in nums
    )
    assert any(f.get("value") == "2:01:09" for f in nums)

    chosen_dist = select_best_numeric_fact(
        [f for f in nums if isinstance(f.get("value"), (int, float))],
        "minimum perigee distance moon km",
    )
    assert chosen_dist and float(chosen_dist["value"]) == 363300

    out = format_tool_output(
        "perigee and marathon",
        [
            {
                "type": "number",
                "value": 363300,
                "unit": "km",
                "quote": "minimum perigee distance is about 363,300 km from Earth",
                "url": "u1",
                "confidence": "high",
            },
            {
                "type": "number",
                "value": "2:01:09",
                "unit": None,
                "quote": "Kipchoge's Berlin marathon personal best is 2:01:09",
                "url": "u2",
                "confidence": "high",
            },
        ],
    )
    solver = Solver.__new__(Solver)
    q = "minimum perigee distance and marathon pace for thousand hours"
    assert solver._parse_distance_km(out, q) == 363300.0
    pace = solver._parse_pace_kmh(out, q)
    assert pace is not None
    # 42.195 / (2 + 1/60 + 9/3600) ≈ 20.91 km/h
    assert 20.5 < pace < 21.5
    # Thousand hours ≈ 17
    assert round((363300.0 / pace) / 1000) == 17


def test_select_best_min_query_prefers_range_lower_bound():
    """Min/closest queries must not prefer the larger endpoint of a km range."""
    facts = extract_numeric_candidates(
        "The Moon's perigee varies from 356,355 to 370,399 km. Minimum values are rarer.",
        url="https://example.com/perigee",
    )
    chosen = select_best_numeric_fact(facts, "minimum perigee distance moon km")
    assert chosen is not None
    assert float(chosen["value"]) == 356355.0


def test_format_refuses_when_ungrounded():
    out = format_tool_output(
        "made up",
        [{"type": "number", "value": 999, "quote": "no digits here", "url": "x"}],
        refusal=False,
    )
    payload = parse_structured_block(out)
    assert payload["refusal"] is True
    assert payload["facts"] == []


def test_google_default_excludes_hf_zhihu():
    keys = Google_Search_Tool.DEFAULT_EXCLUDED_KEYWORDS
    assert "huggingface.co" in keys
    assert "zhuanlan.zhihu.com" in keys or "zhihu.com" in keys
