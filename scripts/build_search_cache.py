#!/usr/bin/env python3
"""Build data/search_cache.json — offline search cache for RL rollouts.

Background: this host cannot reach en.wikipedia.org / hf.co directly
(hf-mirror.com works). DDG/Bing are bot-risk-controlled. During training
the agents' search tools must hit the local cache: publish
data/search_cache.json keyed as:

  "query::<condensed-query>" -> SERP-style text ("wikipedia results for: ..."
  / "bing results for: ..." with numbered title + URL + snippet entries)
  "url::<page-url>"          -> page body plain text (<=2000 chars)

Query keys are built EXACTLY like user_space/*/adapted/context.py's
_condense_query() (strong+weak normalization), which is the shared
contract with mas/tools/search_cache.py (developed in parallel).

Sources, in fallback order (never crash the whole run):
  1. HF mirror: HF_ENDPOINT=https://hf-mirror.com, datasets streaming over
     "hotpotqa/hotpot_qa" "distractor" "validation"; match each question
     by title-then-8gram overlap; render context paragraphs as SERP text.
  2. Local corpus cross-fill from user_space/projects/epc-aw-main/upload/EPC-AW-main/test/hotpotqa/data/data.json
     (has no context in this snapshot -> effectively a no-op, kept for the
     case where a future snapshot carries context) plus data/hotpotqa.json.
  3. Gaps: any question still missing content lands in
     data/search_cache.gaps.json ({"<qid>": ["<missing query>", ...]}) and its
     query keys are filled with the explicit placeholder
     "search_unavailable_in_cache" (NOT an empty string, so the consumer can
     distinguish cache miss from explicit unavailability).

Idempotent: existing keys are kept (incremental build); --rebuild wipes and
rebuilds; --limit N builds only the first N questions per dataset (smoke).

Do NOT touch mas/tools/* — that is the other worker's contract surface.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
CACHE_PATH = DATA_DIR / "search_cache.json"
GAPS_PATH = DATA_DIR / "search_cache.gaps.json"

HOTPOT_PATH = DATA_DIR / "hotpotqa.json"
BAMBOOGLE_PATH = DATA_DIR / "bamboogle.json"
REF_CORPUS_PATH = REPO / "user_space" / "projects" / "epc-aw-main" / "upload" / "EPC-AW-main" / "test" / "hotpotqa" / "data" / "data.json"

HF_ENDPOINT = "https://hf-mirror.com"
HF_DATASET = "hotpotqa/hotpot_qa"
HF_CONFIG = "distractor"
HF_SPLIT = "validation"
HF_STREAM_ROWS = 12000  # validation distractor is 7405 rows; enough for 100 Qs

MAX_BODY_CHARS = 2000
PLACEHOLDER = "search_unavailable_in_cache"
MAX_SENTENCES_PER_BODY = 4  # 2-4 sentence page body per the contract


# --------------------------------------------------------------------------
# condense_query — MUST stay byte-identical in behavior with the two
# projects' adapted/context.py `_condense_query()`. Kept as a verbatim copy
# (not an import) so this script has zero dependency on mas/ or user_space/
# import-time side effects.
# --------------------------------------------------------------------------

_INSTRUCTION_TAIL_RE = re.compile(
    r"(?i)\??\s*when\s+ready\s*,?\s*output\s+the\s+final\s+answer.*$"
    r"|\?\s*when\s+ready\b.*$"
    r"|do\s+not\s+generate\s+any\s+content\b.*$",
    re.S,
)

_IMPERATIVE_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:please\s+)?"
    r"(?:search\s+(?:the\s+web|wikipedia|the\s+internet|online)?\s*for"
    r"|look\s*up"
    r"|find\s+(?:information\s+)?(?:about|on)"
    r"|fetch\s+(?:the\s+)?(?:url|page|website)"
    r"|compute"
    r"|calculate"
    r")\s*[:：]?\s*"
)

_QUERY_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "was", "are", "were", "be", "been", "his", "her", "its", "their",
    "before", "after", "answering", "please", "prefer",
    "search", "web", "internet", "online", "look", "up", "find", "about",
    "when", "ready", "output", "final", "answer", "generate", "any",
    "content", "tag", "tags", "enclosed", "after", "do", "not", "did",
    "what", "which", "who", "where", "whom", "whose", "how",
}


def _condense_query(query: str) -> str:
    """Verbatim port of the projects' _condense_query (shared key contract)."""
    q = str(query or "").strip()
    if not q:
        return q
    text = _INSTRUCTION_TAIL_RE.sub("", q)
    text = _IMPERATIVE_PREFIX_RE.sub("", text).strip()
    if len(text) <= 48 and "?" not in text:
        return text
    text = text.split("?", 1)[0]
    text = re.sub(
        r"(?i)^(what|which|who|where|when|whom|whose|how)\s+"
        r"(is|was|are|were|did|does|do|has|have|had)\b",
        " ", text,
    )
    text = re.sub(
        r"(?i)\b(the actor who|the actress who|the person who|who played"
        r"|that appeared in|which appeared in|use (wikipedia|web|google)[^.]*\.?)",
        " ", text,
    )
    text = re.sub(r"[?.!]+$", " ", text)
    text = re.sub(r"^(?:What|Which|Who|Where|When|Whom|Whose|How)\s+", "", text, flags=re.I)
    words = text.split()
    keep: List[str] = []
    for i, w in enumerate(words):
        lowered = w.lower().strip(".,;:!?\"'()")
        if lowered in _QUERY_STOPWORDS:
            continue
        is_entity = bool(re.match(r"^[A-Z0-9]", w)) or w.isdigit()
        is_year = bool(re.match(r"^\d{4}$", w))
        if is_entity and not is_year:
            keep.append(w)
        elif is_year and keep:
            keep.append(w)
        elif keep and len(keep) <= 5 and w not in ("played", "role", "title"):
            continue
    keep2 = [w for w in (keep or words) if w.lower().strip(".,;:!?\"'()") not in _QUERY_STOPWORDS]
    if len(keep2) > 8:
        keep2 = keep2[:8]
    out = " ".join(keep2).strip()
    return out or q


# --------------------------------------------------------------------------
# Query extraction: 2-3 variants per question, all written as query:: keys.
# --------------------------------------------------------------------------

def question_stem(question: str) -> str:
    """Question without the 'When ready...' instruction tail."""
    return _INSTRUCTION_TAIL_RE.sub("", str(question or "").strip()).strip()


def stopwords_short(stem: str, max_words: int = 8) -> str:
    """Content-word version of the stem (entities + key nouns)."""
    words = stem.split("?", 1)[0]
    words = re.sub(r"[?.!]+$", " ", words)
    out: List[str] = []
    for w in words.split():
        if w.lower().strip(".,;:!?\"'()") in _QUERY_STOPWORDS:
            continue
        out.append(w)
    if len(out) > max_words:
        out = out[:max_words]
    return " ".join(out).strip()


def entity_sequence(stem: str) -> str:
    """Sequential run of capitalized / quoted / year tokens, e.g.
    'Oliver Reed Royal Flash' or 'Citibank'."""
    text = stem.split("?", 1)[0]
    tokens = re.findall(r"[A-Z][\w'-]*|\d{4}|\"[^\"]+\"|'[^']+'", text)
    out = [t for t in tokens if t.lower().strip("\"'") not in _QUERY_STOPWORDS]
    # dedupe preserving order
    seen = set()
    uniq: List[str] = []
    for t in out:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t.strip("\"'"))
    return " ".join(uniq[:8]).strip()


def build_query_variants(question: str) -> List[str]:
    """2-3 condensed query variants per question (deduplicated)."""
    stem = question_stem(question)
    variants: List[str] = []

    # a) whole-stem condensed (this is exactly what _condense_query produces
    #    when the planner forwards the full question — the primary hit path)
    full = _condense_query(stem)
    if full and full not in variants:
        variants.append(full)

    # b) stopword-stripped short version, then condensed (planner may send a
    #    trimmed question on later turns; keys must still hit)
    short = _condense_query(stopwords_short(stem))
    if short and short not in variants:
        variants.append(short)

    # c) entity-name extraction version
    ent = _condense_query(entity_sequence(stem))
    if ent and ent not in variants:
        variants.append(ent)

    return variants[:3]


# --------------------------------------------------------------------------
# Rendering — must match mas/tools/wikipedia.py wikipedia_search() output:
# "wikipedia results for: <query>\n1. <title>\n   URL: <url>" (+ snippets).
# --------------------------------------------------------------------------

def wiki_url(title: str) -> str:
    t = str(title or "").strip().replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{quote(t)}"


def render_wikipedia_results(query: str, entries: List[Tuple[str, str]]) -> str:
    """entries: [(title, body_snippet), ...]"""
    lines = [f"wikipedia results for: {query}"]
    for i, (title, snippet) in enumerate(entries, 1):
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {wiki_url(title)}")
        snippet = " ".join(str(snippet or "").split())
        if snippet:
            lines.append(f"   {snippet[:300]}")
    return "\n".join(lines)


def page_body(sentences: List[str], max_chars: int = MAX_BODY_CHARS) -> str:
    """2-4 sentence page body for a url:: key (<=2000 chars)."""
    body = " ".join(" ".join(str(s) or "" for s in sentences[:MAX_SENTENCES_PER_BODY]).split())
    if len(body) > max_chars:
        body = body[:max_chars]
    return body


# --------------------------------------------------------------------------
# N-gram overlap matcher (question stem <-> corpus paragraph)
# --------------------------------------------------------------------------

def _norm_words(text: str) -> List[str]:
    text = unicodedata.normalize("NFKC", str(text or "").lower())
    return re.findall(r"[a-z0-9]+", text)


def ngrams(words: List[str], n: int) -> set:
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def make_stem_index(stem: str) -> set:
    """8-gram + sorted content-word signature of the question stem."""
    words = _norm_words(question_stem(stem))
    sig: set = ngrams(words, 8)
    content = [w for w in words if w not in _QUERY_STOPWORDS and len(w) > 2]
    sig |= {f"hash::{w}" for w in content}
    return sig


def paragraph_signature(title: str, sentences: List[str]) -> set:
    text = " ".join([str(title or "")] + [str(s) or "" for s in (sentences or [])])
    words = _norm_words(text)
    return ngrams(words, 8) | {f"hash::{w}" for w in words if len(w) > 2}


def match_score(stem_index: set, para_sig: set) -> float:
    if not stem_index or not para_sig:
        return 0.0
    inter = len(stem_index & para_sig)
    return inter / len(stem_index)


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------

def load_questions(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("data") or rows.get("questions") or []
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        qid = str(row.get("pid") if row.get("pid") not in (None, "", "None") else row.get("idx", i))
        out.append({
            "qid": f"{path.stem}:{qid}",
            "question": str(row.get("question") or row.get("query") or ""),
        })
    if limit is not None:
        out = out[:limit]
    return out


def load_hf_corpus(logger) -> Optional[List[Dict[str, Any]]]:
    """Stream hotpot_qa distractor validation via hf-mirror.

    Returns a list of {'question','title','sentences'} paragraph units or
    None on failure (network/library). Never raises.
    """
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **_):  # type: ignore
            return x
    try:
        import datasets  # heavy import; only when actually needed
    except Exception as exc:
        logger(f"[hf-mirror] datasets unavailable: {exc}")
        return None
    try:
        ds = datasets.load_dataset(
            HF_DATASET, HF_CONFIG, split=HF_SPLIT, streaming=True,
        )
        units: List[Dict[str, Any]] = []
        for row in tqdm(ds, total=HF_STREAM_ROWS, desc="hf-mirror hotpot_qa validation", unit="row"):
            q = str(row.get("question") or "")
            ctx = row.get("context") or {}
            titles = list(ctx.get("title") or [])
            sents_by_title = list(ctx.get("sentences") or [])
            for title, sents in zip(titles, sents_by_title):
                title = str(title or "")
                sentences = [str(s) for s in (sents or [])]
                if not title or not sentences:
                    continue
                units.append({"question": q, "title": title, "sentences": sentences})
            if len(units) >= HF_STREAM_ROWS * 10:  # hard cap safety
                break
        if not units:
            logger("[hf-mirror] stream produced 0 paragraph units")
            return None
        logger(f"[hf-mirror] loaded {len(units)} paragraph units")
        return units
    except Exception as exc:
        logger(f"[hf-mirror] load failed: {type(exc).__name__}: {exc}")
        return None


def load_local_corpus(logger) -> List[Dict[str, Any]]:
    """Fallback tier 2: local snapshots with context paragraphs (may be empty)."""
    units: List[Dict[str, Any]] = []
    for path in (REF_CORPUS_PATH, HOTPOT_PATH):
        try:
            if not path.is_file():
                continue
            rows = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(rows, dict):
                rows = rows.get("data") or []
            for row in rows:
                ctx = row.get("context") or {}
                titles = list(ctx.get("title") or [])
                sents_by_title = list(ctx.get("sentences") or [])
                q = str(row.get("question") or "")
                for title, sents in zip(titles, sents_by_title):
                    title = str(title or "")
                    sentences = [str(s) for s in (sents or [])]
                    if title and sentences:
                        units.append({"question": q, "title": title, "sentences": sentences})
        except Exception as exc:
            logger(f"[local:{path.name}] unusable: {exc}")
    return units


# --------------------------------------------------------------------------
# cache writing
# --------------------------------------------------------------------------

def upsert(cache: Dict[str, str], key: str, value: str, ids_only: set = None) -> str:
    """Idempotent insert. Returns 'kept' | 'written' | 'noop'."""
    if key in cache:
        return "kept"
    cache[key] = value
    return "written"


def write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------
# main build
# --------------------------------------------------------------------------

def build(steps: Dict[str, bool], limit: Optional[int], rebuild: bool) -> Dict[str, Any]:
    logs: List[str] = []

    def logger(msg: str) -> None:
        print(msg, flush=True)
        logs.append(msg)

    cache: Dict[str, str] = {}
    gaps: Dict[str, List[str]] = {}
    if CACHE_PATH.is_file() and not rebuild:
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            logger(f"[cache] existing file: {len(cache)} keys (incremental)")
        except Exception as exc:
            logger(f"[cache] unreadable ({exc}); rebuilding empty")
            cache = {}
    if rebuild and CACHE_PATH.is_file():
        logger("[cache] --rebuild: starting from empty cache")

    hotpot_questions = load_questions(HOTPOT_PATH, limit)
    bamboogle_questions = load_questions(BAMBOOGLE_PATH, limit)
    logger(f"[load] hotpot={len(hotpot_questions)} bamboogle={len(bamboogle_questions)}")

    # ---- tier 1: hf-mirror corpus ----
    corpus: Optional[List[Dict[str, Any]]] = None
    source = "none"
    if steps["hf"]:
        corpus = load_hf_corpus(logger)
        if corpus:
            source = "hf-mirror"
    # ---- tier 2: local snapshots ----
    if not corpus and steps["local"]:
        corpus = load_local_corpus(logger)
        if corpus:
            source = "local-snapshot"
    steps_used: Dict[str, Any] = {"corpus_source": source}

    # Pre-index corpus paragraphs for speed
    para_index: List[Tuple[set, Dict[str, Any]]] = []
    if corpus:
        t0 = time.time()
        for unit in corpus:
            para_index.append((paragraph_signature(unit["title"], unit["sentences"]), unit))
        logger(f"[index] {len(para_index)} paragraphs indexed in {time.time()-t0:.1f}s")

    stats = {"questions": 0, "query_keys": 0, "url_keys": 0,
             "filled": 0, "gap_questions": 0, "placeholder_keys": 0,
             "kept": 0, "corpus_source": source}

    def integrate_question(item: Dict[str, Any], corpus_match: bool = True) -> None:
        """Run fallback chain for one question and write cache keys.

        corpus_match: hotpot questions may hit the HotpotQA corpus.
        bamboogle is NOT in the hotpot_qa corpus — per the build contract its
        query keys are written with the explicit placeholder and recorded in
        gaps.json for later backfill (entity-derived Wikipedia summary
        queries, e.g. "Enrico Macias born"), instead of polluting the cache
        with low-relevance 8-gram matches.
        """
        nonlocal stats
        stats["questions"] += 1
        qid, question = item["qid"], item["question"]
        stem = question_stem(question)
        variants = build_query_variants(question)
        stem_index = make_stem_index(stem)

        # rank corpus paragraphs for this question
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        if para_index and corpus_match:
            # cheap prefilter: direct 8-gram/word overlap only on matched unit question
            for sig, unit in para_index:
                s = match_score(stem_index, sig)
                if s >= 0.04:  # at least ~4% signature overlap
                    candidates.append((s, unit))
            candidates.sort(key=lambda t: t[0], reverse=True)
            candidates = candidates[:8]
            # dedupe by title keeping best score
            seen_titles: set = set()
            dedup: List[Tuple[float, Dict[str, Any]]] = []
            for s, unit in candidates:
                if unit["title"] in seen_titles:
                    continue
                seen_titles.add(unit["title"])
                dedup.append((s, unit))
            candidates = dedup[:5]

        if candidates:
            # render SERP text + url:: bodies
            primary_query = variants[0] if variants else stem
            entries: List[Tuple[str, str]] = []
            for s, unit in candidates:
                # question-bearing paragraph (gold) gets the fuller body
                body_short = unit["sentences"][0] + (
                    (" " + unit["sentences"][1]) if len(unit["sentences"]) > 1 else ""
                )
                entries.append((unit["title"], body_short))
                url_key = f"url::{wiki_url(unit['title'])}"
                st = upsert(cache, url_key, page_body(unit["sentences"]))
                if st == "written":
                    stats["url_keys"] += 1
                elif st == "kept":
                    stats["kept"] += 1
            serp = render_wikipedia_results(primary_query, entries)
            # every variant of this question maps to the same SERP payload,
            # rendered with its own query header
            for v in variants:
                text = serp if v == primary_query else render_wikipedia_results(v, entries)
                k = f"query::{v}"
                st = upsert(cache, k, text)
                if st == "written":
                    stats["query_keys"] += 1
                    stats["filled"] += 1
                elif st == "kept":
                    stats["kept"] += 1
        else:
            # no corpus support (bamboogle not in hotpot_qa corpus, or tiers failed)
            missing: List[str] = []
            for v in variants:
                k = f"query::{v}"
                if k in cache:
                    #	Idempotent re-run: if the existing value is still the
                    # placeholder, this query REMAINS a gap — carry it into
                    # this run's gaps so the gaps file is not dropped.
                    if cache[k] == PLACEHOLDER:
                        missing.append(v)
                    stats["kept"] += 1
                    continue
                st = upsert(cache, k, PLACEHOLDER)
                if st == "written":
                    stats["placeholder_keys"] += 1
                    missing.append(v)
                elif st == "kept":
                    stats["kept"] += 1
            if missing:
                gaps.setdefault(qid, []).extend(missing)
                stats["gap_questions"] += 1
                # bamboogle live Wikipedia attempt is deferred to network
                # recovery: log the probe query for later backfill
                logger(f"[gap] {qid}: {len(missing)} query(ies) placeholdered")

    for item in hotpot_questions:
        integrate_question(item, corpus_match=steps.get("hotpot_corpus_match", True))
    for item in bamboogle_questions:
        # bamboogle is outside the hotpot_qa corpus: query keys -> placeholder
        # + gaps.json entries, to be backfilled by live Wikipedia once the
        # network recovers (contract: "正文空缺标记留给网络恢复期回填").
        integrate_question(item, corpus_match=False)

    # gaps file
    if gaps:
        merged = gaps
        if GAPS_PATH.is_file():
            try:
                prev = json.loads(GAPS_PATH.read_text(encoding="utf-8"))
                for qid, missing in prev.items():
                    if qid not in merged:
                        merged[qid] = missing
            except Exception:
                pass
        write_json_atomic(GAPS_PATH, merged)
        logger(f"[gaps] wrote {GAPS_PATH.name} with {len(merged)} questions (this run: {len(gaps)})")
    elif GAPS_PATH.is_file():
        try:
            GAPS_PATH.unlink()
            logger("[gaps] removed stale gaps file (no gaps this run)")
        except Exception:
            pass

    write_json_atomic(CACHE_PATH, cache)

    stats.update({
        "total_keys": len(cache),
        "query_total": sum(1 for k in cache if k.startswith("query::")),
        "url_total": sum(1 for k in cache if k.startswith("url::")),
        "gaps_total": sum(len(v) for v in gaps.values()),
    })
    steps_used.update(stats)
    steps_used["log_tail"] = logs[-12:]
    return steps_used


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="only build the first N questions of each dataset (smoke)")
    ap.add_argument("--rebuild", action="store_true",
                    help="wipe the cache and rebuild from scratch")
    ap.add_argument("--skip-hf", action="store_true", help="skip hf-mirror tier")
    ap.add_argument("--skip-local", action="store_true", help="skip local snapshot tier")
    ap.add_argument("--hotpot-corpus-match", action="store_true", default=True,
                    help="hotpot: match context paragraphs from the corpus (default on)")
    ap.add_argument("--no-hotpot-corpus-match", dest="hotpot_corpus_match",
                    action="store_false",
                    help="hotpot: also placeholder strategy (debug only)")
    ap.add_argument("--stats-only", action="store_true", help="print cache stats and exit")
    args = ap.parse_args(argv)

    if args.stats_only:
        if CACHE_PATH.is_file():
            c = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            qn = sum(1 for k in c if k.startswith("query::"))
            un = sum(1 for k in c if k.startswith("url::"))
            ph = sum(1 for k, v in c.items() if k.startswith("query::") and v == PLACEHOLDER)
            print(f"cache keys: {len(c)} (query={qn}, url={un}, placeholder={ph})")
            g = json.loads(GAPS_PATH.read_text(encoding="utf-8")) if GAPS_PATH.is_file() else {}
            print(f"gaps: {len(g)} questions / {sum(len(v) for v in g.values())} queries")
        else:
            print("cache file missing")
        return 0

    steps = {"hf": not args.skip_hf, "local": not args.skip_local,
             "hotpot_corpus_match": args.hotpot_corpus_match}
    stats = build(steps, args.limit, args.rebuild)

    print("\n=== build stats ===")
    for k in ("total_keys", "query_total", "url_total", "questions", "filled",
              "placeholder_keys", "kept", "gap_questions", "gaps_total",
              "corpus_source"):
        print(f"  {k}: {stats.get(k)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
