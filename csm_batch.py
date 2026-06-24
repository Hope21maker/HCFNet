"""Batch evaluation for the Change-Sensitive Metric (CSM).

The implementation uses two word-level rule dictionaries:
1. Spatial Scale
2. Change Extent

Prediction files may be JSON arrays, dictionaries containing a ``results``
array, or JSON Lines files with one prediction object per line.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SPATIAL_SCALE = BASE_DIR / "CSM Level Rules" / "Spatial Scale.json"
DEFAULT_CHANGE_EXTENT = BASE_DIR / "CSM Level Rules" / "Change Extent.json"
DEFAULT_GT = BASE_DIR / "GTjson" / "whuCCcaptions.json"
DEFAULT_PRED = BASE_DIR / "CapRjson" / "WHU.json"
EPSILON = 1e-12

TokenList = List[str]
Ngram = Tuple[str, ...]
LevelRules = Dict[str, int]


def tokenize(sentence: str) -> TokenList:
    """Apply the whitespace tokenization used by the released CSM results."""
    return sentence.lower().strip().split()


def ngrams(tokens: Sequence[str], n: int) -> List[Ngram]:
    if n < 1:
        raise ValueError("n must be at least 1.")
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def level(word: str, levels: LevelRules) -> int:
    return int(levels.get(word, 0))


def step_penalty(delta_level: int) -> float:
    """Return the CSM mismatch penalty for a semantic-level gap."""
    gap = abs(int(delta_level))
    if gap == 0:
        return -1.0
    if gap == 1:
        return -2.0
    return -3.0


def token_similarity_single(
    ref_token: str,
    cand_token: str,
    levels: LevelRules,
) -> float:
    if ref_token == cand_token:
        return 1.0
    return step_penalty(level(cand_token, levels) - level(ref_token, levels))


def token_similarity_dual(
    ref_token: str,
    cand_token: str,
    spatial_scale_levels: LevelRules,
    change_extent_levels: LevelRules,
) -> float:
    if ref_token == cand_token:
        return 1.0
    spatial_score = token_similarity_single(
        ref_token, cand_token, spatial_scale_levels
    )
    extent_score = token_similarity_single(
        ref_token, cand_token, change_extent_levels
    )
    return 0.5 * (spatial_score + extent_score)


def ngram_similarity_dual(
    ref_ngram: Ngram,
    cand_ngram: Ngram,
    spatial_scale_levels: LevelRules,
    change_extent_levels: LevelRules,
) -> float:
    scores = [
        token_similarity_dual(
            ref_token,
            cand_token,
            spatial_scale_levels,
            change_extent_levels,
        )
        for ref_token, cand_token in zip(ref_ngram, cand_ngram)
    ]
    return sum(scores) / len(scores)


def best_soft_match_excluding_exact(
    cand_ngram: Ngram,
    ref_ngrams: Sequence[Ngram],
    spatial_scale_levels: LevelRules,
    change_extent_levels: LevelRules,
) -> float:
    best = None
    for ref_ngram in ref_ngrams:
        if ref_ngram == cand_ngram:
            continue
        value = ngram_similarity_dual(
            ref_ngram,
            cand_ngram,
            spatial_scale_levels,
            change_extent_levels,
        )
        if best is None or value > best:
            best = value
    return step_penalty(0) if best is None else best


def csm_precision_n_multi(
    ref_list: Sequence[TokenList],
    cand_tokens: TokenList,
    n: int,
    spatial_scale_levels: LevelRules,
    change_extent_levels: LevelRules,
) -> Tuple[float, float]:
    """Return the raw and non-negative clipped CSM precision for order ``n``."""
    cand_ngrams = ngrams(cand_tokens, n)
    if not cand_ngrams:
        return 0.0, 0.0

    ref_ngrams_all: List[Ngram] = []
    ref_max_counts: Counter[Ngram] = Counter()
    for ref_tokens in ref_list:
        ref_ngrams = ngrams(ref_tokens, n)
        ref_ngrams_all.extend(ref_ngrams)
        ref_counts = Counter(ref_ngrams)
        for ref_ngram, count in ref_counts.items():
            if count > ref_max_counts.get(ref_ngram, 0):
                ref_max_counts[ref_ngram] = count

    used_exact: Counter[Ngram] = Counter()
    contributions: List[float] = []

    for cand_ngram in cand_ngrams:
        if used_exact[cand_ngram] < ref_max_counts.get(cand_ngram, 0):
            contributions.append(1.0)
            used_exact[cand_ngram] += 1
            continue

        best = best_soft_match_excluding_exact(
            cand_ngram,
            ref_ngrams_all,
            spatial_scale_levels,
            change_extent_levels,
        )
        contributions.append(min(0.0, float(best)))

    raw_precision = sum(contributions) / len(cand_ngrams)
    return raw_precision, max(0.0, raw_precision)


def csm_k_multi(
    ref_list: Sequence[TokenList],
    cand_tokens: TokenList,
    k: int,
    spatial_scale_levels: LevelRules,
    change_extent_levels: LevelRules,
) -> Tuple[float, List[float], List[float]]:
    """Compute CSM-k with multi-reference clipping and a brevity penalty."""
    if not 1 <= k <= 4:
        raise ValueError("k must be between 1 and 4.")
    if not ref_list or not cand_tokens:
        return 0.0, [], []

    raw_precisions: List[float] = []
    clipped_precisions: List[float] = []
    for n in range(1, k + 1):
        raw_precision, clipped_precision = csm_precision_n_multi(
            ref_list,
            cand_tokens,
            n,
            spatial_scale_levels,
            change_extent_levels,
        )
        raw_precisions.append(raw_precision)
        clipped_precisions.append(clipped_precision)

    candidate_len = len(cand_tokens)
    ref_lens = [len(ref_tokens) for ref_tokens in ref_list]
    closest_ref_len = min(ref_lens, key=lambda ref_len: abs(ref_len - candidate_len))
    brevity_penalty = (
        1.0
        if candidate_len > closest_ref_len
        else math.exp(1.0 - float(closest_ref_len) / candidate_len)
    )

    log_sum = sum(
        math.log(max(EPSILON, precision)) for precision in clipped_precisions
    )
    geometric_mean = math.exp(log_sum / k)
    return (
        float(brevity_penalty * geometric_mean),
        raw_precisions,
        clipped_precisions,
    )


def load_json(path: Path) -> Any:
    try:
        # utf-8-sig accepts ordinary UTF-8 and files written with a BOM by
        # Windows tools such as PowerShell.
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc


def load_level_rules(path: Path) -> LevelRules:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Level-rule file must contain a JSON object: {path}")

    rules: LevelRules = {}
    for word, value in data.items():
        if not isinstance(word, str) or not isinstance(value, int):
            raise ValueError(
                f"Level rules must map strings to integers; invalid entry in {path}: "
                f"{word!r}: {value!r}"
            )
        rules[word.lower()] = value
    return rules


def load_multi_refs_from_gt(gt_json_path: Path) -> Dict[str, List[TokenList]]:
    data = load_json(gt_json_path)
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise ValueError(
            "Ground-truth JSON must be an object containing an 'images' array."
        )

    name_to_refs: Dict[str, List[TokenList]] = {}
    for item in data["images"]:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename", "")).strip()
        if not filename:
            continue

        refs_tokens: List[TokenList] = []
        for sentence in item.get("sentences", []):
            if not isinstance(sentence, dict):
                continue
            sentence_tokens = sentence.get("tokens")
            if isinstance(sentence_tokens, list) and sentence_tokens:
                refs_tokens.append(
                    [str(word).lower() for word in sentence_tokens]
                )
                continue
            raw = str(sentence.get("raw", "")).strip()
            if raw:
                refs_tokens.append(tokenize(raw))

        if refs_tokens:
            name_to_refs[filename] = refs_tokens

    if not name_to_refs:
        raise ValueError(f"No valid references were found in {gt_json_path}.")
    return name_to_refs


def lookup_refs_with_png_fallback(
    name_to_refs: Dict[str, List[TokenList]],
    raw_filename: str,
) -> Tuple[List[TokenList] | None, str | None]:
    raw_filename = str(raw_filename).strip()
    base = os.path.basename(raw_filename)
    candidates = [raw_filename, base]

    root, ext = os.path.splitext(raw_filename)
    if not ext:
        candidates.extend([raw_filename + ".png", base + ".png"])
    elif ext.lower() != ".png":
        candidates.append(root + ".png")
        base_root, _ = os.path.splitext(base)
        candidates.append(base_root + ".png")

    for candidate in dict.fromkeys(candidates):
        refs = name_to_refs.get(candidate)
        if refs:
            return refs, candidate
    return None, None


def load_prediction_items(pred_json_path: Path) -> List[Dict[str, Any]]:
    """Load a JSON array/results object, with JSON Lines as a fallback."""
    try:
        pred_obj = load_json(pred_json_path)
    except ValueError as json_error:
        items: List[Dict[str, Any]] = []
        try:
            with pred_json_path.open("r", encoding="utf-8-sig") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"JSON Lines item {line_number} is not an object."
                        )
                    items.append(item)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"{pred_json_path} is neither valid JSON nor valid JSON Lines."
            ) from exc
        if not items:
            raise json_error
        return items

    if isinstance(pred_obj, list):
        items = pred_obj
    elif isinstance(pred_obj, dict) and isinstance(pred_obj.get("results"), list):
        items = pred_obj["results"]
    else:
        raise ValueError(
            "Prediction JSON must be an array or an object containing a "
            "'results' array."
        )

    if not all(isinstance(item, dict) for item in items):
        raise ValueError("Every prediction item must be a JSON object.")
    return items


def evaluate_prediction_file(
    pred_json_path: Path,
    gt_json_path: Path,
    spatial_scale_levels: LevelRules,
    change_extent_levels: LevelRules,
) -> Dict[str, Any]:
    pred_items = load_prediction_items(pred_json_path)
    name_to_refs = load_multi_refs_from_gt(gt_json_path)

    samples = 0
    invalid_items = 0
    missing_filenames: List[str] = []
    csm_sums = [0.0, 0.0, 0.0, 0.0]

    for item in pred_items:
        filename = str(item.get("filename", "")).strip()
        sentence = str(item.get("sentence", "")).strip()
        if not filename or not sentence:
            invalid_items += 1
            continue

        refs, _ = lookup_refs_with_png_fallback(name_to_refs, filename)
        if not refs:
            missing_filenames.append(filename)
            continue

        cand_tokens = tokenize(sentence)
        for k in range(1, 5):
            csm_score, _, _ = csm_k_multi(
                refs,
                cand_tokens,
                k,
                spatial_scale_levels,
                change_extent_levels,
            )
            csm_sums[k - 1] += csm_score
        samples += 1

    if samples == 0:
        raise ValueError(
            "No valid prediction/reference pairs were found. Check filenames "
            "and input formats."
        )

    mean_csm = [value / samples for value in csm_sums]
    return {
        "csm_1": mean_csm[0],
        "csm_2": mean_csm[1],
        "csm_3": mean_csm[2],
        "csm_4": mean_csm[3],
        "prediction_items": len(pred_items),
        "evaluated_samples": samples,
        "invalid_items": invalid_items,
        "missing_references": len(missing_filenames),
        "missing_filenames": missing_filenames,
    }


def print_result(result: Dict[str, Any], scale: float) -> None:
    print(f"Prediction items : {result['prediction_items']}")
    print(f"Evaluated samples: {result['evaluated_samples']}")
    print(f"Invalid items    : {result['invalid_items']}")
    print(f"Missing refs     : {result['missing_references']}")
    print()
    for k in range(1, 5):
        print(f"CSM-{k}: {result[f'csm_{k}'] * scale:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate remote-sensing change captions with CSM-1 to CSM-4."
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=DEFAULT_PRED,
        help=f"Prediction JSON/JSONL file (default: {DEFAULT_PRED})",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=DEFAULT_GT,
        help=f"Ground-truth caption JSON (default: {DEFAULT_GT})",
    )
    parser.add_argument(
        "--spatial-scale",
        type=Path,
        default=DEFAULT_SPATIAL_SCALE,
        help=f"Spatial Scale rules (default: {DEFAULT_SPATIAL_SCALE})",
    )
    parser.add_argument(
        "--change-extent",
        type=Path,
        default=DEFAULT_CHANGE_EXTENT,
        help=f"Change Extent rules (default: {DEFAULT_CHANGE_EXTENT})",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print scores on the 0-1 scale instead of the default 0-100 scale.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not fail when predictions have invalid items or missing references.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        spatial_scale_levels = load_level_rules(args.spatial_scale)
        change_extent_levels = load_level_rules(args.change_extent)
        result = evaluate_prediction_file(
            args.pred,
            args.gt,
            spatial_scale_levels,
            change_extent_levels,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_result(result, scale=1.0 if args.raw else 100.0)

    unmatched = result["invalid_items"] + result["missing_references"]
    if unmatched and not args.allow_missing:
        print(
            "\nError: some predictions were not evaluated. Re-run with "
            "--allow-missing only if this is intentional."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
