# csm_batch.py

import json
import math
import os
from collections import Counter
from typing import Dict, List, Tuple

DEFAULT_SPATIAL_SCALE = "Spatial Scale.json"
DEFAULT_CHANGE_EXTENT = "Change Extent.json"
DEFAULT_GT = "LevirCCcaptions.json"
INPUT_JSON = "json/xxxx_Levir_CC_gate.json"


def tokenize(sentence: str) -> List[str]:
    return sentence.lower().strip().split()


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def level(word: str, levels: Dict[str, int]) -> int:
    return int(levels.get(word, 0))


def step_penalty(delta_level: int) -> float:
    gap = abs(int(delta_level))
    if gap == 0:
        base = -1.0
    elif gap == 1:
        base = -2.0
    else:
        base = -3.0
    return base


def token_similarity_single(ref_token: str, cand_token: str, levels: Dict[str, int]) -> float:
    if ref_token == cand_token:
        return 1.0
    return step_penalty(level(cand_token, levels) - level(ref_token, levels))


def token_similarity_dual(
    ref_token: str,
    cand_token: str,
    spatial_scale_levels: Dict[str, int],
    change_extent_levels: Dict[str, int],
) -> float:
    if ref_token == cand_token:
        return 1.0
    spatial_score = token_similarity_single(ref_token, cand_token, spatial_scale_levels)
    extent_score = token_similarity_single(ref_token, cand_token, change_extent_levels)
    return 0.5 * (spatial_score + extent_score)


def ngram_similarity_dual(
    ref_ngram: Tuple[str, ...],
    cand_ngram: Tuple[str, ...],
    spatial_scale_levels: Dict[str, int],
    change_extent_levels: Dict[str, int],
) -> float:
    scores = [
        token_similarity_dual(ref_token, cand_token, spatial_scale_levels, change_extent_levels)
        for ref_token, cand_token in zip(ref_ngram, cand_ngram)
    ]
    return sum(scores) / len(scores)


def best_soft_match_excluding_exact(
    cand_ngram: Tuple[str, ...],
    ref_ngrams: List[Tuple[str, ...]],
    spatial_scale_levels: Dict[str, int],
    change_extent_levels: Dict[str, int],
) -> float:
    best = None
    for ref_ngram in ref_ngrams:
        if ref_ngram == cand_ngram:
            continue
        value = ngram_similarity_dual(ref_ngram, cand_ngram, spatial_scale_levels, change_extent_levels)
        if best is None or value > best:
            best = value
    if best is None:
        return step_penalty(0)
    return best


def csm_precision_n_multi(
    ref_list: List[List[str]],
    cand_tokens: List[str],
    n: int,
    spatial_scale_levels: Dict[str, int],
    change_extent_levels: Dict[str, int],
) -> Tuple[float, float]:
    """
    Returns (raw_precision, clipped_precision).
    1. Exact n-grams within reference capacity contribute 1.
    2. Remaining n-grams receive the best non-exact soft match score, clipped at <= 0.
    3. The final precision used by CSM is max(0, raw_precision).
    """
    cand_ngrams = ngrams(cand_tokens, n)
    if not cand_ngrams:
        return 0.0, 0.0

    ref_ngrams_all: List[Tuple[str, ...]] = []
    ref_max_counts: Counter = Counter()
    for ref_tokens in ref_list:
        ref_ngrams = ngrams(ref_tokens, n)
        ref_ngrams_all.extend(ref_ngrams)
        ref_counts = Counter(ref_ngrams)
        for ref_ngram, count in ref_counts.items():
            if count > ref_max_counts.get(ref_ngram, 0):
                ref_max_counts[ref_ngram] = count

    used_exact: Counter = Counter()
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

    raw_precision = sum(contributions) / float(len(cand_ngrams))
    clipped_precision = max(0.0, raw_precision)
    return raw_precision, clipped_precision


def csm_k_multi(
    ref_list: List[List[str]],
    cand_tokens: List[str],
    k: int,
    spatial_scale_levels: Dict[str, int],
    change_extent_levels: Dict[str, int],
) -> Tuple[float, List[float], List[float]]:
    """CSM-k = BP * exp(mean(log(clipped_precision_n)))."""
    if not ref_list:
        return 0.0, [], []

    raw_precisions, clipped_precisions = [], []
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
    closest_ref_len = min(ref_lens, key=lambda ref_len: abs(ref_len - candidate_len)) if ref_lens else 0
    brevity_penalty = 1.0 if candidate_len > closest_ref_len else math.exp(1.0 - float(closest_ref_len) / max(candidate_len, 1))

    log_sum = sum(math.log(max(1e-12, precision)) for precision in clipped_precisions)
    geo_mean = math.exp(log_sum / max(1, k))
    return float(brevity_penalty * geo_mean), raw_precisions, clipped_precisions


def load_multi_refs_from_gt(gt_json_path: str) -> Dict[str, List[List[str]]]:
    with open(gt_json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    name_to_refs: Dict[str, List[List[str]]] = {}
    for item in data.get("images", []):
        filename = str(item.get("filename", "")).strip()
        if not filename:
            continue

        refs_tokens: List[List[str]] = []
        for sentence in item.get("sentences", []):
            tokens = sentence.get("tokens", None)
            if tokens and isinstance(tokens, list):
                refs_tokens.append([str(word).lower() for word in tokens])
            else:
                raw = str(sentence.get("raw", "")).strip()
                if raw:
                    refs_tokens.append(tokenize(raw))

        if refs_tokens:
            name_to_refs[filename] = refs_tokens
    return name_to_refs


def lookup_refs_with_png_fallback(name_to_refs: Dict[str, List[List[str]]], raw_filename: str):
    raw_filename = str(raw_filename).strip()
    base = os.path.basename(raw_filename)
    candidates = [raw_filename, base]

    root, ext = os.path.splitext(raw_filename)
    if ext == "":
        candidates.append(raw_filename + ".png")
        candidates.append(base + ".png")
    elif ext.lower() != ".png":
        candidates.append(root + ".png")
        base_root, _ = os.path.splitext(base)
        candidates.append(base_root + ".png")

    for candidate in candidates:
        refs = name_to_refs.get(candidate)
        if refs:
            return refs, candidate
    return None, None


def load_prediction_items(pred_json_path: str) -> List[Dict[str, str]]:
    with open(pred_json_path, "r", encoding="utf-8") as file:
        pred_obj = json.load(file)

    if isinstance(pred_obj, list):
        return pred_obj
    if isinstance(pred_obj, dict) and "results" in pred_obj:
        return pred_obj["results"]
    raise ValueError("Unsupported prediction JSON format: expected a list or a dict with a 'results' field.")


def evaluate_prediction_file(
    pred_json_path: str,
    gt_json_path: str,
    spatial_scale_levels: Dict[str, int],
    change_extent_levels: Dict[str, int],
) -> Dict[str, float]:
    pred_items = load_prediction_items(pred_json_path)
    name_to_refs = load_multi_refs_from_gt(gt_json_path)

    samples = 0
    csm_sums = [0.0, 0.0, 0.0, 0.0]

    for item in pred_items:
        filename = str(item.get("filename", "")).strip()
        sentence = str(item.get("sentence", "")).strip()
        if not filename or not sentence:
            continue

        refs, _ = lookup_refs_with_png_fallback(name_to_refs, filename)
        if not refs:
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

    mean_csm = [value / samples for value in csm_sums] if samples else [0.0] * 4

    return {
        "csm_1": mean_csm[0],
        "csm_2": mean_csm[1],
        "csm_3": mean_csm[2],
        "csm_4": mean_csm[3],
    }


def print_result(row: Dict[str, float]) -> None:
    print("CSM_1   CSM_2   CSM_3   CSM_4")
    print(
        f"{float(row['csm_1']):.6f} {float(row['csm_2']):.6f} "
        f"{float(row['csm_3']):.6f} {float(row['csm_4']):.6f}"
    )


def main() -> None:
    with open(DEFAULT_SPATIAL_SCALE, "r", encoding="utf-8") as file:
        spatial_scale_levels: Dict[str, int] = json.load(file)
    with open(DEFAULT_CHANGE_EXTENT, "r", encoding="utf-8") as file:
        change_extent_levels: Dict[str, int] = json.load(file)

    row = evaluate_prediction_file(
        INPUT_JSON,
        DEFAULT_GT,
        spatial_scale_levels,
        change_extent_levels,
    )
    print_result(row)


if __name__ == "__main__":
    main()
