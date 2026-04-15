#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# example usage:
# python scripts/analyze_trainset_stats.py \
#   --input \
#   SwimBird-SFT-92K/SwimBird-ZebraCoT \
#   SwimBird-SFT-92K/SwimBird-ThinkMorph \
#   SwimBird-SFT-92K/SwimBird-MathCanvas \
#   SwimBird-SFT-92K/SwimBird-OpenMMReasoner

IMAGE_MARKER = "<image>"
SKIP_FILE_PATTERNS = (
    "_stats.json",
    "_rejected.json",
    "_bak.json",
    "_thought0_latent_",
    "_segment_0_plan",
)


def should_skip_file(path: Path, include_derived: bool) -> bool:
    if include_derived:
        return False
    return any(pattern in path.name for pattern in SKIP_FILE_PATTERNS)


def iter_json_files(paths: list[Path], include_derived: bool):
    for path in paths:
        if path.is_file() and path.suffix == ".json":
            if not should_skip_file(path, include_derived):
                yield path
        elif path.is_dir():
            for file_path in sorted(path.glob("*.json")):
                if file_path.is_file() and not should_skip_file(file_path, include_derived):
                    yield file_path
        else:
            raise FileNotFoundError(f"Unsupported input path: {path}")


def normalize_conversations(conversations):
    if isinstance(conversations, list):
        return conversations
    if isinstance(conversations, dict):
        keys = list(conversations.keys())
        if not keys:
            return []
        length = len(conversations[keys[0]])
        return [{k: conversations[k][i] for k in keys} for i in range(length)]
    return []


def split_segments(text: str):
    parts = re.split(r"(<image>)", text or "")
    text_segments = [part.strip() for part in parts if part != IMAGE_MARKER and part.strip()]
    image_segments = sum(1 for part in parts if part == IMAGE_MARKER)
    return text_segments, image_segments


def classify_sample(sample: dict):
    conversations = normalize_conversations(sample.get("conversations", []))
    human_turn = next((turn for turn in conversations if turn.get("from") == "human"), {})
    gpt_turn = next((turn for turn in conversations if turn.get("from") == "gpt"), {})

    human_text = human_turn.get("value", "")
    gpt_text = gpt_turn.get("value", "")

    user_text_segments, user_image_segments = split_segments(human_text)
    assistant_text_segments, assistant_image_segments = split_segments(gpt_text)

    answer = sample.get("answer", "")
    answer_present = bool(str(answer).strip())

    user_image_count = len(sample.get("image", [])) if isinstance(sample.get("image", []), list) else 0
    reasoning_image_count = (
        len(sample.get("reasoning_image", []))
        if isinstance(sample.get("reasoning_image", []), list)
        else 0
    )

    vision_or_interleave = sample.get("vision_or_interleave", None)

    if assistant_image_segments > 0 and not assistant_text_segments:
        assistant_pattern = "image_only"
    elif assistant_image_segments == 0 and assistant_text_segments:
        assistant_pattern = "text_only"
    elif assistant_image_segments > 0 and assistant_text_segments:
        assistant_pattern = "image_and_text"
    else:
        assistant_pattern = "empty"

    if user_image_count == 0 and reasoning_image_count == 0 and assistant_pattern == "text_only":
        high_level_pattern = "pure_text"
    elif assistant_pattern == "image_only":
        high_level_pattern = "image_only_answer"
    elif assistant_pattern == "image_and_text":
        high_level_pattern = "image_text_answer"
    elif assistant_pattern == "text_only":
        high_level_pattern = "text_answer"
    else:
        high_level_pattern = "other"

    return {
        "vision_or_interleave": vision_or_interleave,
        "user_text_segments": len(user_text_segments),
        "user_image_segments": user_image_segments,
        "assistant_text_segments": len(assistant_text_segments),
        "assistant_image_segments": assistant_image_segments,
        "assistant_pattern": assistant_pattern,
        "high_level_pattern": high_level_pattern,
        "user_image_count": user_image_count,
        "reasoning_image_count": reasoning_image_count,
        "answer_present": answer_present,
    }


def init_stats():
    return {
        "samples": 0,
        "vision_or_interleave": Counter(),
        "assistant_pattern": Counter(),
        "high_level_pattern": Counter(),
        "answer_present": 0,
        "user_image_count_sum": 0,
        "reasoning_image_count_sum": 0,
        "user_text_segments_sum": 0,
        "user_image_segments_sum": 0,
        "assistant_text_segments_sum": 0,
        "assistant_image_segments_sum": 0,
        "total_segments_sum": 0,
    }


def update_stats(stats: dict, features: dict):
    stats["samples"] += 1
    stats["vision_or_interleave"][str(features["vision_or_interleave"])] += 1
    stats["assistant_pattern"][features["assistant_pattern"]] += 1
    stats["high_level_pattern"][features["high_level_pattern"]] += 1
    stats["answer_present"] += int(features["answer_present"])
    stats["user_image_count_sum"] += features["user_image_count"]
    stats["reasoning_image_count_sum"] += features["reasoning_image_count"]
    stats["user_text_segments_sum"] += features["user_text_segments"]
    stats["user_image_segments_sum"] += features["user_image_segments"]
    stats["assistant_text_segments_sum"] += features["assistant_text_segments"]
    stats["assistant_image_segments_sum"] += features["assistant_image_segments"]
    stats["total_segments_sum"] += (
        features["user_text_segments"]
        + features["user_image_segments"]
        + features["assistant_text_segments"]
        + features["assistant_image_segments"]
    )


def pct(value: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{100.0 * value / total:.2f}%"


def print_stats(name: str, stats: dict):
    total = stats["samples"]
    if total == 0:
        print(f"\n=== {name} ===")
        print("samples: 0")
        return

    print(f"\n=== {name} ===")
    print(f"samples: {total}")
    print(f"answer_present_ratio: {pct(stats['answer_present'], total)}")
    print(
        "vision_or_interleave: "
        + ", ".join(
            f"{k}={v} ({pct(v, total)})"
            for k, v in sorted(stats["vision_or_interleave"].items())
        )
    )
    print(
        "high_level_pattern: "
        + ", ".join(
            f"{k}={v} ({pct(v, total)})"
            for k, v in sorted(stats["high_level_pattern"].items())
        )
    )
    print(
        "assistant_pattern: "
        + ", ".join(
            f"{k}={v} ({pct(v, total)})"
            for k, v in sorted(stats["assistant_pattern"].items())
        )
    )
    print(
        "avg_counts: "
        f"user_images={stats['user_image_count_sum'] / total:.2f}, "
        f"reasoning_images={stats['reasoning_image_count_sum'] / total:.2f}, "
        f"user_text_segments={stats['user_text_segments_sum'] / total:.2f}, "
        f"user_image_segments={stats['user_image_segments_sum'] / total:.2f}, "
        f"assistant_text_segments={stats['assistant_text_segments_sum'] / total:.2f}, "
        f"assistant_image_segments={stats['assistant_image_segments_sum'] / total:.2f}, "
        f"total_segments={stats['total_segments_sum'] / total:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Quickly summarize dataset content statistics for SwimBird trainsets."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input json files or directories containing json files.",
    )
    parser.add_argument(
        "--include-derived",
        action="store_true",
        help="Include derived/helper files such as *_bak.json, *_rejected.json, and *_stats.json.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    per_file = {}
    overall = init_stats()
    per_group = defaultdict(init_stats)

    for file_path in iter_json_files(input_paths, include_derived=args.include_derived):
        stats = init_stats()
        data = json.load(open(file_path, "r", encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]

        for sample in data:
            features = classify_sample(sample)
            update_stats(stats, features)
            update_stats(overall, features)
            update_stats(per_group[file_path.parent.name], features)

        per_file[str(file_path)] = stats

    print_stats("OVERALL", overall)
    for group_name in sorted(per_group):
        print_stats(f"GROUP {group_name}", per_group[group_name])
    for file_name in sorted(per_file):
        print_stats(file_name, per_file[file_name])


if __name__ == "__main__":
    main()
