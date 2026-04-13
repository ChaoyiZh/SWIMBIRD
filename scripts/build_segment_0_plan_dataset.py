#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Iterable


IMAGE_MARKER = "<image>"
PLAN_START = "<|plan_start|>"
PLAN_END = "<|plan_end|>"
LATENT = "<|latent|>"


def make_plan_span(plan_length: int) -> str:
    return PLAN_START + (LATENT * plan_length) + PLAN_END


def iter_input_files(input_paths: list[Path]) -> Iterable[Path]:
    for path in input_paths:
        if path.is_file() and path.suffix == ".json":
            yield path
        elif path.is_dir():
            for file_path in sorted(path.glob("*.json")):
                if file_path.is_file():
                    yield file_path
        else:
            raise FileNotFoundError(f"Unsupported input path: {path}")


def replace_first_visible_reasoning_segment(text: str, plan_span: str):
    parts = re.split(r"(<image>)", text)
    replaced = False
    original_segment = None
    updated_parts = []

    for part in parts:
        if replaced or part == IMAGE_MARKER:
            updated_parts.append(part)
            continue

        if part.strip():
            leading_ws = re.match(r"^\s*", part).group(0)
            trailing_ws = re.search(r"\s*$", part).group(0)
            original_segment = part.strip()
            updated_parts.append(f"{leading_ws}{plan_span}{trailing_ws}")
            replaced = True
        else:
            updated_parts.append(part)

    return "".join(updated_parts), replaced, original_segment


def transform_sample(sample: dict, plan_span: str):
    conversations = sample.get("conversations", [])
    if not isinstance(conversations, list):
        return sample, False, "non_list_conversations", None, None

    transformed = False
    original_segment = None
    new_gpt_value = None
    new_sample = dict(sample)
    new_conversations = []

    for turn in conversations:
        new_turn = dict(turn)
        if turn.get("from") == "gpt":
            gpt_value = turn.get("value", "")
            new_value, replaced, old_segment = replace_first_visible_reasoning_segment(
                gpt_value, plan_span
            )
            if replaced:
                transformed = True
                original_segment = old_segment
                new_gpt_value = new_value
                new_turn["value"] = new_value
        new_conversations.append(new_turn)

    if not transformed:
        return sample, False, "no_visible_reasoning_segment", None, None

    new_sample["conversations"] = new_conversations
    return new_sample, True, "transformed", original_segment, new_gpt_value


def build_output_path(output_root: Path, input_file: Path) -> Path:
    return output_root / input_file.parent.name / input_file.name


def main():
    parser = argparse.ArgumentParser(
        description="Build an offline segment_0_plan training dataset variant."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input json files or directories containing json files.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root directory for processed dataset files.",
    )
    parser.add_argument(
        "--plan-length",
        type=int,
        default=8,
        help="Number of <|latent|> tokens inside the plan span.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Number of transformed samples to print for debugging.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plan_span = make_plan_span(args.plan_length)

    stats = {
        "transform_version": "segment_0_plan_v1",
        "plan_length": args.plan_length,
        "input_paths": [str(p) for p in input_paths],
        "output_root": str(output_root),
        "files_processed": 0,
        "total_samples": 0,
        "transformed_samples": 0,
        "unchanged_samples": 0,
        "reason_breakdown": {},
        "output_files": [],
    }
    preview_budget = args.preview

    for input_file in iter_input_files(input_paths):
        data = json.load(open(input_file, "r", encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]

        processed = []
        file_transformed = 0
        stats["files_processed"] += 1

        for sample in data:
            new_sample, transformed, reason, original_segment, new_gpt_value = transform_sample(
                sample, plan_span
            )
            processed.append(new_sample)
            stats["total_samples"] += 1
            stats["reason_breakdown"][reason] = stats["reason_breakdown"].get(reason, 0) + 1

            if transformed:
                stats["transformed_samples"] += 1
                file_transformed += 1
                if preview_budget > 0:
                    sample_id = sample.get("id", "<no-id>")
                    print("=" * 80)
                    print(f"[preview] file={input_file}")
                    print(f"[preview] sample_id={sample_id}")
                    print(f"[preview] original_first_segment={original_segment[:400]!r}")
                    print(f"[preview] transformed_gpt_prefix={new_gpt_value[:600]!r}")
                    preview_budget -= 1
            else:
                stats["unchanged_samples"] += 1

        output_file = build_output_path(output_root, input_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        stats["output_files"].append(
            {
                "input_file": str(input_file),
                "output_file": str(output_file),
                "samples": len(processed),
                "transformed_samples": file_transformed,
            }
        )

    stats["transform_ratio"] = (
        stats["transformed_samples"] / stats["total_samples"]
        if stats["total_samples"]
        else 0.0
    )

    stats_path = output_root / f"segment_0_plan_stats_plan{args.plan_length}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[done] segment_0_plan dataset build finished")
    print(f"[done] output_root={output_root}")
    print(f"[done] stats_path={stats_path}")
    print(f"[done] total_samples={stats['total_samples']}")
    print(f"[done] transformed_samples={stats['transformed_samples']}")
    print(f"[done] unchanged_samples={stats['unchanged_samples']}")
    print(f"[done] transform_ratio={stats['transform_ratio']:.4f}")
    print(f"[done] reason_breakdown={stats['reason_breakdown']}")


if __name__ == "__main__":
    main()
