#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Optional


BENCHMARK_ORDER = [
    ("VStarBench", "vstar"),
    ("HRBench4K", "hr4k"),
    ("HRBench8K", "hr8k"),
    ("RealWorldQA", "realworld"),
]

# example usage:
    # python scripts/export_eval_latex.py swimbird_2b_thought0_latent__eval --model-prefix "SwimBird-2B-thought0-Latent"


def find_latest_log(ckpt_dir: Path, prefix: str) -> Optional[Path]:
    matches = sorted(ckpt_dir.glob(f"{prefix}_*.out"))
    return matches[-1] if matches else None


def parse_vstar_or_realworld(path: Path) -> Optional[float]:
    pattern = re.compile(r"^\s*Overall\s+([0-9]*\.?[0-9]+)\s*$")
    for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        match = pattern.match(line)
        if match:
            return float(match.group(1))
    return None


def parse_hrbench(path: Path) -> Optional[float]:
    pattern = re.compile(r"^\s*\d+\s+Average\s+all\s+([0-9]*\.?[0-9]+)\s*$")
    for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        match = pattern.match(line)
        if match:
            return float(match.group(1))
    return None


def format_score(score: Optional[float]) -> str:
    if score is None:
        return "--"
    return f"{score * 100:.1f}"


def extract_checkpoint_number(ckpt_dir: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", ckpt_dir.name)
    if not match:
        raise ValueError(f"Unexpected checkpoint directory name: {ckpt_dir}")
    return int(match.group(1))


def collect_scores(ckpt_dir: Path) -> list[Optional[float]]:
    scores = []
    for benchmark, kind in BENCHMARK_ORDER:
        log_path = find_latest_log(ckpt_dir, benchmark)
        if log_path is None:
            scores.append(None)
            continue

        if kind in {"vstar", "realworld"}:
            scores.append(parse_vstar_or_realworld(log_path))
        else:
            scores.append(parse_hrbench(log_path))
    return scores


def build_row(model_prefix: str, ckpt_dir: Path) -> str:
    ckpt_num = extract_checkpoint_number(ckpt_dir)
    scores = collect_scores(ckpt_dir)
    avg = None if any(score is None for score in scores) else sum(scores) / len(scores)

    cols = [f"\\textbf{{{model_prefix}-{ckpt_num}}}"]
    cols.extend(format_score(score) for score in scores)
    cols.append(format_score(avg))
    return " & ".join(cols) + r" \\"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan checkpoint eval logs and export LaTeX table rows."
    )
    parser.add_argument(
        "eval_dir",
        nargs="?",
        default="swimbird_2b_thought0_latent__eval",
        help="Directory that contains checkpoint-* subdirectories.",
    )
    parser.add_argument(
        "--model-prefix",
        default="swimbird_2b_thought0_latent",
        help="Prefix used in the LaTeX row name.",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    ckpt_dirs = sorted(
        [path for path in eval_dir.glob("checkpoint-*") if path.is_dir()],
        key=extract_checkpoint_number,
    )

    if not ckpt_dirs:
        raise SystemExit(f"No checkpoint directories found under {eval_dir}")

    for ckpt_dir in ckpt_dirs:
        print(build_row(args.model_prefix, ckpt_dir))


if __name__ == "__main__":
    main()
    
    
