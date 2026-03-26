#!/usr/bin/env python3
"""Stage, commit, and push changes for the current git repository."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_git_command(args: list[str], cwd: Path, capture_output: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=capture_output,
        check=False,
    )


def fail(message: str, exit_code: int = 1) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def get_repo_root(cwd: Path) -> Path:
    result = run_git_command(["rev-parse", "--show-toplevel"], cwd, capture_output=True)
    if result.returncode != 0:
        fail("current directory is not inside a git repository.")
    return Path(result.stdout.strip())


def get_current_branch(repo_root: Path) -> str:
    result = run_git_command(["branch", "--show-current"], repo_root, capture_output=True)
    if result.returncode != 0:
        fail("failed to determine the current branch.")
    branch = result.stdout.strip()
    if not branch:
        fail("detached HEAD is not supported for push.")
    return branch


def has_staged_changes(repo_root: Path) -> bool:
    result = run_git_command(["diff", "--cached", "--quiet"], repo_root)
    if result.returncode == 0:
        return False
    if result.returncode == 1:
        return True
    fail("failed to inspect staged changes.")
    return False


def ensure_remote_exists(repo_root: Path, remote: str) -> None:
    result = run_git_command(["remote"], repo_root, capture_output=True)
    if result.returncode != 0:
        fail("failed to list git remotes.")
    remotes = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if remote not in remotes:
        fail(f"remote '{remote}' does not exist.")


def stage_commit_push(repo_root: Path, message: str, remote: str, branch: str) -> None:
    result = run_git_command(["add", "."], repo_root)
    if result.returncode != 0:
        fail("git add failed.")

    if not has_staged_changes(repo_root):
        print("No staged changes detected after 'git add .'. Nothing to commit.")
        return

    result = run_git_command(["commit", "-m", message], repo_root)
    if result.returncode != 0:
        fail("git commit failed.")

    result = run_git_command(["push", remote, branch], repo_root)
    if result.returncode != 0:
        fail(f"git push failed for '{remote}/{branch}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 'git add .', create a commit, and push to a remote branch."
    )
    parser.add_argument(
        "-m",
        "--message",
        required=True,
        help="Commit message used for 'git commit -m'.",
    )
    parser.add_argument(
        "-r",
        "--remote",
        default="origin",
        help="Remote name to push to. Default: origin.",
    )
    parser.add_argument(
        "-b",
        "--branch",
        help="Branch name to push. Default: current branch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    repo_root = get_repo_root(cwd)
    ensure_remote_exists(repo_root, args.remote)
    branch = args.branch or get_current_branch(repo_root)

    print(f"Repository: {repo_root}")
    print(f"Remote: {args.remote}")
    print(f"Branch: {branch}")

    stage_commit_push(repo_root, args.message, args.remote, branch)
    print("Git add, commit, and push completed successfully.")


if __name__ == "__main__":
    main()
