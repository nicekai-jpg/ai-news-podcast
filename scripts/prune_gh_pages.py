#!/usr/bin/env python3
"""Prune gh-pages branch to only retain files for episodes in data/episodes.json,

and reset git history of the gh-pages branch to release deleted space.
"""

import json

# Setup logging
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prune_gh_pages")


def run_cmd(
    cmd: list[str], cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    log.info("Running command: %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if check and res.returncode != 0:
        log.error(
            "Command failed: %s\nStdout:\n%s\nStderr:\n%s", " ".join(cmd), res.stdout, res.stderr
        )
        raise RuntimeError(f"Command failed with exit code {res.returncode}")
    return res


def main() -> int:
    # 1. Paths setup
    root_dir = Path(__file__).resolve().parents[1]
    episodes_json_path = root_dir / "data" / "episodes.json"
    site_dir = root_dir / "site"

    if not episodes_json_path.exists():
        log.error("episodes.json not found at: %s", episodes_json_path)
        return 1

    # 2. Get valid episode IDs from index
    try:
        with open(episodes_json_path, "r", encoding="utf-8") as f:
            episodes = json.load(f)
    except Exception as e:
        log.error("Failed to load episodes.json: %s", e)
        return 1

    # episodes.json might be a list or a dict containing a list
    if isinstance(episodes, dict):
        episodes_list = episodes.get("episodes", [])
    elif isinstance(episodes, list):
        episodes_list = episodes
    else:
        episodes_list = []

    valid_ids = {
        str(ep["id"]).strip() for ep in episodes_list if isinstance(ep, dict) and "id" in ep
    }
    log.info("Loaded %d valid episode IDs: %s", len(valid_ids), sorted(list(valid_ids)))

    if not valid_ids:
        log.error(
            "No valid episode IDs found in episodes.json, aborting to prevent clearing all assets."
        )
        return 1

    # 3. Environment check
    github_token = os.environ.get("GITHUB_TOKEN")
    github_repo = os.environ.get("GITHUB_REPOSITORY")
    if not github_token or not github_repo:
        log.error("Missing required environment variables: GITHUB_TOKEN or GITHUB_REPOSITORY")
        return 1

    # 4. Clone gh-pages branch to temp folder
    clone_dir = root_dir / "tmp_gh_pages_clone"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    remote_url = f"https://x-access-token:{github_token}@github.com/{github_repo}.git"
    log.info("Cloning gh-pages branch...")
    try:
        run_cmd(
            ["git", "clone", "--branch", "gh-pages", "--single-branch", remote_url, str(clone_dir)]
        )
    except Exception as e:
        log.warning(
            "Failed to clone gh-pages branch directly (possibly branch doesn't exist yet): %s", e
        )
        log.info("Will attempt to initialize a new gh-pages branch from scratch.")
        # If gh-pages doesn't exist, we'll create the directory and initialize it as a git repo
        clone_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(["git", "init"], cwd=clone_dir)
        run_cmd(["git", "checkout", "-b", "gh-pages"], cwd=clone_dir)
        run_cmd(["git", "remote", "add", "origin", remote_url], cwd=clone_dir)

    # 5. Backup valid audio and chunk directories from clone if they exist
    backup_dir = root_dir / "tmp_backup_episodes"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    source_episodes_dir = clone_dir / "episodes"
    backed_up_count = 0
    if source_episodes_dir.exists():
        log.info("Backing up valid episode resources from cloned gh-pages...")
        for eid in valid_ids:
            # Backup files (.mp3, .html, .txt)
            for suffix in [".mp3", ".html", ".txt"]:
                file_name = f"{eid}{suffix}"
                file_path = source_episodes_dir / file_name
                if file_path.exists():
                    shutil.copy2(file_path, backup_dir / file_name)
                    backed_up_count += 1

            # Backup chunk folder
            chunk_folder = source_episodes_dir / eid
            if chunk_folder.exists() and chunk_folder.is_dir():
                shutil.copytree(chunk_folder, backup_dir / eid)
                log.info("Backed up chunk folder for episode: %s", eid)
                backed_up_count += 1
    log.info("Backup complete. Total items backed up: %d", backed_up_count)

    # 6. Rebuild gh-pages branch as a clean orphan branch
    log.info("Rebuilding gh-pages branch as a clean orphan...")
    run_cmd(["git", "checkout", "--orphan", "temp-orphan-pages"], cwd=clone_dir)
    # Remove all tracked and untracked files
    run_cmd(["git", "rm", "-rf", "."], cwd=clone_dir, check=False)

    # Clean up directory thoroughly
    for item in clone_dir.iterdir():
        if item.name == ".git":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # 7. Copy static site structure from main's site/ dir (if exists)
    if site_dir.exists():
        log.info("Copying static site assets from main's site/...")
        for item in site_dir.iterdir():
            if item.name == "episodes":  # we will handle episodes separately
                continue
            if item.is_dir():
                shutil.copytree(item, clone_dir / item.name)
            else:
                shutil.copy2(item, clone_dir / item.name)

    # 8. Restore backed up valid episodes
    target_episodes_dir = clone_dir / "episodes"
    target_episodes_dir.mkdir(parents=True, exist_ok=True)

    # Copy from backup
    log.info("Restoring valid episodes to the new gh-pages workspace...")
    for item in backup_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, target_episodes_dir / item.name)
        else:
            shutil.copy2(item, target_episodes_dir / item.name)

    # Also check if main's site/episodes has anything generated *today* that we should include
    # (since the script could run after a new episode is built locally but before pushed to gh-pages)
    main_episodes_dir = site_dir / "episodes"
    if main_episodes_dir.exists():
        log.info("Merging newly generated episodes from site/episodes/...")
        for item in main_episodes_dir.iterdir():
            # Check if this item matches one of the valid ids
            is_valid = False
            for eid in valid_ids:
                if (
                    item.name == f"{eid}.mp3"
                    or item.name == f"{eid}.html"
                    or item.name == f"{eid}.txt"
                    or item.name == eid
                ):
                    is_valid = True
                    break

            if is_valid:
                target_path = target_episodes_dir / item.name
                if target_path.exists():
                    if target_path.is_dir():
                        shutil.rmtree(target_path)
                    else:
                        target_path.unlink()

                if item.is_dir():
                    shutil.copytree(item, target_path)
                else:
                    shutil.copy2(item, target_path)

    # 9. Commit and force push to origin gh-pages
    log.info("Configuring git user...")
    run_cmd(["git", "config", "user.name", "github-actions[bot]"], cwd=clone_dir)
    run_cmd(
        ["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"],
        cwd=clone_dir,
    )

    log.info("Committing changes...")
    run_cmd(["git", "add", "-A"], cwd=clone_dir)
    run_cmd(
        ["git", "commit", "-m", "chore: prune gh-pages to latest 30 days and reset history"],
        cwd=clone_dir,
    )

    log.info("Force pushing to gh-pages...")
    run_cmd(["git", "push", "origin", "temp-orphan-pages:gh-pages", "--force"], cwd=clone_dir)

    # 10. Cleanup local temporary dirs
    log.info("Cleaning up temporary local directories...")
    shutil.rmtree(clone_dir, ignore_errors=True)
    shutil.rmtree(backup_dir, ignore_errors=True)

    log.info("Success! gh-pages branch has been pruned and git history reset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
