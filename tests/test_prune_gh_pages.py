"""Tests for scripts/prune_gh_pages.py.

scripts/ 不是包,这里用 importlib 直接按路径加载脚本模块。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prune_gh_pages.py"

_spec = importlib.util.spec_from_file_location("prune_gh_pages", _SCRIPT)
prune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prune)


def _res(returncode: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class TestEnsureGhPagesClone:
    def test_aborts_when_remote_unreachable(self, tmp_path: Path, monkeypatch) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd, cwd=None, check=True):
            calls.append(cmd)
            return _res(returncode=128, stderr="network reset")

        monkeypatch.setattr(prune, "run_cmd", fake_run)

        with pytest.raises(RuntimeError, match="Cannot reach remote"):
            prune.ensure_gh_pages_clone(tmp_path, "https://example.com/repo.git")

        # 只允许 ls-remote 探测,绝不能走到 init/clone
        assert len(calls) == 1
        assert calls[0][:2] == ["git", "ls-remote"]

    def test_initializes_fresh_branch_when_missing(self, tmp_path: Path, monkeypatch) -> None:
        calls: list[tuple[list[str], Path | None]] = []

        def fake_run(cmd, cwd=None, check=True):
            calls.append((cmd, cwd))
            return _res(stdout="")

        monkeypatch.setattr(prune, "run_cmd", fake_run)

        clone_dir = prune.ensure_gh_pages_clone(tmp_path, "https://example.com/repo.git")

        assert clone_dir == tmp_path / "tmp_gh_pages_clone"
        assert clone_dir.exists()
        cmds = [c for c, _ in calls]
        assert ["git", "ls-remote", "--heads", "https://example.com/repo.git", "gh-pages"] in cmds
        assert ["git", "init"] in cmds
        assert ["git", "checkout", "-b", "gh-pages"] in cmds
        assert ["git", "remote", "add", "origin", "https://example.com/repo.git"] in cmds
        assert not any(c[:2] == ["git", "clone"] for c in cmds)

    def test_clones_when_branch_exists(self, tmp_path: Path, monkeypatch) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd, cwd=None, check=True):
            calls.append(cmd)
            return _res(stdout="deadbeef refs/heads/gh-pages\n")

        monkeypatch.setattr(prune, "run_cmd", fake_run)

        clone_dir = prune.ensure_gh_pages_clone(tmp_path, "https://example.com/repo.git")

        assert clone_dir == tmp_path / "tmp_gh_pages_clone"
        assert calls[-1] == [
            "git",
            "clone",
            "--branch",
            "gh-pages",
            "--single-branch",
            "https://example.com/repo.git",
            str(clone_dir),
        ]
        assert not any(c[:2] == ["git", "init"] for c in calls)


class TestRestoreEpisodes:
    def test_dirs_skip_files_overwrite(self, tmp_path: Path) -> None:
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        backup_chunk = backup_dir / "2024-01-01"
        backup_chunk.mkdir()
        (backup_chunk / "chunk-001.mp3").write_text("chunk audio")
        (backup_chunk / "playlist.json").write_text('{"backup": true}')
        (backup_dir / "2024-01-01.mp3").write_text("backup mp3")
        (backup_dir / "2024-01-01.txt").write_text("backup txt")

        site_dir = tmp_path / "site"
        main_ep = site_dir / "episodes"
        main_ep.mkdir(parents=True)
        # main 侧只有 playlist.json 的残缺切片目录(CI checkout 的典型形态)
        partial_chunk = main_ep / "2024-01-01"
        partial_chunk.mkdir()
        (partial_chunk / "playlist.json").write_text('{"partial": true}')
        # main 侧更新过的普通文件,应当覆盖备份
        (main_ep / "2024-01-01.txt").write_text("fresh txt")
        # 有效 id 的新文件,应当被合并进来
        (main_ep / "2024-01-02.mp3").write_text("new mp3")
        # 不在索引里的文件必须被过滤掉
        (main_ep / "2024-02-30.txt").write_text("invalid")

        clone_dir = tmp_path / "clone"
        clone_dir.mkdir()
        valid_ids = {"2024-01-01", "2024-01-02"}

        prune.restore_episodes(backup_dir, site_dir, clone_dir, valid_ids)

        episodes = clone_dir / "episodes"
        chunk = episodes / "2024-01-01"
        # 目录跳过:备份的完整切片目录不能被 main 侧残缺目录覆盖
        assert (chunk / "chunk-001.mp3").read_text() == "chunk audio"
        assert (chunk / "playlist.json").read_text() == '{"backup": true}'
        # 文件覆盖:main 侧最新的普通文件胜出
        assert (episodes / "2024-01-01.txt").read_text() == "fresh txt"
        # 新的有效文件被合并
        assert (episodes / "2024-01-02.mp3").read_text() == "new mp3"
        # 备份里的文件在 main 侧没有更新时保留
        assert (episodes / "2024-01-01.mp3").read_text() == "backup mp3"
        # 无效 id 被过滤
        assert not (episodes / "2024-02-30.txt").exists()


class TestLoadValidEpisodeIds:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert prune.load_valid_episode_ids(tmp_path / "episodes.json") is None

    def test_dict_payload(self, tmp_path: Path) -> None:
        path = tmp_path / "episodes.json"
        path.write_text('{"episodes": [{"id": "2024-03-01"}, {"id": "2024-03-02"}]}')
        assert prune.load_valid_episode_ids(path) == {"2024-03-01", "2024-03-02"}

    def test_list_payload(self, tmp_path: Path) -> None:
        path = tmp_path / "episodes.json"
        path.write_text('[{"id": "2024-03-01"}]')
        assert prune.load_valid_episode_ids(path) == {"2024-03-01"}


class TestBackupValidEpisodes:
    def test_backs_up_assets_and_chunk_dir(self, tmp_path: Path) -> None:
        clone_dir = tmp_path / "clone"
        src = clone_dir / "episodes"
        src.mkdir(parents=True)
        (src / "2024-01-01.mp3").write_text("mp3")
        (src / "2024-01-01.html").write_text("notes")
        (src / "2024-01-01.txt").write_text("script")
        (src / "2024-01-01").mkdir()
        (src / "2024-01-01" / "playlist.json").write_text("{}")
        # 不在 valid_ids 里的文件不应被备份
        (src / "2024-01-02.mp3").write_text("dropped")

        backup_dir = prune.backup_valid_episodes(tmp_path, clone_dir, {"2024-01-01"})

        assert (backup_dir / "2024-01-01.mp3").read_text() == "mp3"
        assert (backup_dir / "2024-01-01.html").read_text() == "notes"
        assert (backup_dir / "2024-01-01.txt").read_text() == "script"
        assert (backup_dir / "2024-01-01" / "playlist.json").read_text() == "{}"
        assert not (backup_dir / "2024-01-02.mp3").exists()
