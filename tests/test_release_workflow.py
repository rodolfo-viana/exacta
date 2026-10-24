"""Exercise publication guards against real temporary Git histories."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_release.py"


@pytest.fixture
def repository(tmp_path):
    env = dict(os.environ, GIT_AUTHOR_NAME="Rodolfo Viana", GIT_AUTHOR_EMAIL="eu@rodolfoviana.com.br",
               GIT_COMMITTER_NAME="Rodolfo Viana", GIT_COMMITTER_EMAIL="eu@rodolfoviana.com.br")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, env=env, text=True,
                                       stderr=subprocess.PIPE).strip()

    git("init", "-b", "main")
    source = tmp_path / "src" / "exacta" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text('__version__ = "0.0.1a4"\n')
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "-m", "Prepare release fixture")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    return tmp_path, source, git


def check(repository, tag="v0.0.1a4", ref=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--ref", ref or f"refs/tags/{tag}", "--tag", tag],
        cwd=repository[0], capture_output=True, text=True,
    )


@pytest.mark.parametrize("annotated", [False, True])
def test_accepts_merged_release_tags_after_main_advances(repository, annotated):
    _, _, git = repository
    git("tag", *(('-a', '-m', 'Release fixture') if annotated else ()), "v0.0.1a4")
    commit = git("rev-parse", "HEAD")
    git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "Next development change")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    git("checkout", "--detach", commit)
    result = check(repository)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "version=0.0.1a4\n"


def test_rejects_unmerged_feature_tag(repository):
    _, _, git = repository
    git("checkout", "-b", "feature/unmerged")
    git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "Unmerged change")
    git("tag", "v0.0.1a4")
    result = check(repository)
    assert result.returncode != 0
    assert "already be merged" in result.stderr
    assert result.stdout == ""


def test_rejects_branch_ref(repository):
    result = check(repository, ref="refs/heads/main")
    assert result.returncode != 0
    assert "version tag" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("version", ["0.0.1a4.dev0", "0.0.1a4+local", "0.0.1alpha4", "invalid"])
def test_rejects_invalid_release_versions(repository, version):
    _, source, git = repository
    source.write_text(f'__version__ = "{version}"\n')
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "-m", "Invalid release fixture")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    git("tag", f"v{version}")
    result = check(repository, tag=f"v{version}")
    assert result.returncode != 0
    assert "Release rejected" in result.stderr
    assert result.stdout == ""


def test_rejects_tag_source_mismatch(repository):
    repository[2]("tag", "v0.0.1a5")
    result = check(repository, tag="v0.0.1a5")
    assert result.returncode != 0
    assert "source __version__" in result.stderr
    assert result.stdout == ""


def test_rejects_wrong_checked_out_commit(repository):
    _, _, git = repository
    git("tag", "v0.0.1a4")
    git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "Another commit")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    result = check(repository)
    assert result.returncode != 0
    assert "does not match the release tag" in result.stderr
    assert result.stdout == ""


def test_rejects_missing_main_history(repository):
    _, _, git = repository
    git("tag", "v0.0.1a4")
    git("update-ref", "-d", "refs/remotes/origin/main")
    result = check(repository)
    assert result.returncode != 0
    assert "fetch full history" in result.stderr
    assert result.stdout == ""
