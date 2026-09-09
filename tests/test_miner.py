from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from git import Actor, GitCommandError, Repo

from sentinel.miner import (
    CSV_COLUMNS,
    CloneError,
    EmptyRepositoryError,
    InvalidRepositoryURLError,
    NoJavaFilesError,
    clone_repository,
    export_to_csv,
    mine_local_repository,
    validate_github_url,
)


def commit_files(repo: Repo, message: str, files: dict[str, str]) -> str:
    work_tree = Path(repo.working_tree_dir or "")
    for relative_path, content in files.items():
        path = work_tree / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    repo.index.add(list(files))
    actor = Actor("Sentinel Tester", "tester@example.com")
    commit = repo.index.commit(message, author=actor, committer=actor)
    return commit.hexsha


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not-a-url",
        "http://github.com/owner/repo.git",
        "https://gitlab.com/owner/repo.git",
        "https://github.com/owner",
        "https://github.com/owner/repo/issues",
        "https://github.com/owner/repo.git?tab=readme",
    ],
)
def test_validate_github_url_rejects_invalid_urls(url: str) -> None:
    with pytest.raises(InvalidRepositoryURLError):
        validate_github_url(url)


def test_validate_github_url_normalizes_repository() -> None:
    repository = validate_github_url("https://github.com/apache/commons-lang.git")

    assert repository.slug == "apache/commons-lang"
    assert repository.clone_url == "https://github.com/apache/commons-lang.git"


def test_clone_failure_has_domain_specific_error(monkeypatch, tmp_path: Path) -> None:
    repository = validate_github_url("https://github.com/owner/repository.git")

    def fail_clone(*args, **kwargs):
        raise GitCommandError("clone", 128, stderr="repository not found")

    monkeypatch.setattr(Repo, "clone_from", fail_clone)

    with pytest.raises(CloneError, match="Could not clone owner/repository"):
        clone_repository(repository, tmp_path / "clone")


def test_mine_local_repository_extracts_only_java_changes(tmp_path: Path) -> None:
    repo = Repo.init(tmp_path / "repository")
    first_hash = commit_files(
        repo,
        "Add application",
        {
            "src/main/java/App.java": "class App {\n    int value = 1;\n}\n",
            "README.md": "# Example\n",
        },
    )
    second_hash = commit_files(
        repo,
        "Update application and docs",
        {
            "src/main/java/App.java": "class App {\n    int value = 2;\n}\n",
            "README.md": "# Updated example\n",
        },
    )

    result = mine_local_repository(repo, "owner/repository")

    assert list(result.columns) == CSV_COLUMNS
    assert len(result) == 2
    assert set(result["commit_hash"]) == {first_hash, second_hash}
    assert set(result["file_path"]) == {"src/main/java/App.java"}
    assert set(result["repository"]) == {"owner/repository"}
    assert set(result["author_name"]) == {"Sentinel Tester"}
    assert set(result["author_email"]) == {"tester@example.com"}
    assert result["lines_added"].sum() == 4
    assert result["lines_deleted"].sum() == 1
    assert all("T" in value for value in result["commit_date"])


def test_empty_repository_raises_clear_error(tmp_path: Path) -> None:
    repo = Repo.init(tmp_path / "empty")

    with pytest.raises(EmptyRepositoryError, match="has no commits"):
        mine_local_repository(repo, "owner/empty")


def test_repository_without_java_changes_raises_clear_error(tmp_path: Path) -> None:
    repo = Repo.init(tmp_path / "python-only")
    commit_files(repo, "Add script", {"app.py": "print('hello')\n"})

    with pytest.raises(NoJavaFilesError, match="no Java file changes"):
        mine_local_repository(repo, "owner/python-only")


def test_export_to_csv_preserves_column_order(tmp_path: Path) -> None:
    row = {column: "value" for column in CSV_COLUMNS}
    row["lines_added"] = 3
    row["lines_deleted"] = 1
    data = pd.DataFrame([row], columns=CSV_COLUMNS)

    destination = export_to_csv(data, tmp_path / "exports" / "history.csv")
    exported = pd.read_csv(destination)

    assert destination.exists()
    assert list(exported.columns) == CSV_COLUMNS
    assert exported.loc[0, "lines_added"] == 3

