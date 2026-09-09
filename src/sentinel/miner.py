"""Clone public GitHub repositories and mine Java file change history."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from git import GitCommandError, Repo

CSV_COLUMNS = [
    "repository",
    "commit_hash",
    "author_name",
    "author_email",
    "commit_date",
    "commit_message",
    "file_path",
    "lines_added",
    "lines_deleted",
]

_OWNER_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class RepositoryMiningError(Exception):
    """Base class for expected repository mining failures."""


class InvalidRepositoryURLError(RepositoryMiningError):
    """Raised when an input is not a supported public GitHub URL."""


class CloneError(RepositoryMiningError):
    """Raised when Git cannot clone the requested repository."""


class EmptyRepositoryError(RepositoryMiningError):
    """Raised when a repository has no commits."""


class NoJavaFilesError(RepositoryMiningError):
    """Raised when no Java file changes exist in repository history."""


@dataclass(frozen=True)
class GitHubRepository:
    """Validated identity for a public GitHub repository."""

    owner: str
    name: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"


def validate_github_url(repository_url: str) -> GitHubRepository:
    """Validate and normalize a public HTTPS GitHub repository URL."""
    if not isinstance(repository_url, str) or not repository_url.strip():
        raise InvalidRepositoryURLError("Repository URL must be a non-empty string.")

    try:
        parsed = urlparse(repository_url.strip())
        port = parsed.port
    except ValueError as exc:
        raise InvalidRepositoryURLError("Repository URL is malformed.") from exc

    if (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").lower() not in {"github.com", "www.github.com"}
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or parsed.params
    ):
        raise InvalidRepositoryURLError(
            "Use a public GitHub HTTPS URL such as "
            "https://github.com/apache/commons-lang.git."
        )

    path_parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(path_parts) != 2:
        raise InvalidRepositoryURLError(
            "GitHub repository URL must contain exactly an owner and repository name."
        )

    owner, name = path_parts
    if name.endswith(".git"):
        name = name[:-4]

    if (
        not _OWNER_PATTERN.fullmatch(owner)
        or not name
        or name in {".", ".."}
        or not _REPOSITORY_PATTERN.fullmatch(name)
    ):
        raise InvalidRepositoryURLError("GitHub owner or repository name is invalid.")

    return GitHubRepository(owner=owner, name=name)


def clone_repository(repository: GitHubRepository, destination: str | Path) -> Repo:
    """Clone a validated GitHub repository into ``destination``."""
    try:
        return Repo.clone_from(repository.clone_url, Path(destination))
    except (GitCommandError, OSError) as exc:
        raise CloneError(f"Could not clone {repository.slug}: {exc}") from exc


def mine_local_repository(repo: Repo, repository_name: str) -> pd.DataFrame:
    """Extract Java file changes from a local GitPython repository."""
    try:
        if not repo.head.is_valid():
            raise EmptyRepositoryError(f"Repository {repository_name} has no commits.")
        commits = repo.iter_commits("HEAD", reverse=True)
    except (GitCommandError, ValueError) as exc:
        raise EmptyRepositoryError(f"Repository {repository_name} has no commits.") from exc

    rows: list[dict[str, object]] = []
    commit_count = 0

    try:
        for commit in commits:
            commit_count += 1
            for file_path, stats in commit.stats.files.items():
                if not file_path.endswith(".java"):
                    continue

                rows.append(
                    {
                        "repository": repository_name,
                        "commit_hash": commit.hexsha,
                        "author_name": commit.author.name,
                        "author_email": commit.author.email,
                        "commit_date": commit.committed_datetime.isoformat(),
                        "commit_message": commit.message.strip(),
                        "file_path": file_path,
                        "lines_added": int(stats.get("insertions", 0)),
                        "lines_deleted": int(stats.get("deletions", 0)),
                    }
                )
    except (GitCommandError, OSError, ValueError) as exc:
        raise RepositoryMiningError(
            f"Could not read commit history for {repository_name}: {exc}"
        ) from exc

    if commit_count == 0:
        raise EmptyRepositoryError(f"Repository {repository_name} has no commits.")
    if not rows:
        raise NoJavaFilesError(
            f"Repository {repository_name} contains no Java file changes."
        )

    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def mine_repository(repository_url: str) -> pd.DataFrame:
    """Clone a public GitHub repository and return its Java history as a DataFrame."""
    repository = validate_github_url(repository_url)

    with tempfile.TemporaryDirectory(prefix="sentinel-") as temporary_directory:
        clone_path = Path(temporary_directory) / repository.name
        repo = clone_repository(repository, clone_path)
        try:
            return mine_local_repository(repo, repository.slug)
        finally:
            repo.close()


def export_to_csv(data: pd.DataFrame, output_path: str | Path) -> Path:
    """Export mined data to CSV and return the resolved output path."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False, columns=CSV_COLUMNS)
    return destination
