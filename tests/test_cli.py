from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentinel import mine
from sentinel.miner import CSV_COLUMNS


def test_cli_writes_requested_csv(monkeypatch, tmp_path: Path, capsys) -> None:
    row = {column: "value" for column in CSV_COLUMNS}
    row.update(
        {
            "repository": "owner/repository",
            "lines_added": 2,
            "lines_deleted": 1,
        }
    )
    data = pd.DataFrame([row], columns=CSV_COLUMNS)
    monkeypatch.setattr(mine, "mine_repository", lambda url: data)
    output = tmp_path / "result.csv"

    exit_code = mine.main(
        ["https://github.com/owner/repository.git", "--output", str(output)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output.exists()
    assert "Mined 1 Java file changes" in captured.out
    assert captured.err == ""


def test_cli_reports_invalid_url(capsys) -> None:
    exit_code = mine.main(["https://example.com/owner/repository.git"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error:" in captured.err

