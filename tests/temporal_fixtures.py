"""Explicit synthetic coverage attestations; never used for real datasets."""

import pandas as pd
from sentinel.provenance import (
    PROVENANCE_KEY,
    build_feature_metadata,
    save_feature_metadata,
)


def attest(frame, observation_end="2030-12-31"):
    frame = frame.copy()
    history = pd.DataFrame(
        {
            "repository": sorted(frame.repository.unique()),
            "commit_date": observation_end,
        }
    )
    frame.attrs[PROVENANCE_KEY] = build_feature_metadata(history, frame)
    return frame


def write_fixture(frame, path):
    frame = attest(frame)
    frame.to_csv(path, index=False)
    save_feature_metadata(frame, path)
