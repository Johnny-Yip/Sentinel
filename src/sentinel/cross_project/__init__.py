"""Leave-one-project-out evaluation for Sentinel V4."""

from sentinel.cross_project.experiment import (
    CrossProjectResult,
    run_cross_project_evaluation,
)

__all__ = ["CrossProjectResult", "run_cross_project_evaluation"]
