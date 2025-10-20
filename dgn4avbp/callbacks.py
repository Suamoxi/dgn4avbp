"""Custom Lightning callbacks used across the project."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only


@dataclass
class _TimingStats:
    """Utility container to accumulate timing statistics."""

    count_total: int = 0
    count_included: int = 0
    total: float = 0.0
    since_log: int = 0

    def update(self, duration: float, *, warmup: int) -> Tuple[bool, float]:
        """Update the accumulator.

        Args:
            duration: Measured duration in seconds.
            warmup:   Number of initial measurements to ignore.

        Returns:
            A tuple ``(should_log, average)`` where ``should_log`` indicates
            whether a new average is ready to be reported.
        """

        self.count_total += 1
        if self.count_total <= warmup:
            return False, 0.0

        self.count_included += 1
        self.total += duration
        self.since_log += 1

        if self.count_included == 0:
            return False, 0.0

        average = self.total / self.count_included
        return True, average


class StepTimeTracker(Callback):
    """Track the time spent in training/validation steps and dataloading.

    The callback measures two types of durations for every stage (`train`,
    `val`, `test`, `predict`):

    * ``<stage>/step_time`` – time spent inside the Lightning ``*_step``
      method.
    * ``<stage>/data_time`` – time between consecutive batches, a proxy for
      data loading time.

    The running averages are periodically logged via ``LightningModule.log``
    so they appear both in the progress bar and in any attached logger.
    A human-readable summary is also printed on rank 0 when the stage ends.
    """

    def __init__(
        self,
        *,
        warmup_batches: int = 5,
        log_every_n_steps: int = 25,
        log_to_prog_bar: bool = True,
    ) -> None:
        super().__init__()
        self.warmup_batches = max(0, warmup_batches)
        self.log_every_n_steps = max(1, log_every_n_steps)
        self.log_to_prog_bar = log_to_prog_bar

        self._batch_start_time: Dict[str, float] = {}
        self._last_batch_end: Dict[str, float] = {}
        self._stats: Dict[Tuple[str, str], _TimingStats] = defaultdict(_TimingStats)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _metric_name(self, stage: str, kind: str) -> str:
        return f"{stage}/{kind}_time"

    def _record(self, stage: str, kind: str, value: float, trainer, pl_module) -> None:
        stats = self._stats[(stage, kind)]
        should_log, avg = stats.update(value, warmup=self.warmup_batches)

        if not should_log:
            return

        # Log at the requested interval only.
        if stats.since_log < self.log_every_n_steps:
            return

        metric = self._metric_name(stage, kind)
        pl_module.log(
            metric,
            avg,
            on_step=True,
            on_epoch=False,
            prog_bar=self.log_to_prog_bar and stage == "train" and kind == "step",
            batch_size=1,
        )
        stats.since_log = 0

    def _summaries(self) -> Dict[str, float]:
        summary = {}
        for (stage, kind), stats in self._stats.items():
            if stats.count_included == 0:
                continue
            metric = self._metric_name(stage, kind)
            summary[metric] = stats.total / stats.count_included
        return summary

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------
    def setup(self, trainer, pl_module, stage: str) -> None:
        # Reset the timing state when entering a new stage (fit/validate/...)
        self._batch_start_time.clear()
        self._last_batch_end.clear()
        self._stats.clear()

    # Train ----------------------------------------------------------------
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self._batch_start("train", trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._batch_end("train", trainer, pl_module)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0) -> None:
        self._batch_start("val", trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self._batch_end("val", trainer, pl_module)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0) -> None:
        self._batch_start("test", trainer, pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self._batch_end("test", trainer, pl_module)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0) -> None:
        self._batch_start("predict", trainer, pl_module)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self._batch_end("predict", trainer, pl_module)

    def on_fit_end(self, trainer, pl_module) -> None:
        self._log_summary("fit")

    def on_validation_end(self, trainer, pl_module) -> None:
        self._log_summary("validate")

    def on_test_end(self, trainer, pl_module) -> None:
        self._log_summary("test")

    def on_predict_end(self, trainer, pl_module) -> None:
        self._log_summary("predict")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _batch_start(self, stage: str, trainer, pl_module) -> None:
        now = time.perf_counter()
        prev_end = self._last_batch_end.get(stage)
        if prev_end is not None:
            self._record(stage, "data", now - prev_end, trainer, pl_module)
        self._batch_start_time[stage] = now

    def _batch_end(self, stage: str, trainer, pl_module) -> None:
        now = time.perf_counter()
        start = self._batch_start_time.get(stage)
        if start is not None:
            self._record(stage, "step", now - start, trainer, pl_module)
        self._last_batch_end[stage] = now

    @rank_zero_only
    def _log_summary(self, stage_name: str) -> None:
        summary = self._summaries()
        if not summary:
            return
        print("\n--- StepTimeTracker summary (%s) ---" % stage_name)
        for name, value in sorted(summary.items()):
            print(f"  {name}: {value:.4f} s")

