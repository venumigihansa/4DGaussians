from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PriorConfig:
    residual_threshold: float = 1.0
    flow_consistency_threshold: float = 1.5
    min_depth_confidence: float = 0.0
    sam_support_ratio: float = 0.05
    sam_support_pixels: int = 64
    min_component_area: int = 64

    def validate(self) -> "PriorConfig":
        if self.residual_threshold < 0:
            raise ValueError("residual_threshold must be non-negative")
        if self.flow_consistency_threshold < 0:
            raise ValueError("flow_consistency_threshold must be non-negative")
        if not 0 <= self.sam_support_ratio <= 1:
            raise ValueError("sam_support_ratio must be in [0, 1]")
        if self.sam_support_pixels < 1 or self.min_component_area < 1:
            raise ValueError("pixel and component thresholds must be positive")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicSplitConfig:
    prior_dir: Path
    output_dir: Path
    iterations: int = 3_000
    lr_init: float = 0.05
    lr_final: float = 0.005
    threshold: float = 7.0
    seed: int = 6666
    checkpoint_path: str | None = None
    source_path: str | None = None

    def validate(self, require_priors: bool = True) -> "DynamicSplitConfig":
        if self.iterations < 1:
            raise ValueError("dynamic iterations must be positive")
        if self.lr_init <= 0 or self.lr_final <= 0:
            raise ValueError("dynamic learning rates must be positive")
        if require_priors and not self.prior_dir.is_dir():
            raise FileNotFoundError(f"Dynamic prior directory not found: {self.prior_dir}")
        return self

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["prior_dir"] = str(self.prior_dir)
        result["output_dir"] = str(self.output_dir)
        return result
