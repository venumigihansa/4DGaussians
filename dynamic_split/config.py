from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PriorConfig:
    mad_multiplier: float = 5.0
    flow_consistency_threshold: float = 1.5
    depth_confidence_quantile: float = 0.10
    irls_iterations: int = 30
    min_component_area: int = 256
    sam_box_padding_ratio: float = 0.05
    sam_min_component_coverage: float = 0.50
    refine_pose_flow: bool = True

    def validate(self) -> "PriorConfig":
        if self.mad_multiplier < 0:
            raise ValueError("mad_multiplier must be non-negative")
        if self.flow_consistency_threshold < 0:
            raise ValueError("flow_consistency_threshold must be non-negative")
        if not 0 <= self.depth_confidence_quantile < 1:
            raise ValueError("depth_confidence_quantile must be in [0, 1)")
        if self.irls_iterations < 1:
            raise ValueError("irls_iterations must be positive")
        if self.min_component_area < 1:
            raise ValueError("min_component_area must be positive")
        if self.sam_box_padding_ratio < 0:
            raise ValueError("sam_box_padding_ratio must be non-negative")
        if not 0 <= self.sam_min_component_coverage <= 1:
            raise ValueError("sam_min_component_coverage must be in [0, 1]")
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
    support_weight: float = 0.1
    support_temperature: float = 1.0
    seed: int = 6666
    checkpoint_path: str | None = None
    source_path: str | None = None

    def validate(self, require_priors: bool = True) -> "DynamicSplitConfig":
        if self.iterations < 1:
            raise ValueError("dynamic iterations must be positive")
        if self.lr_init <= 0 or self.lr_final <= 0:
            raise ValueError("dynamic learning rates must be positive")
        if self.support_weight < 0:
            raise ValueError("dynamic support weight must be non-negative")
        if self.support_temperature <= 0:
            raise ValueError("dynamic support temperature must be positive")
        if require_priors and not self.prior_dir.is_dir():
            raise FileNotFoundError(f"Dynamic prior directory not found: {self.prior_dir}")
        return self

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["prior_dir"] = str(self.prior_dir)
        result["output_dir"] = str(self.output_dir)
        return result
