from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from matplotlib.colors import LinearSegmentedColormap
import numpy as np

__all__ = ["PaletteStop", "PalColorTable", "load_pal_table"]

_KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.*?)\s*$")


@dataclass(frozen=True)
class PaletteStop:
    value: float
    rgb: tuple[float, float, float]


@dataclass(frozen=True)
class PalColorTable:
    path: str
    product: str | None
    units: str | None
    step: float | None
    scale: float
    nd: tuple[float, float, float] | None
    rf: tuple[float, float, float] | None
    stops: tuple[PaletteStop, ...]

    @property
    def name(self) -> str:
        return Path(self.path).name

    def value_limits(self, *, internal_units: bool = True) -> tuple[float, float]:
        values = np.asarray([stop.value for stop in self.stops], dtype=float)
        if internal_units and self.scale not in (0.0, 1.0):
            values = values / self.scale
        return float(values.min()), float(values.max())

    def matplotlib_colormap(self, *, name: str | None = None, internal_units: bool = True) -> tuple[LinearSegmentedColormap, float, float]:
        if not self.stops:
            raise ValueError(f"palette {self.path!r} has no color stops")

        values = np.asarray([stop.value for stop in self.stops], dtype=float)
        colors = np.asarray([stop.rgb for stop in self.stops], dtype=float) / 255.0
        if internal_units and self.scale not in (0.0, 1.0):
            values = values / self.scale

        order = np.argsort(values)
        values = values[order]
        colors = colors[order]
        vmin = float(values[0])
        vmax = float(values[-1])
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        positions = (values - vmin) / (vmax - vmin)
        cmap_name = name or Path(self.path).stem
        cmap = LinearSegmentedColormap.from_list(cmap_name, list(zip(positions, colors)), N=512)
        return cmap, vmin, vmax


def _strip_comment(line: str) -> str:
    return line.split(";", 1)[0].strip()


def _parse_rgb(tokens: list[str]) -> tuple[float, float, float] | None:
    if len(tokens) < 3:
        return None
    try:
        return float(tokens[0]), float(tokens[1]), float(tokens[2])
    except ValueError:
        return None


def load_pal_table(path: str | Path) -> PalColorTable:
    palette_path = Path(path)
    product: str | None = None
    units: str | None = None
    step: float | None = None
    scale = 1.0
    nd: tuple[float, float, float] | None = None
    rf: tuple[float, float, float] | None = None
    stops: list[PaletteStop] = []

    for raw_line in palette_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue

        match = _KEY_VALUE_RE.match(line)
        if not match:
            continue
        key = match.group(1).lower()
        payload = match.group(2).strip()
        tokens = payload.split()

        if key == "product":
            product = payload.upper()
            continue
        if key == "units":
            units = payload
            continue
        if key == "step":
            try:
                step = float(tokens[0])
            except (IndexError, ValueError):
                step = None
            continue
        if key == "scale":
            try:
                scale = float(tokens[0])
            except (IndexError, ValueError):
                scale = 1.0
            continue
        if key == "nd":
            nd = _parse_rgb(tokens)
            continue
        if key == "rf":
            rf = _parse_rgb(tokens)
            continue
        if key in {"color", "solidcolor", "color4"}:
            if len(tokens) < 4:
                continue
            try:
                value = float(tokens[0])
            except ValueError:
                continue
            rgb = _parse_rgb(tokens[1:])
            if rgb is None:
                continue
            stops.append(PaletteStop(value=value, rgb=rgb))

    if not stops:
        raise ValueError(f"no palette stops found in {palette_path}")

    return PalColorTable(
        path=str(palette_path),
        product=product,
        units=units,
        step=step,
        scale=scale,
        nd=nd,
        rf=rf,
        stops=tuple(stops),
    )
