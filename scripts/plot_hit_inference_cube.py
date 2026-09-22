#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors

from dgn4avbp.hit_benchmark import infer_cartesian_grid_3d


STATE_CHANNELS = {
    "rho": 0,
    "rhou": 1,
    "rhov": 2,
    "rhow": 3,
    "rhoE": 4,
}

SIGNED_FIELDS = {
    "rhou",
    "rhov",
    "rhow",
    "u",
    "v",
    "w",
    "u_prime",
    "v_prime",
    "w_prime",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render DGN HIT inference fields as a three-face scientific cube. "
            "The input is a sample_*.pt artifact written by generate_hit_dgn.py."
        )
    )
    parser.add_argument("input", help="Generated sample_*.pt file or directory containing sample_*.pt files.")
    parser.add_argument(
        "--field",
        default="u_prime",
        choices=(
            "rho",
            "rhou",
            "rhov",
            "rhow",
            "rhoE",
            "u",
            "v",
            "w",
            "u_prime",
            "v_prime",
            "w_prime",
            "speed",
            "kinetic_energy",
        ),
        help="Scalar field rendered on the cube faces.",
    )
    parser.add_argument(
        "--state-space",
        default="nondimensional",
        choices=("nondimensional", "dimensional"),
        help="Stored state representation used to derive the requested field.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults next to the input.")
    parser.add_argument("--max-samples", type=int, default=None, help="When input is a directory, render at most this many samples.")
    parser.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap.")
    parser.add_argument(
        "--robust-percentile",
        type=float,
        default=99.0,
        help="Symmetric signed range uses this percentile of |field|; unsigned range uses complementary percentiles.",
    )
    parser.add_argument("--vmin", type=float, default=None, help="Manual color minimum.")
    parser.add_argument("--vmax", type=float, default=None, help="Manual color maximum.")
    parser.add_argument("--elev", type=float, default=24.0, help="Camera elevation in degrees.")
    parser.add_argument("--azim", type=float, default=-52.0, help="Camera azimuth in degrees.")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--size", type=float, default=6.0, help="Square figure size in inches.")
    parser.add_argument("--colorbar", action="store_true", help="Add a compact colorbar.")
    parser.add_argument("--title", action="store_true", help="Add field/sample title.")
    parser.add_argument("--transparent", action="store_true", help="Save with transparent background.")
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def state_and_position(payload: dict, state_space: str) -> tuple[np.ndarray, np.ndarray]:
    state_key = f"state_{state_space}"
    pos_key = f"pos_{state_space}"
    state = payload.get(state_key)
    pos = payload.get(pos_key)
    if not isinstance(state, torch.Tensor):
        raise ValueError(f"Missing tensor '{state_key}' in generation artifact.")
    if not isinstance(pos, torch.Tensor):
        raise ValueError(f"Missing tensor '{pos_key}' in generation artifact.")
    if state.ndim != 2 or state.shape[1] != 5:
        raise ValueError(f"{state_key} must have shape [N,5], got {tuple(state.shape)}.")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"{pos_key} must have shape [N,3], got {tuple(pos.shape)}.")
    if state.shape[0] != pos.shape[0]:
        raise ValueError("State and position node counts do not match.")
    state_np = state.detach().cpu().numpy().astype(np.float64, copy=False)
    pos_np = pos.detach().cpu().numpy().astype(np.float64, copy=False)
    if not np.isfinite(state_np).all() or not np.isfinite(pos_np).all():
        raise ValueError("Generation artifact contains non-finite state or position values.")
    return state_np, pos_np


def scalar_field(state: np.ndarray, field: str, rho_floor: float = 1.0e-12) -> np.ndarray:
    if field in STATE_CHANNELS:
        return state[:, STATE_CHANNELS[field]]

    rho = state[:, 0]
    if np.any(np.abs(rho) <= rho_floor):
        raise ValueError("Density too close to zero to derive velocity safely.")
    velocity = state[:, 1:4] / rho[:, None]

    if field == "u":
        return velocity[:, 0]
    if field == "v":
        return velocity[:, 1]
    if field == "w":
        return velocity[:, 2]

    centered = velocity - np.mean(velocity, axis=0, keepdims=True)
    if field == "u_prime":
        return centered[:, 0]
    if field == "v_prime":
        return centered[:, 1]
    if field == "w_prime":
        return centered[:, 2]
    if field == "speed":
        return np.linalg.norm(velocity, axis=1)
    if field == "kinetic_energy":
        return 0.5 * np.sum(centered**2, axis=1)

    raise ValueError(f"Unsupported field: {field}")


def structured_scalar(pos: np.ndarray, values: np.ndarray):
    grid = infer_cartesian_grid_3d(pos)
    if values.shape != (pos.shape[0],):
        raise ValueError(f"Expected scalar values [N], got {values.shape}.")
    flat = np.empty(int(np.prod(grid.shape)), dtype=np.float64)
    flat[grid.node_to_flat] = values
    scalar = flat.reshape(grid.shape)
    return grid, scalar


def make_norm(
    values: np.ndarray,
    field: str,
    percentile: float,
    manual_vmin: float | None,
    manual_vmax: float | None,
):
    if not 50.0 < percentile <= 100.0:
        raise ValueError("--robust-percentile must be in (50,100].")

    if manual_vmin is not None or manual_vmax is not None:
        if manual_vmin is None or manual_vmax is None:
            raise ValueError("--vmin and --vmax must be supplied together.")
        if manual_vmax <= manual_vmin:
            raise ValueError("--vmax must be greater than --vmin.")
        if field in SIGNED_FIELDS and manual_vmin < 0.0 < manual_vmax:
            return colors.TwoSlopeNorm(vmin=manual_vmin, vcenter=0.0, vmax=manual_vmax)
        return colors.Normalize(vmin=manual_vmin, vmax=manual_vmax)

    if field in SIGNED_FIELDS:
        limit = float(np.percentile(np.abs(values), percentile))
        if limit <= 0.0:
            limit = float(np.max(np.abs(values)))
        if limit <= 0.0:
            limit = 1.0
        return colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    tail = (100.0 - percentile) / 2.0
    vmin = float(np.percentile(values, tail))
    vmax = float(np.percentile(values, 100.0 - tail))
    if vmax <= vmin:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return colors.Normalize(vmin=vmin, vmax=vmax)


def surface_facecolors(cmap, norm, values: np.ndarray) -> np.ndarray:
    return cmap(norm(values))


def render_cube(
    *,
    pos: np.ndarray,
    values: np.ndarray,
    field: str,
    output: Path,
    cmap_name: str,
    percentile: float,
    vmin: float | None,
    vmax: float | None,
    elev: float,
    azim: float,
    dpi: int,
    size: float,
    colorbar: bool,
    title: str | None,
    transparent: bool,
) -> None:
    grid, scalar = structured_scalar(pos, values)
    x, y, z = grid.axes

    cmap = plt.get_cmap(cmap_name)
    norm = make_norm(values, field, percentile, vmin, vmax)

    fig = plt.figure(figsize=(size, size), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Top face z=max.
    xx, yy = np.meshgrid(x, y, indexing="ij")
    zz = np.full_like(xx, z[-1], dtype=np.float64)
    top = scalar[:, :, -1]
    ax.plot_surface(
        xx,
        yy,
        zz,
        facecolors=surface_facecolors(cmap, norm, top),
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=False,
        linewidth=0.0,
    )

    # x=max face.
    yy_x, zz_x = np.meshgrid(y, z, indexing="ij")
    xx_x = np.full_like(yy_x, x[-1], dtype=np.float64)
    xface = scalar[-1, :, :]
    ax.plot_surface(
        xx_x,
        yy_x,
        zz_x,
        facecolors=surface_facecolors(cmap, norm, xface),
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=False,
        linewidth=0.0,
    )

    # y=min face.
    xx_y, zz_y = np.meshgrid(x, z, indexing="ij")
    yy_y = np.full_like(xx_y, y[0], dtype=np.float64)
    yface = scalar[:, 0, :]
    ax.plot_surface(
        xx_y,
        yy_y,
        zz_y,
        facecolors=surface_facecolors(cmap, norm, yface),
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=False,
        linewidth=0.0,
    )

    # Thin cube edges improve the silhouette without showing mesh lines.
    edge_color = (0.35, 0.35, 0.35, 0.65)
    corners = [
        (x[0], y[0], z[0]),
        (x[-1], y[0], z[0]),
        (x[0], y[-1], z[0]),
        (x[-1], y[-1], z[0]),
        (x[0], y[0], z[-1]),
        (x[-1], y[0], z[-1]),
        (x[0], y[-1], z[-1]),
        (x[-1], y[-1], z[-1]),
    ]
    edge_pairs = (
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for i, j in edge_pairs:
        p0, p1 = corners[i], corners[j]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=edge_color,
            linewidth=0.45,
        )

    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(float(y[0]), float(y[-1]))
    ax.set_zlim(float(z[0]), float(z[-1]))
    ax.set_box_aspect((float(np.ptp(x)), float(np.ptp(y)), float(np.ptp(z))))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0 if transparent else 1.0))
    fig.patch.set_alpha(0.0 if transparent else 1.0)

    if title is not None:
        ax.set_title(title, pad=4)

    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01, shrink=0.72)
        cb.ax.tick_params(labelsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=transparent,
    )
    plt.close(fig)


def input_files(path: Path, max_samples: int | None) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(path.glob("sample_*.pt"))
    if not files:
        raise FileNotFoundError(f"No sample_*.pt files found in {path}.")
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        files = files[:max_samples]
    return files


def main() -> None:
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    files = input_files(source, args.max_samples)

    if args.output_dir is None:
        base = source.parent if source.is_file() else source
        output_dir = base / "cube_plots"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()

    for path in files:
        payload = load_payload(path)
        state, pos = state_and_position(payload, args.state_space)
        values = scalar_field(state, args.field)

        sample_index = payload.get("sample_index")
        suffix = f"{int(sample_index):05d}" if sample_index is not None else path.stem
        output = output_dir / f"{args.field}_{suffix}.png"
        plot_title = f"{args.field} | {path.stem}" if args.title else None

        render_cube(
            pos=pos,
            values=values,
            field=args.field,
            output=output,
            cmap_name=args.cmap,
            percentile=args.robust_percentile,
            vmin=args.vmin,
            vmax=args.vmax,
            elev=args.elev,
            azim=args.azim,
            dpi=args.dpi,
            size=args.size,
            colorbar=args.colorbar,
            title=plot_title,
            transparent=args.transparent,
        )
        print(output)


if __name__ == "__main__":
    main()
