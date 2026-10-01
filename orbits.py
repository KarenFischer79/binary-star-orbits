#!/usr/bin/env python3
"""Plot WDS historical measurements and a published visual-binary orbit.

This first version reads the edited USNO data-request text files used by the
Orbits project.  Each file contains a MEASURES section followed by the
published ORBITAL ELEMENTS.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import json
import hashlib
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Observation:
    epoch: float
    theta_deg: float
    rho_arcsec: float
    aperture_m: float | None = None
    reference: str = ""
    technique: str = ""
    theta_flag: str = ""
    rho_flag: str = ""


@dataclass(frozen=True)
class OrbitElements:
    designation: str
    period_years: float
    semimajor_arcsec: float
    inclination_deg: float
    node_deg: float
    periastron_epoch: float
    eccentricity: float
    omega_deg: float
    grade: str = ""
    reference: str = ""
    uncertainties: tuple[float | None, ...] = (None,) * 7


@dataclass(frozen=True)
class WDSData:
    wds_id: str
    observations: tuple[Observation, ...]
    orbit: OrbitElements
    incomplete_measurement_lines: int = 0


def _number(text: str) -> float | None:
    """Return a float from a WDS numeric field, or None for a missing value."""
    value = text.strip().replace(":", "")
    if not value or value == ".":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _period_in_years(period: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "y":
        return period
    if unit == "d":
        return period / 365.25
    if unit == "m":
        return period / 12.0
    if unit == "c":
        return period * 100.0
    raise ValueError(f"Unsupported orbital-period unit: {unit!r}")


def _axis_in_arcsec(axis: float, unit: str) -> float:
    # ORB6 uses "a" for arcseconds and "m" for milliarcseconds.
    unit = unit.lower()
    if unit == "a":
        return axis
    if unit == "m":
        return axis / 1000.0
    raise ValueError(f"Unsupported semimajor-axis unit: {unit!r}")


def _parse_orbit_line(line: str) -> OrbitElements:
    tokens = line.split()
    first_number = next(
        (index for index, token in enumerate(tokens) if _number(token) is not None),
        None,
    )
    if first_number is None or len(tokens) < first_number + 12:
        raise ValueError(f"Could not interpret orbital-elements line: {line}")

    designation = " ".join(tokens[:first_number])
    values = tokens[first_number:]
    try:
        period = _period_in_years(float(values[0]), values[1])
        axis = _axis_in_arcsec(float(values[2]), values[3])
        inclination = float(values[4])
        node = float(values[5])
        periastron = float(values[6])
        # values[7] is the epoch unit. The current project files use years.
        if values[7].lower() != "y":
            raise ValueError("Periastron epochs other than years are not yet supported")
        eccentricity = float(values[8])
        omega = float(values[9])
        grade = values[10]
        reference = values[11]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not interpret orbital-elements line: {line}") from exc

    return OrbitElements(
        designation=designation,
        period_years=period,
        semimajor_arcsec=axis,
        inclination_deg=inclination,
        node_deg=node,
        periastron_epoch=periastron,
        eccentricity=eccentricity,
        omega_deg=omega,
        grade=grade,
        reference=reference,
    )


def read_wds_file(path: str | Path) -> WDSData:
    """Read one edited USNO WDS data-request file."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()

    wds_match = re.search(r"WDS\s+([0-9]{5}[+-][0-9]{4})", "\n".join(lines[:8]))
    wds_id = wds_match.group(1) if wds_match else path.stem

    try:
        measures_start = next(i for i, line in enumerate(lines) if line.strip() == "MEASURES:")
        orbit_start = next(i for i, line in enumerate(lines) if line.strip() == "ORBITAL ELEMENTS:")
    except StopIteration as exc:
        raise ValueError(
            "The file must contain MEASURES: and ORBITAL ELEMENTS: sections."
        ) from exc

    observations: list[Observation] = []
    incomplete = 0
    for line in lines[measures_start + 1 : orbit_start]:
        if len(line) < 35:
            continue
        # A standalone WDS measurement record has 14-character identifier and
        # 7-character component fields.  The human-readable data-request file
        # omits those blank fields, shifting the remaining columns 17 places
        # to the left.  Project files use this condensed 120-character form.
        shift = -17 if len(line) < 130 else 0
        epoch = _number(line[24 + shift : 34 + shift])
        if epoch is None:
            continue

        theta = _number(line[36 + shift : 43 + shift])
        rho = _number(line[52 + shift : 61 + shift])
        if theta is None or rho is None:
            incomplete += 1
            continue

        aperture = _number(line[109 + shift : 114 + shift])
        reference = line[119 + shift : 127 + shift].strip()
        technique = line[128 + shift : 130 + shift].strip()
        observations.append(
            Observation(
                epoch=epoch,
                theta_deg=theta % 360.0,
                rho_arcsec=rho,
                aperture_m=aperture,
                reference=reference,
                technique=technique,
                theta_flag=line[35 + shift : 36 + shift].strip(),
                rho_flag=line[51 + shift : 52 + shift].strip(),
            )
        )

    if not observations:
        raise ValueError("No complete epoch/theta/rho measurements were found.")

    orbit_line_index = None
    orbit = None
    for i in range(orbit_start + 1, min(len(lines), orbit_start + 12)):
        line = lines[i]
        if not line.strip() or "---" in line or "Reference" in line:
            continue
        try:
            orbit = _parse_orbit_line(line)
            orbit_line_index = i
            break
        except ValueError:
            continue
    if orbit is None or orbit_line_index is None:
        raise ValueError("No published orbital-elements record was found.")

    uncertainties: tuple[float | None, ...] = (None,) * 7
    for line in lines[orbit_line_index + 1 : orbit_line_index + 4]:
        if "+/-" in line:
            numbers = [float(value) for value in re.findall(r"(?<![A-Za-z])\d*\.\d+|(?<![A-Za-z])\d+", line)]
            if len(numbers) >= 7:
                uncertainties = tuple(numbers[:7])
            break
    orbit = OrbitElements(**{**orbit.__dict__, "uncertainties": uncertainties})

    return WDSData(
        wds_id=wds_id,
        observations=tuple(observations),
        orbit=orbit,
        incomplete_measurement_lines=incomplete,
    )


def read_additional_csv(path: str | Path) -> tuple[Observation, ...]:
    """Read optional new observations from the project's small CSV format."""
    result: list[Observation] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            normalized = {re.sub(r"[^a-z]", "", key.lower()): value for key, value in row.items() if key}
            result.append(
                Observation(
                    epoch=float(normalized["date"]),
                    theta_deg=float(normalized.get("thetadeg", normalized.get("theta", "nan"))),
                    rho_arcsec=float(normalized.get("rhoarcsec", normalized.get("rho", "nan"))),
                    aperture_m=_number(normalized.get("aperturemeters", "")),
                    reference=normalized.get("reference", ""),
                    technique=normalized.get("techcode", ""),
                )
            )
    return tuple(result)


def _reported_aperture(observation: Observation) -> str:
    """Format a meaningful ground-based aperture for CSV output.

    WDS technique codes beginning with H are space-based measurements (Gaia,
    Hipparcos, Tycho, HST, and related instruments).  The WDS aperture field
    is therefore not reported as a ground-based telescope aperture for them.
    """
    if observation.aperture_m is None or observation.technique.upper().startswith("H"):
        return ""
    return f"{observation.aperture_m:g}"


def _wrapped_angle_difference(observed_deg: np.ndarray, calculated_deg: np.ndarray) -> np.ndarray:
    """Return observed-minus-calculated angles in the interval [-180, 180)."""
    return (observed_deg - calculated_deg + 180.0) % 360.0 - 180.0


def write_residuals_csv(
    data: WDSData,
    output: str | Path,
    additional: Iterable[Observation] = (),
    weights: dict[str, float] | None = None,
) -> Path:
    """Write complete theta-rho pairs and published-orbit residuals for Excel."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    historical = list(data.observations)
    extra = list(additional)
    observations = historical + extra
    if weights is not None:
        observation_weights(observations, weights)

    epochs = np.array([item.epoch for item in observations], dtype=float)
    observed_theta = np.array([item.theta_deg for item in observations], dtype=float)
    observed_rho = np.array([item.rho_arcsec for item in observations], dtype=float)
    observed_theta_rad = np.deg2rad(observed_theta)
    observed_east = observed_rho * np.sin(observed_theta_rad)
    observed_north = observed_rho * np.cos(observed_theta_rad)

    calculated_east, calculated_north, calculated_theta, calculated_rho = apparent_position(
        epochs, data.orbit
    )
    theta_residual = _wrapped_angle_difference(observed_theta, calculated_theta)
    rho_residual = observed_rho - calculated_rho
    east_residual = observed_east - calculated_east
    north_residual = observed_north - calculated_north
    total_residual = np.hypot(east_residual, north_residual)

    columns = [
        "wds_id",
        "designation",
        "source",
        "epoch",
        "technique_code",
        "aperture_m",
        "reference",
        "theta_flag",
        "rho_flag",
        "observed_theta_deg",
        "observed_rho_arcsec",
        "calculated_theta_deg",
        "calculated_rho_arcsec",
        "theta_residual_deg_O_minus_C",
        "rho_residual_arcsec_O_minus_C",
        "east_residual_arcsec_O_minus_C",
        "north_residual_arcsec_O_minus_C",
        "total_residual_arcsec",
        "total_residual_mas",
    ]
    if weights is not None:
        columns.append("technique_weight")
    # utf-8-sig writes a BOM that helps Excel recognize the CSV encoding.
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, observation in enumerate(observations):
            writer.writerow(
                {
                    "wds_id": data.wds_id,
                    "designation": data.orbit.designation,
                    "source": "historical" if index < len(historical) else "additional",
                    "epoch": f"{observation.epoch:.5f}",
                    "technique_code": observation.technique,
                    "aperture_m": _reported_aperture(observation),
                    "reference": observation.reference,
                    "theta_flag": observation.theta_flag,
                    "rho_flag": observation.rho_flag,
                    "observed_theta_deg": f"{observation.theta_deg:.5f}",
                    "observed_rho_arcsec": f"{observation.rho_arcsec:.5f}",
                    "calculated_theta_deg": f"{calculated_theta[index]:.5f}",
                    "calculated_rho_arcsec": f"{calculated_rho[index]:.6f}",
                    "theta_residual_deg_O_minus_C": f"{theta_residual[index]:.5f}",
                    "rho_residual_arcsec_O_minus_C": f"{rho_residual[index]:.6f}",
                    "east_residual_arcsec_O_minus_C": f"{east_residual[index]:.6f}",
                    "north_residual_arcsec_O_minus_C": f"{north_residual[index]:.6f}",
                    "total_residual_arcsec": f"{total_residual[index]:.6f}",
                    "total_residual_mas": f"{1000.0 * total_residual[index]:.3f}",
                    **({"technique_weight": weights[observation.technique]} if weights is not None else {}),
                }
            )
    return output


def solve_kepler(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    """Solve E - e sin(E) = M with safeguarded Newton iteration."""
    mean_anomaly = np.mod(np.asarray(mean_anomaly, dtype=float), 2.0 * np.pi)
    eccentric_anomaly = np.where(eccentricity < 0.8, mean_anomaly, np.pi)
    lower = np.zeros_like(mean_anomaly)
    upper = np.full_like(mean_anomaly, 2 * np.pi)
    for iteration in range(100):
        error = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly
        correction = error / (1.0 - eccentricity * np.cos(eccentric_anomaly))
        # Near e=1, cancellation can keep the Newton correction above tolerance
        # even when the bracket has reached floating-point resolution.
        converged = (np.abs(correction) < 1e-13) | ((upper - lower) < 2e-13)
        if np.all(converged):
            return eccentric_anomaly
        lower = np.where(error < 0, eccentric_anomaly, lower)
        upper = np.where(error > 0, eccentric_anomaly, upper)
        candidate = eccentric_anomaly - correction
        # Fall back to guaranteed bracket contraction if Newton stalls.
        candidate = np.where((candidate < lower) | (candidate > upper) | (iteration >= 12), (lower + upper) / 2, candidate)
        eccentric_anomaly = np.where(converged, eccentric_anomaly, candidate)
    raise ValueError("Kepler equation did not converge.")


def apparent_position(
    epochs: Iterable[float] | float, orbit: OrbitElements
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return east, north, theta, and rho for one or more Julian epochs."""
    epochs_array = np.atleast_1d(np.asarray(epochs, dtype=float))
    mean_anomaly = 2.0 * np.pi * (
        epochs_array - orbit.periastron_epoch
    ) / orbit.period_years
    eccentric_anomaly = solve_kepler(mean_anomaly, orbit.eccentricity)

    ellipse_x = np.cos(eccentric_anomaly) - orbit.eccentricity
    ellipse_y = math.sqrt(1.0 - orbit.eccentricity**2) * np.sin(eccentric_anomaly)

    inclination, node, omega = np.deg2rad(
        [orbit.inclination_deg, orbit.node_deg, orbit.omega_deg]
    )
    cos_node, sin_node = np.cos(node), np.sin(node)
    cos_omega, sin_omega = np.cos(omega), np.sin(omega)
    cos_i = np.cos(inclination)
    axis = orbit.semimajor_arcsec

    # Thiele-Innes constants. North is rho*cos(theta), east is rho*sin(theta).
    A = axis * (cos_node * cos_omega - sin_node * sin_omega * cos_i)
    B = axis * (sin_node * cos_omega + cos_node * sin_omega * cos_i)
    F = axis * (-cos_node * sin_omega - sin_node * cos_omega * cos_i)
    G = axis * (-sin_node * sin_omega + cos_node * cos_omega * cos_i)

    north = A * ellipse_x + F * ellipse_y
    east = B * ellipse_x + G * ellipse_y
    rho = np.hypot(east, north)
    theta = np.mod(np.degrees(np.arctan2(east, north)), 360.0)
    return east, north, theta, rho


def make_plot(
    data: WDSData,
    output: str | Path,
    additional: Iterable[Observation] = (),
    show: bool = False,
    published: OrbitElements | None = None,
    weights: dict[str, float] | None = None,
    filter_mas: float | None = None,
) -> Path:
    """Create the published-orbit plot and parameter display."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    observations = data.observations
    extra = tuple(additional)

    epoch = np.array([item.epoch for item in observations])
    theta = np.deg2rad([item.theta_deg for item in observations])
    rho = np.array([item.rho_arcsec for item in observations])
    obs_east = rho * np.sin(theta)
    obs_north = rho * np.cos(theta)

    curve_epochs = np.linspace(
        data.orbit.periastron_epoch,
        data.orbit.periastron_epoch + data.orbit.period_years,
        1400,
    )
    curve_east, curve_north, _, _ = apparent_position(curve_epochs, data.orbit)
    peri_east, peri_north, _, _ = apparent_position(
        [data.orbit.periastron_epoch], data.orbit
    )

    fig = plt.figure(figsize=(12.6, 7.5), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.55, 1.0))
    ax = fig.add_subplot(grid[0, 0])
    info = fig.add_subplot(grid[0, 1])

    ax.plot(curve_east, curve_north, color="#1f4e79", linewidth=2.2, label="Optimized orbit" if published else "Published orbit")
    if published is not None:
        pe, pn, _, _ = apparent_position(np.linspace(published.periastron_epoch, published.periastron_epoch + published.period_years, 1400), published)
        ax.plot(pe, pn, color="#b65b35", linestyle="--", linewidth=1.5, label="Published orbit")
    points = ax.scatter(
        obs_east,
        obs_north,
        c=epoch,
        cmap="viridis",
        s=24,
        alpha=0.82,
        linewidths=0.25,
        edgecolors="white",
        label="Historical measurements",
        zorder=3,
    )
    ax.scatter([0], [0], marker="*", s=180, color="#f2b134", edgecolor="black", linewidth=0.6, label="Primary", zorder=5)
    ax.scatter(peri_east, peri_north, marker="D", s=46, color="#c53b32", label="Periastron", zorder=5)

    if extra:
        extra_theta = np.deg2rad([item.theta_deg for item in extra])
        extra_rho = np.array([item.rho_arcsec for item in extra])
        ax.scatter(
            extra_rho * np.sin(extra_theta),
            extra_rho * np.cos(extra_theta),
            marker="X",
            s=95,
            color="#e05263",
            edgecolor="black",
            linewidth=0.6,
            label="Additional observation",
            zorder=6,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_xaxis()  # Standard double-star convention: east is to the left.
    ax.axhline(0, color="0.82", linewidth=0.7, zorder=0)
    ax.axvline(0, color="0.82", linewidth=0.7, zorder=0)
    ax.grid(color="0.9", linewidth=0.7)
    ax.set_xlabel("East–west offset (arcsec; east is left)")
    ax.set_ylabel("North–south offset (arcsec)")
    ax.set_title(f"WDS {data.wds_id} — {data.orbit.designation}", fontsize=15, weight="bold")
    ax.legend(loc="best", fontsize=9)
    colorbar = fig.colorbar(points, ax=ax, shrink=0.82, pad=0.03)
    colorbar.set_label("Observation epoch")

    info.axis("off")
    info.text(0.0, 0.98, "Optimized orbital elements" if published else "Published orbital elements", fontsize=15, weight="bold", va="top")
    uncertainty = data.orbit.uncertainties
    rows = [
        ("Period, P", data.orbit.period_years, uncertainty[0], "yr"),
        ("Semimajor axis, a", data.orbit.semimajor_arcsec, uncertainty[1], "arcsec"),
        ("Inclination, i", data.orbit.inclination_deg, uncertainty[2], "deg"),
        ("Node, Ω", data.orbit.node_deg, uncertainty[3], "deg"),
        ("Periastron epoch, T", data.orbit.periastron_epoch, uncertainty[4], "yr"),
        ("Eccentricity, e", data.orbit.eccentricity, uncertainty[5], ""),
        ("Argument, ω", data.orbit.omega_deg, uncertainty[6], "deg"),
    ]
    y = 0.88
    for label, value, error, unit in rows:
        value_text = f"{value:.5f}"
        if error is not None:
            value_text += f" ± {error:g}"
        if unit:
            value_text += f" {unit}"
        info.text(0.02, y, label, fontsize=10.5, color="0.25")
        info.text(0.98, y, value_text, fontsize=10.5, ha="right")
        y -= 0.067

    info.axhline(y + 0.025, xmin=0.02, xmax=0.98, color="0.8", linewidth=0.8)
    y -= 0.018
    facts = [
        ("Orbit grade", data.orbit.grade),
        ("Orbit reference", data.orbit.reference),
        ("Retained historical pairs" if filter_mas is not None else "Complete historical pairs", f"{len(observations):,}"),
        ("Incomplete records skipped", f"{data.incomplete_measurement_lines:,}"),
        ("Historical date range", f"{epoch.min():.3f}–{epoch.max():.3f}"),
    ]
    if extra:
        facts.append(("Additional observations", str(len(extra))))
    if weights is not None:
        all_pairs = tuple(observations) + extra
        assigned = observation_weights(all_pairs, weights)
        active = assigned > 0

        def wrms_label(orbit):
            if not np.any(active):
                return "n.a. (all weights zero)"
            fit_epochs = np.array([item.epoch for item in all_pairs])[active]
            fit_theta = np.deg2rad([item.theta_deg for item in all_pairs])[active]
            fit_rho = np.array([item.rho_arcsec for item in all_pairs])[active]
            east, north, _, _ = apparent_position(fit_epochs, orbit)
            squared_distance = (fit_rho * np.sin(fit_theta) - east)**2 + (fit_rho * np.cos(fit_theta) - north)**2
            normalized = assigned[active] / assigned[active].max()
            return f"{1000 * np.sqrt(np.sum(normalized * squared_distance) / normalized.sum()):.3f} mas"

        if published is not None:
            facts.append(("Published WRMS", wrms_label(published)))
        facts.append(("Optimized WRMS" if published else "Published WRMS", wrms_label(data.orbit)))
    for label, value in facts:
        info.text(0.02, y, label, fontsize=10.5, color="0.25")
        info.text(0.98, y, value, fontsize=10.5, ha="right")
        y -= 0.050

    info.text(
        0.02,
        0.018,
        f"Retained pairs: published residual <= {filter_mas:g} mas" if filter_mas is not None else ("North up • east left • all complete pairs shown" if published else "North up • east left • published orbit only (no optimization)"),
        fontsize=9,
        color="0.35",
        va="bottom",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output


def filter_by_residual(data: WDSData, additional, threshold_mas: float):
    """Apply one fixed cutoff against the published orbit, before optimization."""
    if not math.isfinite(threshold_mas) or threshold_mas < 0:
        raise ValueError("--max-residual-mas must be a finite, nonnegative number.")
    retained = {"historical": [], "additional": []}
    audit = []
    for source, observations in [("historical", data.observations), ("additional", additional)]:
        for index, item in enumerate(observations, 1):
            east, north, _, _ = apparent_position([item.epoch], data.orbit)
            theta = math.radians(item.theta_deg)
            residual = float(1000 * np.hypot(item.rho_arcsec * math.sin(theta) - east[0],
                                            item.rho_arcsec * math.cos(theta) - north[0]))
            if not math.isfinite(residual):
                raise ValueError(f"Nonfinite residual for {source} observation {index}.")
            keep = residual <= threshold_mas
            if keep:
                retained[source].append(item)
            audit.append({"source": source, "observation_index": index, "epoch": item.epoch,
                          "technique_code": item.technique, "theta_deg": item.theta_deg,
                          "rho_arcsec": item.rho_arcsec, "aperture_m": item.aperture_m,
                          "reference": item.reference, "theta_flag": item.theta_flag,
                          "rho_flag": item.rho_flag, "published_residual_mas": residual,
                          "threshold_mas": threshold_mas, "status": "retained" if keep else "excluded",
                          "reason": "" if keep else "Published positional residual exceeds cutoff"})
    if not retained["historical"]:
        raise ValueError("Residual cutoff leaves no historical pairs to plot or fit.")
    return replace(data, observations=tuple(retained["historical"])), tuple(retained["additional"]), audit


def write_filter_audit(path: Path, audit):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)


def optimize_orbit(data: WDSData, weights: dict[str, float], additional=()) -> tuple[WDSData, dict]:
    """Local weighted Cartesian least squares starting from the published orbit."""
    try:
        from scipy.optimize import least_squares
        import scipy
    except ImportError as exc:
        raise RuntimeError("Optimization requires SciPy: python3 -m pip install --user scipy") from exc
    observations = tuple(data.observations) + tuple(additional)
    assigned = observation_weights(observations, weights)
    active = assigned > 0
    if np.count_nonzero(active) < 4:
        raise ValueError("Optimization requires at least four positive-weight complete pairs.")
    epochs = np.array([item.epoch for item in observations])[active]
    if len(np.unique(epochs)) < 4:
        raise ValueError("Optimization requires at least four distinct positive-weight epochs.")
    theta = np.deg2rad([item.theta_deg for item in observations])[active]
    rho = np.array([item.rho_arcsec for item in observations])[active]
    target = np.column_stack((rho * np.sin(theta), rho * np.cos(theta)))
    w = assigned[active]
    # Normalize for solver scaling only: multiplying all input weights by a
    # constant must not change the fit. Square roots implement sum(w*r**2).
    root_w = np.sqrt(w / w.max())[:, None]
    start = data.orbit
    reference_epoch = float(np.median(epochs))
    phase = ((reference_epoch - start.periastron_epoch) / start.period_years) % 1
    initial = np.array([np.log(start.period_years), np.log(start.semimajor_arcsec),
                        np.deg2rad(start.inclination_deg), np.deg2rad(start.node_deg),
                        phase, start.eccentricity, np.deg2rad(start.omega_deg)])
    lower = [np.log(0.001), np.log(1e-8), 0, -np.inf, -np.inf, 0, -np.inf]
    upper = [np.log(1e7), np.log(1e5), np.pi, np.inf, np.inf, 0.999999, np.inf]

    def elements(parameters):
        log_period, log_axis, inc, node, phase_value, ecc, omega = parameters
        period = float(np.exp(log_period))
        return replace(start, period_years=period, semimajor_arcsec=float(np.exp(log_axis)),
                       inclination_deg=float(np.rad2deg(inc)), node_deg=float(np.rad2deg(node) % 360),
                       periastron_epoch=float(reference_epoch - phase_value * period),
                       eccentricity=float(ecc), omega_deg=float(np.rad2deg(omega) % 360),
                       grade="Not assigned", reference="Weighted local fit", uncertainties=(None,) * 7)

    def offsets(parameters):
        east, north, _, _ = apparent_position(epochs, elements(parameters))
        return target - np.column_stack((east, north))

    def residual(parameters):
        return (offsets(parameters) * root_w).ravel()

    fit = least_squares(residual, initial, bounds=(lower, upper), method="trf",
                        x_scale="jac", max_nfev=5000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"Optimization did not converge: {fit.message}")
    before = float(np.sum(w[:, None] * offsets(initial)**2))
    after = float(np.sum(w[:, None] * offsets(fit.x)**2))
    if after > before + max(1e-12, abs(before) * 1e-8):
        raise RuntimeError("Optimization increased the weighted error; no fit accepted.")
    fitted = elements(fit.x)
    # Report the equivalent periastron epoch closest to the published epoch.
    fitted = replace(fitted, periastron_epoch=fitted.periastron_epoch +
                     round((start.periastron_epoch - fitted.periastron_epoch) / fitted.period_years) * fitted.period_years)
    rank = int(np.linalg.matrix_rank(fit.jac))
    warnings = ["Local solution from the published orbit; global optimum is not established.",
                "Relative technique weights assume equal east/north precision; no formal parameter uncertainties estimated."]
    if rank < 7:
        warnings.append("Jacobian is rank deficient: the seven orbital parameters are not independently constrained.")
    if np.any(fit.active_mask):
        warnings.append("One or more parameters reached a search bound.")
    report = {
        "wds_id": data.wds_id, "method": "scipy least_squares trf, linear loss, one published-orbit start",
        "objective": "sum(weight * (east_residual_arcsec**2 + north_residual_arcsec**2))",
        "published_elements": asdict(start), "optimized_elements": asdict(fitted),
        "technique_weights": weights, "positive_weight_pairs": int(active.sum()),
        "zero_weight_pairs": int((~active).sum()), "weighted_sse_before": before, "weighted_sse_after": after,
        "weighted_positional_rms_arcsec_before": float(np.sqrt(before / w.sum())),
        "weighted_positional_rms_arcsec_after": float(np.sqrt(after / w.sum())),
        "converged": bool(fit.success), "solver_message": fit.message, "function_evaluations": fit.nfev,
        "jacobian_rank": rank, "active_bounds": fit.active_mask.tolist(),
        "bounds": {"period_years": [0.001, 1e7], "semimajor_arcsec": [1e-8, 1e5],
                   "inclination_deg": [0, 180], "eccentricity": [0, 0.999999]},
        "scipy_version": scipy.__version__, "warnings": warnings,
    }
    return replace(data, orbit=fitted), report


def choose_file() -> str:
    try:
        from tkinter import Tk, filedialog
    except ImportError as exc:
        raise RuntimeError("Supply the WDS text filename on the command line.") from exc
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filename = filedialog.askopenfilename(
        title="Choose an edited WDS historical-data file",
        filetypes=[("Text files", "*.txt"), ("All files", "*")],
    )
    root.destroy()
    return filename


def default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_published_orbit.png")


def read_weights_csv(path: str | Path) -> dict[str, float]:
    """Read explicit, finite, nonnegative technique weights; never infer defaults."""
    weights: dict[str, float] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if not {"technique_code", "weight"}.issubset(headers) or len(headers) != len(set(headers)):
            raise ValueError("Weights CSV requires unique headers including technique_code and weight.")
        for row in reader:
            code = (row.get("technique_code") or "").strip()
            if None in row:
                raise ValueError(f"Malformed weights CSV row {reader.line_num}.")
            if code in weights:
                raise ValueError(f"Duplicate technique code in weights file: {code!r}")
            try:
                weight = float(row.get("weight") or "")
            except ValueError as exc:
                raise ValueError(f"Missing or nonnumeric weight for technique {code!r}.") from exc
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"Weight for technique {code!r} must be finite and nonnegative.")
            weights[code] = weight
    if not weights:
        raise ValueError("Weights file contains no technique rows.")
    return weights


def observation_weights(observations: Iterable[Observation], weights: dict[str, float]) -> np.ndarray:
    """Match weights by exact technique code, in observation order."""
    observations = tuple(observations)
    missing = sorted({item.technique for item in observations} - weights.keys())
    if missing:
        raise ValueError(f"Weights missing for technique codes: {', '.join(repr(code) for code in missing)}")
    return np.array([weights[item.technique] for item in observations], dtype=float)


def create_weights_csv(data: WDSData, input_path: Path) -> Path:
    """Create an editable template from complete historical pairs only.

    Zero is an initial placeholder, not an estimated technique weight.
    Exclusive creation protects any previously edited weights.
    """
    counts = Counter(observation.technique for observation in data.observations)
    if not counts:
        raise ValueError("No complete historical measurements available for weights.")
    output = input_path.with_name(f"{input_path.stem}_weights.csv")
    try:
        with output.open("x", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["technique_code", "measurement_count", "weight"])
            for code, count in sorted(counts.items()):
                writer.writerow([code, count, 0])
    except FileExistsError as exc:
        raise ValueError(
            f"Weights file already exists: {output}. Your edits were preserved. "
            "Rename or move it before creating a new template."
        ) from exc
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot historical WDS measurements and the published ORB6 orbit."
    )
    parser.add_argument("input", nargs="?", help="Edited USNO/WDS text file")
    parser.add_argument("--additional", help="Optional CSV containing a new observation")
    parser.add_argument("--output", help="PNG or PDF output filename")
    parser.add_argument(
        "--residuals",
        metavar="CSV",
        help="Write complete theta-rho pairs and published-orbit residuals to CSV",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot after saving it")
    parser.add_argument("--weights", metavar="CSV", help="Load technique weights for future optimization and residual export")
    parser.add_argument("--optimize", action="store_true", help="Fit from the published orbit using --weights; save a separate experiment folder")
    parser.add_argument("--max-residual-mas", type=float, metavar="MAS", help="Exclude pairs whose published-orbit positional residual exceeds MAS; save a filtering audit CSV")
    parser.add_argument(
        "--create-weights", action="store_true",
        help="Create a zero-weight technique CSV beside the input file, without plotting; never overwrite",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.optimize and not args.weights:
        parser.error("--optimize requires --weights CSV")
    if args.optimize and (args.output or args.residuals):
        parser.error("--optimize automatically saves plot, residuals and report in a new experiment folder; omit --output and --residuals")
    if args.create_weights and args.max_residual_mas is not None:
        parser.error("--create-weights cannot be combined with --max-residual-mas")
    if args.create_weights and (args.additional or args.output or args.residuals or args.show or args.weights or args.optimize):
        parser.error("--create-weights must be used without --additional, --output, --residuals, --show, or --weights")
    input_name = args.input or choose_file()
    if not input_name:
        print("No file selected.")
        return 1
    input_path = Path(input_name)
    output = Path(args.output) if args.output else default_output(input_path)
    try:
        data = read_wds_file(input_path)
        if args.create_weights:
            weights_saved = create_weights_csv(data, input_path)
            print(f"Read {len(data.observations):,} complete historical measurements.")
            print(f"Saved weights template: {weights_saved}")
            print("Edit the weight column in Excel and save as CSV. Initial zeros are placeholders; zero in a future fit means exclusion.")
            return 0
        additional = read_additional_csv(args.additional) if args.additional else ()
        audit = None
        if args.max_residual_mas is not None:
            data, additional, audit = filter_by_residual(data, additional, args.max_residual_mas)
            excluded_count = sum(row["status"] == "excluded" for row in audit)
            print(f"Published residual cutoff {args.max_residual_mas:g} mas: {excluded_count} excluded; {len(audit) - excluded_count} retained complete pairs.", flush=True)
        weights = read_weights_csv(args.weights) if args.weights else None
        if weights is not None:
            observations = tuple(data.observations) + tuple(additional)
            assigned = observation_weights(observations, weights)
            positive = int(np.count_nonzero(assigned > 0))
            print(f"Loaded weights: {positive} measurements with positive weights; {len(assigned) - positive} with zero weights.", flush=True)
            unused = sorted(weights.keys() - {item.technique for item in observations})
            if unused:
                print(f"Unused weight codes: {', '.join(repr(code) for code in unused)}", flush=True)
            if not positive:
                print("All matched weights are zero: this template is not ready for optimization.", flush=True)
        if args.optimize:
            fitted, report = optimize_orbit(data, weights, additional)
            run_name = f"{Path(args.weights).stem}_fit_{datetime.now():%Y%m%d_%H%M%S_%f}"
            run_dir = input_path.parent / run_name
            run_dir.mkdir(exist_ok=False)
            if audit is not None:
                write_filter_audit(run_dir / "filter_audit.csv", audit)
                report["filter"] = {"max_residual_mas": args.max_residual_mas,
                                    "basis": "published orbit, applied once before fitting to historical and additional pairs",
                                    "excluded_pairs": excluded_count, "retained_pairs": len(audit) - excluded_count}
            for label, source in [("historical_input", input_path), ("weights_input", Path(args.weights))] + ([("additional_input", Path(args.additional))] if args.additional else []):
                contents = source.read_bytes()
                snapshot = run_dir / f"{label}{source.suffix}"
                snapshot.write_bytes(contents)
                report[label] = {"path": str(source.resolve()), "sha256": hashlib.sha256(contents).hexdigest(), "snapshot": snapshot.name}
            (run_dir / "fit_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            write_residuals_csv(data, run_dir / "published_residuals.csv", additional, weights)
            write_residuals_csv(fitted, run_dir / "optimized_residuals.csv", additional, weights)
            print(f"Weighted positional RMS: {1000 * report['weighted_positional_rms_arcsec_before']:.3f} -> {1000 * report['weighted_positional_rms_arcsec_after']:.3f} mas", flush=True)
            print(f"Saved optimization results: {run_dir}", flush=True)
            for warning in report["warnings"]:
                print(warning, flush=True)
            make_plot(fitted, run_dir / "orbit_comparison.png", additional, args.show, published=data.orbit, weights=weights, filter_mas=args.max_residual_mas)
            return 0
        if audit is not None:
            if not args.output:
                output = input_path.with_name(f"{input_path.stem}_filtered_{args.max_residual_mas:g}mas.png")
            audit_path = output.with_name(output.stem + "_filter_audit.csv")
            write_filter_audit(audit_path, audit)
            print(f"Saved filtering audit: {audit_path}", flush=True)
        saved = make_plot(data, output, additional=additional, show=args.show, weights=weights, filter_mas=args.max_residual_mas)
        residuals_saved = (
            write_residuals_csv(data, args.residuals, additional=additional, weights=weights)
            if args.residuals
            else None
        )
    except Exception as exc:
        print(f"Orbits could not process the file: {exc}", file=sys.stderr)
        return 2

    print(f"Read {len(data.observations):,} complete historical measurements.")
    print(f"Published orbit: {data.orbit.designation}, grade {data.orbit.grade}, {data.orbit.reference}")
    print(f"Saved plot: {saved}")
    if residuals_saved:
        print(f"Saved residuals: {residuals_saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
