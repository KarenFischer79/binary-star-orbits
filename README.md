# Orbits — Part 1

This Python program reads one of the edited USNO historical-data files, plots
the complete epoch/position-angle/separation measurements, calculates the
published apparent orbit, and displays the published orbital parameters.

Optional weighted optimization starts from the published orbit using an editable
technique-weights CSV.

## Easiest way to run it on the Mac

1. Double-click **Run Orbits.command**.
2. If macOS asks, allow it to run in Terminal.
3. Choose an edited historical file such as:
   `wds16289+1825_edit.txt`
4. The program saves a PNG beside the selected data file and opens the plot.

If macOS refuses to open the launcher because it was downloaded or created by
another application, Control-click it, choose **Open**, and then choose
**Open** once more.

## Terminal use

Install required packages with:

</> bash
python3 -m pip install -r requirements.txt


```bash
python3 orbits.py "/path/to/wds16289+1825_edit.txt" --show
```

Choose a particular output name:

```bash
python3 orbits.py input.txt --output published_orbit.pdf
```

Add the project's new observation CSV:

```bash
python3 orbits.py input.txt --additional TR_WDS16289+1825_addedObservation.csv --show
```

Write the complete theta-rho pairs and published-orbit residuals to an
Excel-compatible CSV file:

```bash
python3 orbits.py input.txt --residuals WDS16289_residuals.csv --show
```

The residual CSV includes observed and calculated theta and rho, wrapped
theta O-C, rho O-C, east and north O-C, and the total two-dimensional residual
in arcseconds and milliarcseconds. Aperture is blank when it is absent or when
the technique code indicates space-based astrometry (codes beginning with H,
including Gaia, Hipparcos, and Tycho).

## Create a weights template

From the InputOrbit folder, run:

```bash
python3 ../orbits.py wds16289+1825_edit.txt --create-weights
```

This creates `wds16289+1825_edit_weights.csv` beside the historical input.
It contains `technique_code`, `measurement_count`, and `weight`, with one
row per technique among complete epoch/theta/rho measurements only. Counts
exclude incomplete records. A blank technique code remains blank for review.
All weights start at zero. Open in Excel, edit the weight column, and save
as CSV. These zeros are placeholders; zero in a future weighted fit will
mean exclusion. Automatic TBW defaults are not implemented.

This option creates only the template, without opening or saving a plot.
Use it separately from plot, residual, and additional-observation options.
Existing weights files are never overwritten. To deliberately regenerate a
template, first rename or move the existing file to preserve your edits.

## Load weights in preparation for optimization

From InputOrbit:

```bash
python3 ../orbits.py wds16289+1825_edit.txt --weights wds16289+1825_edit_weights.csv --residuals WDS16289_weighted_input.csv --show
```

Weights are matched by exact technique code, not row order. The CSV must
have `technique_code` and `weight` headers; `measurement_count` is informational.
Values must be finite, nonnegative numbers. Duplicate codes and missing weights
for any complete observation (including optional additional observations) are
errors. Unused codes are reported. A blank technique code can have its own row.
Zero means exclusion from a future fit; all-zero files can be loaded for review
but are reported as not ready for optimization.

Without `--optimize`, the plot and raw residuals show every complete measurement against the
published orbit. With `--weights`, residual exports also include the matched
`technique_weight`. No automatic TBW assignment is performed.
The loader does not change the weights file.

## Optimize with a selected weights file

From InputOrbit:

```bash
python3 ../orbits.py wds16289+1825_edit.txt --weights wds16289+1825_edit_weights.csv --optimize --show
```

The published orbital elements are read from the historical file. To compare
experiments, save separate CSV files (for example `weights_trial2.csv`) and
change only the filename after `--weights`. Every run starts from the published
orbit and creates a new timestamped folder beside the historical file, named
after the selected weights file. Even repeated runs preserve earlier results.

Each folder contains a published/optimized comparison PNG, published and
optimized residual CSVs with weights, and `fit_report.json` with both sets of
elements, convergence status, weighted positional RMS, search bounds, and
input hashes. Copies of the historical, weights, and optional additional files
record the inputs used. CSVs and the report are saved before the plot opens.
Omit `--show` to save without opening a window. Do not combine optimization
with `--output`, `--residuals`, or `--create-weights`; output names are automatic.

The fit uses SciPy bounded nonlinear least squares and minimizes
`sum(weight * (east_residual_arcsec**2 + north_residual_arcsec**2))`.
All seven elements vary, using log period, log semimajor axis, and phase at
the median epoch internally for scaling. Weights are relative inverse-variance
style coefficients, not standard deviations. No weights are estimated or changed.
Zero-weight pairs remain in the plots and residual exports but do not affect
the fit. At least four positive-weight pairs at distinct epochs are required;
this minimum does not guarantee a well-constrained orbit.

This is a local fit with one starting orbit, not a global search. A converged
fit is not proof of a unique physical solution. Rank deficiency and active
bounds are reported. Formal parameter uncertainties and orbit grades are not
assigned to fitted elements. Weighted RMS values across different weighting
schemes should not be used alone to rank the schemes.

## Residual filtering before plotting or fitting

Use `--max-residual-mas 300` to exclude complete pairs whose positional
residual against the published orbit exceeds 300 mas. Equality is retained.
The cutoff is applied once before optimization, to historical and optional
additional observations. It is independent of technique weights and is not
recomputed against the optimized orbit. Without this option no cutoff applies.

```bash
python3 ../orbits.py wds18540+3723_edit.txt --weights wds18540+3723_edit_weights.csv --max-residual-mas 300 --optimize --show
```

Both published and optimized WRMS use the same retained observations.
The plot and residual exports contain only retained pairs. The original input
files are unchanged. `filter_audit.csv` in the optimization folder lists every
complete pair with its original sequence index, epoch, code, reference, observed
theta/rho, published residual, cutoff, retained/excluded status and reason.
The report also records the cutoff and counts.

For a published-only review, omit `--optimize`. A filtered plot and matching
`_filter_audit.csv` are saved beside the input unless `--output` is supplied.
The audit is saved before the window opens. Reusing published-only output
names overwrites those outputs. Filtering cannot accompany `--create-weights`.
At least one historical pair must remain, and optimization also requires
at least four positive-weight pairs at distinct epochs.

## Required Python packages

- Python 3.10 or newer
- NumPy
- Matplotlib
- SciPy (optimization only)

These packages are already available in the current Codex Python environment.

## Plot convention

- Position angle is measured from north through east.
- North is up.
- East is shown on the left, following the usual double-star plotting
  convention.
- Missing theta/rho records are counted and omitted from the plot.
