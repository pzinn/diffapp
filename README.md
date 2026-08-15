# diffapp

`diffapp` analyses the asymptotic behaviour of a power series $F(x)$ using
inhomogeneous differential approximants.  It is a modern replacement for the
double-precision Fortran program `newgrqd`, see the "Legacy" section.

The fitted equation is

$$\sum_{k=0}^{M} Q_k(x)\left(x\frac{d}{dx}\right)^k F(x)=P(x),$$

where $Q_0,\ldots,Q_M$ and $P$ are polynomials of prescribed degrees,
and $Q_M(0) = 1$.  Candidate singularities are zeros of $Q_M$.  For a simple
root $x_c$, the reported exponent is

$$\theta=M-1-\frac{Q_{M-1}(x_c)}{x_c Q'_M(x_c)},$$

using the convention $F(x) \approx (1 - x/x_c)^\theta$.

## Current interface

To use `diffapp` without installing it, run these commands from the project
directory:

```console
PYTHONPATH=src python -m diffapp fit coefficients.txt
PYTHONPATH=src python -m diffapp sweep coefficients.txt --root-min 0.2 --root-max 0.3
PYTHONPATH=src python -m diffapp extend coefficients.txt --terms 30
PYTHONPATH=src python -m diffapp extend-sweep coefficients.txt --terms 30
```

Use `--backend mpmath --precision 100` to solve and find roots at arbitrary
precision.  The default `float64` backend equilibrates rows and columns before
calling SciPy's LAPACK solver and reports the scaled condition number and
backward residual.

## Sweeps

By default, `sweep` examines first- and second-order equations, varies `P` from
degree zero up to a modest length-dependent maximum, balances the `Q` degrees
to within two, and uses all or nearly all the available coefficients. It
rejects numerically rank-deficient fits, groups nearby roots, and reports
clusters occurring in at least half the accepted approximants.

Specifying a physical-root interval is important when its approximate location
is known:

```console
PYTHONPATH=src python -m diffapp sweep coefficients.txt --root-min 0.2 --root-max 0.3
```

With an interval, each approximant contributes its nearest-origin almost-real
root in that interval. Without one, every root is considered and clustered.
An approximant often has several positive real roots, and mathematics alone
does not decide which represents the singularity under study.

The main controls are:

- `--orders 1,2` accepts comma-separated orders or inclusive ranges such as
  `1:3`.
- `--p-degrees -1,0:4` overrides the automatic `P` range; `-1` denotes a
  homogeneous equation.
- `--degree-spread 2` limits the difference between the largest and smallest
  `Q` degree.
- `--max-terms-omitted 10` controls how far below the full series length the
  generated fits may go.
- `--cluster-tolerance 1e-4` sets the hybrid relative distance used to group
  roots, and `--minimum-cluster-fraction 0.5` sets the recurring threshold.
- `--show-approximants` adds the individual roots, exponents, condition
  numbers, and combined cancellation score to the default cluster table.
- `--all-clusters` also displays weakly supported and singleton clusters.
- `--output json` and `--output csv` provide structured output for subsequent
  analysis.

For `N` coefficients, the automatic `P` degrees are
`0..min(8, max(1, floor(N/3)))`. The default family omits at most
`min(10, N-3)` trailing coefficients. These are deliberately broad exploratory
defaults; a known root interval and a narrower degree range usually give a
more interpretable scientific comparison.

## Extending a series

`extend` uses one approximant's recurrence to extrapolate a series. By default
it generates ten additional coefficients. `--terms` overrides this with the
desired total coefficient count, including the supplied coefficients:

```console
PYTHONPATH=src python -m diffapp extend coefficients.txt --terms 30
```

The default table distinguishes supplied and extrapolated coefficients. Use
`--predicted-only` to omit the supplied prefix, or select `--output plain`,
`json`, or `csv`. The command accepts the same `--q-degrees`, `--p-degree`,
`--order`, and numerical-backend options as `fit`.

An automatic fit deliberately reserves trailing coefficients for model
selection. `extend` reports the selected model's normalized holdout error, but
preserves the supplied values in its output. This checks the recurrence before
using it beyond the known series.

`extend-sweep` applies the same idea to a balanced family:

```console
PYTHONPATH=src python -m diffapp extend-sweep coefficients.txt --terms 30 --show-models
```

It also defaults to ten additional coefficients when `--terms` is omitted.

For each new coefficient it reports the ensemble median, median absolute
deviation (`MAD`), relative spread, and number of contributing models. Models
that omit trailing input coefficients are scored against those holdouts.
`--maximum-holdout-error` can reject models above a chosen normalized error;
models using every supplied coefficient have no holdout score. Rank-deficient
models are rejected by default, as in a root sweep. JSON and CSV output are
also available.

Extension is extrapolation, not additional data. Agreement across the family
and small holdout errors are useful diagnostics, but uncertainty can grow very
quickly with the distance beyond the known coefficients.

For a single fit, `Q` and `P` degrees are optional. Automatic selection tests
balanced first- and second-order candidates with homogeneous, constant, and
linear `P`. Each candidate is fitted to a prefix and asked to predict the
unused trailing coefficients. The default minimizes the maximum normalized
holdout error, preferring the simpler specification when errors differ only at
the numerical-noise floor. The selected holdout count and errors are printed.
Very short series, which cannot reserve a useful holdout, fall back to a
full-length balanced fit.

Use `--order` or `--p-degree` to restrict this validated search, or specify
`--q-degrees` for one exact specification. An automatic single fit is a useful
starting point, not a replacement for examining a family of nearby
approximants.

Each candidate singularity also has three cancellation diagnostics:

- `cancellation: max` is the largest normalized value of every other active
  `Q` polynomial and `P` at the candidate root. It can be small only when all
  terms nearly share the factor.
- `root-distance` is the largest normalized distance to the nearest root of
  each of those polynomials.
- `gcd-svd` is the largest pairwise smallest/ largest singular-value ratio for
  their coefficient-scaled Sylvester matrices. It tests for an approximate
  polynomial GCD, although not necessarily at this particular root.

Values near zero support a removable common factor. They should be interpreted
alongside the fit condition number; an ill-conditioned fit may not determine
its polynomial coefficients accurately enough for a reliable verdict.

The fit also estimates numerical rank. Automatic fitting rejects
rank-deficient specifications and tries a smaller balanced approximant.
Explicit fits retain the requested specification but print a warning that
their roots may depend on working precision. Leading coefficients below the
reported uncertainty-based relative cutoff are omitted when calculating
effective polynomial degrees and roots; nominal fitted coefficients remain
available unchanged through the Python API.

## Legacy format and attribution

This project is based on Anthony J. Guttmann's fortran program `newgrqd.f`.
G. S. Joyce and A. J. Guttmann introduced the recurrence-relation method in
[1972][guttmann-joyce-1972]. The inhomogeneous differential-approximant form was
developed in 1979 by [M. E. Fisher and H. Au-Yang][fisher-au-yang-1979] and by D. [L. Hunter and G. A. Baker][hunter-baker-1979].

The original program reads a fixed-column Fortran card deck. Its records are,
in order:

| Record | Fortran format | Meaning |
| --- | --- | --- |
| Title | `10A8` | An 80-character label |
| Output flags | `3I2` | Three print/output controls; a first flag of 9 ends input |
| Series and bias controls | `8I2,D40.30` | Highest series power, counts of prescribed singularities and exponents, and the singularity increment |
| Coefficients | `D50.0` | The `N + 1` coefficients, from the constant term through power `N` |
| Approximant sweep | `12I5` | Bounds controlling equation order and polynomial degrees |
| Optional biases | `2D40.30`, then `I4,D40.30` | Prescribed singularities and critical exponents |

`diffapp` currently accepts a practical compact transcription rather than that
full card deck:

```text
output_flag_1 output_flag_2 output_flag_3
N
c_0
c_1
...
c_N
min_order max_order min_Q_degree max_Q_degree min_Q1_offset max_Q1_offset min_Q2_offset max_Q2_offset min_P_degree max_P_degree [min_terms max_terms]
[terminator]
```

The sweep line must contain at least ten integers; the final two default to
zero. `fit --format legacy` reads the coefficients but otherwise uses the model
selected on the command line (or by automatic selection). `legacy-sweep` uses
the order, common `Q`-degree, and `P`-degree bounds from the compact control line
and constructs the neighbouring balanced degree family. The other stored sweep
fields, the original title/control record, fixed-column parsing, and biased
singularities are not yet implemented.

Decimal coefficient strings are preserved until the numerical backend has been
chosen. For example:

```console
PYTHONPATH=src python -m diffapp fit legacy-input.dat --format legacy
PYTHONPATH=src python -m diffapp legacy-sweep legacy-input.dat
```

[newgrqd-provenance]: https://doi.org/10.1007/s10955-005-4409-y "Spanning Forests and the q-State Potts Model in the Limit q to 0; Appendix A and reference 158"
[guttmann-joyce-1972]: https://doi.org/10.1088/0305-4470/5/9/001
[fisher-au-yang-1979]: https://doi.org/10.1088/0305-4470/12/10/014
[hunter-baker-1979]: https://doi.org/10.1103/PhysRevB.19.3808
