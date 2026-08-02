# diffapp

`diffapp` analyses the asymptotic behaviour of a power series using
inhomogeneous differential approximants.  It is a modern replacement for the
legacy `newgrqd.f` program in this directory.

The fitted equation is

\[
\sum_{k=0}^{M} Q_k(x)\left(x\frac{d}{dx}\right)^k F(x)=P(x),
\]

with `Q_M(0) = 1`.  Candidate singularities are zeros of `Q_M`.  For a simple
root `x_c`, the reported exponent is

\[
\theta=M-1-\frac{Q_{M-1}(x_c)}{x_c Q'_M(x_c)},
\]

using the convention `F(x) ~ (1 - x/x_c)^theta`.  Thus a generating function
that diverges with exponent `gamma > 0` has `theta = -gamma`.

## Current interface

Run directly from a checkout:

```console
PYTHONPATH=src python -m diffapp fit coefficients.txt
PYTHONPATH=src python -m diffapp fit zinn.dat --format legacy \
  --q-degrees 5,5 --p-degree 1
PYTHONPATH=src python -m diffapp legacy-sweep zinn.dat \
  --root-min 0.0875169 --root-max 0.0877169
```

Use `--backend mpmath --precision 100` to solve and find roots at arbitrary
precision.  The default `float64` backend equilibrates rows and columns before
calling SciPy's LAPACK solver and reports the scaled condition number and
backward residual.

For a sweep, specifying a physical-root interval is important: an approximant
often has several positive real roots, and the closest one need not represent
the singularity under study.

For a single fit, `Q` and `P` degrees are optional. The automatic choice uses
all available coefficients, normally selects a second-order equation with
`P` degree 1, and distributes the `Q` degrees as evenly as possible. Short
series fall back to a homogeneous or first-order approximant. Use `--order`
to retain automatic degree balancing at a chosen differential-equation order.
An automatic single fit is a convenient starting point, not a replacement for
examining a family of nearby approximants.

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

The legacy reader preserves decimal input values until the numerical backend
has been chosen.  It supports the compact format used by `zinn.dat`,
`test.dat`, and `orbvar.dat`; fixed-column card-deck input and biased
singularities are not yet implemented.
