# Visit Forecast 0.2.0 post-merge verification

Date: 2026-07-30

Merged PR: `Kashund/visit-forecast#12`

Merge commit: `4fa88ba22220d4f8a90b22a1899596980e29b63f`

## Merged-state evidence

- The merge commit's Git tree exactly matches the reviewed PR head
  `c2ddadb509b2f17e6cae432e9d665830f5ca377a`.
- The `main` push workflow completed successfully.
- The locked post-merge test suite passed with 89 tests and nine expected
  Statsmodels warnings.

## Packaging issue found during staging smoke

The documented CLI failed after the default editable `uv sync` on the local
Python 3.13 runtime:

```text
ModuleNotFoundError: No module named 'visit_forecast'
```

Python skipped the hidden `__editable__.visit_forecast-0.2.0.pth` file created
by the editable installation. The existing tests did not detect this because
their configuration adds `src/` directly to `sys.path`.

## Correction

- Deployment and CI use `uv sync --locked --extra dev --no-editable`.
- All `uv run` verification and launch commands specify `--no-editable`.
- CI now imports `visit_forecast` from the installed package.
- CI now starts the documented CLI with `--help`.
- README installation, CLI, Streamlit, and deployment commands match the
  verified non-editable environment.

## Corrected-environment evidence

| Check | Result |
| --- | --- |
| Installed package import | Passed from `.venv/site-packages` |
| CLI help | Passed |
| CLI synthetic forecast | Passed; seasonal-naive metrics and fiscal summary rendered |
| Full locked test suite | 89 passed |
| Source compilation | Passed |
| Wheel build | `visit_forecast-0.2.0-py3-none-any.whl` built |
| Local Streamlit health | HTTP 200 using the installed-package launch command |
| Synthetic golden baseline | Exact match at `rtol=1e-12`, `atol=1e-12` |

## External gates

- No GitHub environments or prior deployments are configured for this
  repository, so there is no discoverable staging target.
- A production-data golden comparison still requires an approved,
  de-identified production-shaped dataset.
- Keep `VISIT_FORECAST_PERSIST_HISTORY` unset on any multi-user host.
