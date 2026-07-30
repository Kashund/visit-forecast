# Visit Forecast 0.2.0 release verification

Date: 2026-07-23

Branch: `agent/visit-forecast-v0-2-hardening`

GitHub base: `Kashund/visit-forecast@4bab2b2ddbfcfe2137b3daa6555c38265fd9389d`

## Release scope

- Preserve the current GitHub history while replacing the application with the
  reviewed 0.2.0 rebuild.
- Correct calibration leakage, duplicate-month handling, incomplete department
  totals, Prophet covariate intervals, and cross-session history exposure.
- Add multi-model forecasting, department and scenario workflows, stronger
  input validation, reproducible dependencies, and CI enforcement.
- Remove generated Python bytecode that was tracked in the previous repository.

## Verification evidence

| Check | Command or method | Result |
| --- | --- | --- |
| Dependency resolution | `uv lock --check` | Passed; 91 packages resolved |
| Locked test suite | `.venv/bin/python -m pytest -q` | 89 passed; 9 expected statsmodels warnings |
| Source compilation | `.venv/bin/python -m compileall -q src apps scripts` | Passed |
| Wheel build | `uv build --wheel` | Built `visit_forecast-0.2.0-py3-none-any.whl` |
| Streamlit component smoke | `streamlit.testing.v1.AppTest` over Home and all four pages | 5 pages; 0 exceptions |
| Live readiness | `GET /_stcore/health` | HTTP 200 |
| Desktop browser QA | Home and Run Forecast pages | No visible exceptions or console errors |
| Mobile browser QA | All five pages at 390x844 | No visible exceptions or horizontal overflow |
| Live forecast workflow | Synthetic sample, seasonal naive, conformal interval, manual tuning | Forecast validation and plot rendered; no visible or console errors |
| Bundled sample contract | Both CSV samples through `forecast_visits` | Passed in the locked test suite |
| Default session isolation | Independent session-state regression with stale disk history | Passed in the locked test suite |
| Diff hygiene | `git diff --check` and focused credential-pattern scan | Passed |

The visual evidence is stored with the release review outputs:

- `visit-forecast-desktop-home.png`
- `visit-forecast-desktop-run.png`
- `visit-forecast-mobile-run.png`
- `visit-forecast-live-result.png`

## Known remaining risks

1. The release has not been validated against the real production dataset.
   Before changing production traffic, run a golden comparison using a
   de-identified production-shaped dataset and approve any material forecast
   deltas.
2. The target hosting platform has not yet run this exact branch. Keep history
   persistence disabled for multi-user hosting and complete a post-deploy smoke
   test before considering rollout complete.
3. Statistical-model warnings on short synthetic ARIMA series are expected and
   non-fatal, but production monitoring should track model failures, fallback
   usage, coverage, and forecast drift.

## Deployment gate

Merge only after GitHub Actions passes. Deploy the merged commit to a staging
environment first, verify health and one forecast with approved non-sensitive
data, then promote the same immutable commit to production.
