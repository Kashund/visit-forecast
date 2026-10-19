# README capability review and implementation plan

Reviewed: 2026-10-19. Baseline: `main` at `d6afcf191f6208e40bad4580892362c16ac934aa`.

## Implementation status — 2026-10-19

The original review below documents the baseline, not the current implementation. Application changes have now been implemented in this working tree; they are not committed, deployed, or published.

| Delivery item | Current result |
| --- | --- |
| 1. Shared ingestion/validation | Implemented: CSV/XLSX reader, strict row diagnostics, shared Data Check rules, and department/month coverage. |
| 2. Department totals | Implemented: common forecast origin/calendar, rejected unknown requests, exclusion precedence, per-date completeness, and explicit failures in workbooks. |
| 3. Evaluation/uncertainty | Implemented: purged unobservable calibration outcomes, a reserved final block when sufficient history exists, preserved requested coverage, model-aware methods, and stored/exported fallback bounds. Empirical intervals are explicitly not advertised as a formal nominal coverage guarantee. |
| 4. Auto selection/tuning | Implemented: zero-safe ranking, allowed-candidate enforcement, explicit failed comparisons/variants and shared validation windows, covariates threaded through tuning, and skipped Prophet tuning when excluded. |
| 5. Capacity/scenarios | Implemented: shared four-phase editor, one-month support, full percentage ranges, safe horizon clamping, retained baseline bounds, scenario baseline reuse, and stale-result warnings. |
| 6. Fiscal Summary | Implemented: one monthly/quarterly/yearly selector drives the table, chart, and CSV; actual/original/adjusted values reconcile; partial coverage is visible; view changes do not refit or create history entries. |
| 7. Interfaces/documentation | Implemented: public API exports, CLI forecast/department/scenario workflows, core flags and file outputs, dataset sharing, department drilldown, JSON/Excel manifests, and all-workflow session history. |
| 8. Engineering validation | Implemented: regression and populated-page interaction tests, isolated installed-wheel verification, CI extension, atomic locked history persistence, launcher health/occupied-port checks, and bounded session-private forecast reuse. Final verification results are recorded below. |

A performance reference run during implementation used aggregate synthetic data, 60 historical months, 3-month forecast, Auto candidates ETS/Theta/seasonal naive, conformal intervals, and a reserved final block completed in **9.952 seconds** in this cloud instance. This is a current reference measurement, not a before/after speedup claim or a production latency guarantee. Forecast cache capacity is three runs per session, keyed by data, settings, and engine source; phase-only scenario runs reuse a separate baseline. Fitting remains serial to avoid uncontrolled native-model concurrency, with progress callbacks. Streamlit's run interruption remains available; custom asynchronous job cancellation/resource budgets are a future operational extension.

Remaining follow-ups are explicit:

- Production accuracy, calibration stability, and chosen acceptance thresholds require an approved operational dataset. No production-data readiness claim is made.
- Hosted rollout checks require the target deployment. Full browser/mobile/accessibility QA remains unperformed; AppTest checks populated workflows but cannot establish visual/browser behavior.
- Holiday/clinic calendars and additional regressors require calendar definitions and operational-data validation. The current covariate remains Monday–Friday business days.
- Hierarchical reconciliation, calibrated aggregate uncertainty, optional projected FY total columns, and stricter asynchronous job budgets are further product enhancements. Independent department point forecasts and their correctly labeled sums are implemented; coherence or aggregate interval calibration is not claimed.

A subsequent runtime eligibility check established that the pinned Darts ARIMA wrapper requires 30 observations even in its non-seasonal configuration. The implementation reports that limit explicitly and excludes an ineligible ARIMA candidate from Auto; ETS/Theta retain short-history non-seasonal variants. README now documents this supported limit rather than promising an ARIMA fit the dependency cannot perform.

The factual review findings and acceptance matrix below are retained for traceability. Statements such as “code inspection” and “executed reproduction” in those sections refer to the original baseline.

## Implementation verification

- Full suite before the final shared-window refinement: **109 passed, 9 statsmodels warnings**, in 268.86 seconds. All tests executed in that run.
- Final affected-code regression pass, including the shared-window and short-ARIMA eligibility checks: **70 passed, 1 deselected, 5 statsmodels warnings**, in 21.02 seconds. The deselected expensive joint-tuning integration test passed in the full run; it is not disabled in CI. Counts overlap and must not be added together.
- Populated AppTest workflows cover all three fiscal views without refitting, one-month forecasts, four-phase scenarios and baseline reuse, department diagnostics/exports, cache invalidation, and session history. Excel exports were read back to verify numeric content.
- Built wheel installed and verified with Python isolated mode from outside the checkout: public imports, a synthetic forecast, monthly fiscal output, CSV/Excel/JSON creation, and manifest passed.
- Locked dependency check, source compilation, undefined-name checks, and diff hygiene passed. The latest Streamlit process returned `ok` from its live health endpoint after restart.
- The reusable environment installation/start draft now forces refresh of the non-editable package after source changes. Draft persistence is confirmed; publishing remains a separate environment-settings action.

## Assessment

Most major README features exist: five forecasting models, automatic selection, calendar covariates, Prophet diagnostics, department forecasts, scenario comparison, fiscal summaries, exports, and session history. The work should extend and correct those implementations rather than rebuild them. The largest gaps are inconsistent access across interfaces and correctness at data, horizon, aggregation, and uncertainty boundaries.

The environment setup previously completed the unchanged baseline suite: **89 passed, 9 statsmodels warnings**, plus compilation, wheel construction, synthetic CLI forecasting, and initial rendering of all five Streamlit pages. Those checks did not cover several failures reproduced in this review. A healthy server or passing baseline suite does not establish feature completeness or statistical validity.

The initial review changed documentation only. Application fixes were subsequently implemented as recorded above; deployment, real production-data validation, and browser visual/accessibility audit remain unperformed. Findings below distinguish executed reproductions, code-inspection findings, and proposed enhancements.

## README feature coverage

| README promise | Current coverage | Work needed |
| --- | --- | --- |
| Prophet, ETS, Theta, SARIMA, seasonal naive, Auto | Implemented in `src/visit_forecast/models.py`; main and department pages expose all choices | Scenario page omits explicit seasonal naive. Fix zero-demand scoring and candidate-respecting fallback. |
| Auto comparison using rolling MAPE/RMSE and naive baseline | Implemented, with candidate diagnostics | Candidate lists can omit naive; all-invalid scores default to Prophet even when excluded. Add zero-safe metrics, explicit eligibility/failure reasons, and a separate evaluation set for the chosen configuration. |
| Non-Prophet models use conformal intervals | Partially consistent | With Auto uncertainty and unavailable coverage, ETS can be labeled as using Prophet while its active bounds are conformal. Restrict methods by selected model and make provenance consistent. |
| Seasonal fallback for short history | Implemented | Explain actual variant and training requirements. The seasonal threshold is 26 observations for ETS/Theta/SARIMA, whereas README says two full seasons for SARIMA. Validate boundary behavior and document it accurately. |
| Calendar covariates in fit, backtest, model comparison | Implemented for Prophet/SARIMA | Prophet tuning searches do not receive covariates. Scenario page has no covariate control; CLI has no covariate flag. Clarify weekdays versus a holiday-aware clinic calendar. |
| Prophet trend/yearly/calendar component plots and native CV | Implemented on Run Forecast | Failures are broadly swallowed. Make unavailable/unsupported/failed states distinct and exercise populated sections in CI. |
| CSV/XLSX, pasted CSV, synthetic input | UI readers support them; main page alone supports paste | Shared `load_data()` always calls `read_excel`, breaking CSV filenames in Python/CLI. Consolidate ingestion and validation. |
| Finite non-negative visits, monthly normalization, duplicate summation, complete history | Normalization, duplicate summation, negative rejection, and single-series gap checks exist | Non-finite/nonnumeric visits are silently dropped. Aggregate All can hide a missing department/month. Data Check uses different validation rules. |
| Up to four capacity phases, loss through 100%, additions over 100% | Service supports these ranges | Main/scenario UI caps percentages at 50%; scenario UI has only three phases. Both phase editors fail at a one-month horizon. |
| Disabled phases ignored; horizon changes handled safely | Works for well-typed disabled phases and main-page range clamping | Disabled dictionary values are converted before checking enabled. Scenario range changes silently reset, without main-page clamping/warning. |
| Per-department independent models, chart, summary, Excel/CSV, partial totals | Implemented | Completeness is based on run failures, not contributions per calendar month. Different history endpoints produce misleading complete totals. Add filters, diagnostics drilldown, and explicit failures in exports. |
| Scenario A/B from one baseline, deltas, FY tables, three Excel sheets | Implemented | Changing phases requires pressing Run comparison, which refits the baseline. Cache baseline independently; mark stale controls/results. Reapplying phases leaves interval columns unchanged. |
| Prediction bands, MAE/RMSE bands, recorded fallback source | Explicit bounds and chart-only RMSE approximation exist | RMSE fallback is drawn but not stored in exported forecast data/provenance. Review chronological separation of calibration/evaluation and target coverage handling. |
| Session-only history, opt-in local persistence, last 20 runs | Implemented on Run Forecast | Department/scenario runs are not recorded. Disk write errors are silent; no atomic updates or concurrency protection. Define whether history covers every workflow. |
| Apr–Mar FY summaries, samples, notebook workspace, locked CI/build | Implemented | Notebook tooling is optional/unlisted. CI lacks page interactions, populated result sections, file round trips, and installed-wheel isolation. Home still describes Prophet-only forecasting. |
| Added user requirement: fiscal monthly, quarterly, and yearly views | Fiscal Summary currently offers year and quarter; shared aggregation has no monthly mode | Add a single granularity selector that updates both table and chart for historical, original forecast, and adjusted forecast values, with matching exports. See the fiscal rollup specification below. |

## Confirmed failures and priority

P0 means resolve before relying on outputs for capacity decisions. P1 means core feature parity/reliability. P2 means workflow quality and operational maturity. These are implementation priorities, not claims of a production outage.

### P0 — misleading complete totals and incomplete input validation

**Executed reproductions:**

- Generate departments A/B for Jan 2020–Dec 2023; remove B's Nov/Dec 2023 rows; request a two-month departmental seasonal-naive forecast. No department fails. B forecasts Nov/Dec 2023 and A forecasts Jan/Feb 2024. The result contains four total rows, all marked `All (sum of departments)`, `aggregate_complete=True`, and count 2, although each row contains one department. Grouping by both `ds` and relative `forecast_month` can also split the same date when origins differ. See `service.py:1154`.
- With A present Jan/Feb/Mar and B present Jan/Mar, aggregate All accepts monthly totals 30/10/30 without identifying B's missing February. The synthetic decline can be mistaken for real demand change. See `io.py:102`.
- With otherwise valid duplicate-month rows, `Visits=inf` and `Visits='bad'` disappear silently while valid rows remain. Data Check calls such input invalid, but forecasting can proceed with reduced totals. See `io.py:70`.

**Implement:** shared validation with row-level errors and a department-by-month coverage report; explicit treatment of inactive departments versus missing submissions; common forecast origin/date grid; per-date contributing/expected department counts; rejection or explicit partial labeling of incomplete totals. Validate unknown requested department names instead of silently intersecting them away. Do not fill missing observations with zero automatically.

**Accept when:** missing rows cannot become silent demand reductions; two different endpoints cannot create a complete total; every complete total contains every required department once per date; chart and CSV/XLSX completeness agree; valid duplicates still sum correctly. Define boundary coverage against the selected training window, including newly opened/closed departments.

### P0 — interval evaluation is not fully separated in time

**Executed reproduction:** construct monthly cutoffs Jan–Aug 2023 with a three-month horizon and call `_partition_conformal_residuals`. Calibration includes target dates through Sep 2023, while evaluation begins at the Jul 2023 origin; Aug/Sep target dates occur in both partitions. The implementation separates origin IDs but does not purge calibration outcomes unavailable at the first evaluation origin. See `model_prophet.py:263`.

This contradicts the stronger implication of independent held-out coverage. It does not prove the numerical coverage is always optimistic; it shows that the current split does not enforce the claimed temporal independence.

**Implement:** a chronological evaluation contract that only uses residuals whose outcomes are observable by each evaluation origin. Use a purge gap for overlapping multi-step windows or an expanding chronological calibrator. Separate model/hyperparameter/method selection from final evaluation. Persist cutoff/date ranges, horizon, residual counts, and validation status. Review finite-sample quantiles and dependence assumptions before describing these intervals as having guaranteed nominal coverage.

**Accept when:** no calibration target crosses the evaluation information boundary; target dates used for final evaluation are excluded from selection/calibration where required; unavailable coverage stays unavailable; diagnostics show sample counts and the evaluated horizon. Add 1-, 12-, and 36-month cases and short histories. Preserve the existing no-resubstitution behavior.

### P1 — input and public interface parity

**Executed:** `load_data(False, 'data/sample/synthetic_visits.csv', ...)` raises `ValueError: Excel file format cannot be determined`. UI CSV ingestion bypasses this loader, which explains differing behavior. See `io.py:53`.

**Code inspection:** package-root exports contain only `forecast_visits` and `ForecastResult`. Department/scenario helpers are available from `visit_forecast.service`, but README does not show that import. CLI exposes model/capacity options but lacks covariates, exclusions, uncertainty selection, output files, and department/scenario workflows. Those additional CLI operations are proposed parity improvements; README does not explicitly promise all of them as CLI flags.

**Implement:** one CSV/XLSX reader shared by API/CLI/UI, clear unsupported-format errors, documented public imports, and a defined interface support matrix. Expose core missing flags and machine-readable output first; add department/scenario CLI commands if full CLI parity is adopted.

**Accept when:** included CSVs and equivalent XLSX files work by filename in API and CLI; legacy XLS is rejected clearly; import examples run in a clean installed environment; selected outputs and settings round-trip through exports.

### P1 — capacity UI and scenario behavior

**Executed:** setting forecast months to 1 produces a phase slider error on both forecast pages. Run Forecast catches it as a misleading “Data source error”; Scenario Comparison raises an app exception. Setting an enabled scenario phase to months 10–12 and reducing the horizon to 6 resets it to 1–6 with no warning.

**Code inspection:** main UI percentage sliders allow only 0–50; scenario has the same cap and `MAX_PHASES=3`, versus the service's four phases, loss up to 100, and additions over 100. Scenario lacks explicit seasonal naive, calendar covariates, department/exclusion filters, and pasted data. These page-to-page gaps should be resolved or documented as intentional limitations.

**Executed:** a disabled phase dictionary with `percent='invalid'` still raises during normalization, despite README saying disabled phases are not validated. A frame with point 100/bounds 80–120 becomes point 50/bounds still 80–120 after applying 50% loss through the public scenario helper. The page does not currently draw these bounds, but API consumers receive inconsistent values.

**Implement:** shared phase editor and validator; fixed one-month ranges; four phases; separate loss/add numeric limits; explicit warning/confirmation of clamped ranges; skip disabled payload conversions; recompute scenario bounds from retained baseline bounds and adjustment factors. Use explicit factors rather than adjusted/original ratios, which fail at a zero baseline. Cache baseline by data/model settings so phase edits do not retrain it. Show the settings actually used and mark results stale when pending controls differ.

**Accept when:** both pages handle 1 and maximum months; 100% loss and >100% additions work; four phases and overlapping/no-op/disabled cases match Python; repeated application starts from baseline points and bounds; phase-only edits do not refit; horizon changes cannot silently move a phase to earlier months. Verify Excel deltas/FY sums from full-precision data before display rounding.

### P1 — uncertainty selection and provenance

**Executed:** ETS, Jan–Dec 2020 synthetic history, horizon 3, `uncertainty_method='auto'` returns `uncertainty_method='prophet'` and “Auto-selected Prophet interval” while the active source is conformal. Coverage was unavailable, triggering a generic Prophet fallback. See `service.py:478` and `service.py:951`.

**Code inspection:** explicit Prophet requests for classical models are overwritten before recording `requested_uncertainty_method`; separate uncertainty tuning searches coverage targets and replaces the user's requested target; calibration-generation exceptions lose their reason; final Prophet CV uses fixed 365-day horizon rather than the requested horizon. Chart-only RMSE fallback uses 1.96 regardless of target coverage and leaves service/export source as `none`.

**Implement:** immutable requested settings, explicit effective settings, model-aware method choices, fixed requested coverage by default, structured diagnostic failures, horizon-aware evaluation, and a canonical interval source/bounds representation. If approximate RMSE intervals remain, store them and label their assumptions and nominal level. Validate finite bounds, paired endpoints, and coverage range at the API boundary. Do not label missing metrics “Poor”; use “Unavailable”.

**Accept when:** classical models never select Prophet uncertainty; unavailable coverage does not invent a validated method; requested/effective settings and fallback reason remain visible in UI/history/exports; chart and downloaded bounds agree; a user-requested 90% target stays 90% unless an explicit target-search mode is selected.

### P1 — model selection at zero demand and tuning consistency

**Executed:** a 36-month all-zero series with `candidates=['seasonal_naive']` returns Prophet. Seasonal naive has valid point errors, but MAPE is unavailable; dropping NaN MAPE leaves no scored candidate and the generic fallback ignores the allowed candidate list. See `models.py:225`.

**Code inspection:** Prophet tuning occurs before model selection even when Auto candidates exclude Prophet. Tuning helpers fit Prophet without the requested calendar covariates; final selection/fitting can therefore use a different model specification. A fallback to short-history non-seasonal models is not surfaced as a distinct model variant.

**Implement:** candidate validation, finite score filtering, zero-safe selection using an explicitly chosen metric policy, candidate-respecting fallback, eligibility reasons, and explicit effective variants. Keep seasonal naive in default comparisons and explain any deliberate exclusion. Use the same covariates and evaluation dates for tuning, selection, and final diagnostics. Separate selected-model score from Prophet tuning score. For Auto, avoid Prophet tuning if Prophet is excluded; otherwise budget it within the comparison.

**Accept when:** all-zero, intermittent, constant, short, and long-horizon histories either produce a supported result or a clear actionable limit; excluded candidates never reappear; comparisons report comparable fold dates/counts; changing coverage alone does not silently change the evaluation target; a calendar-tuned Prophet is evaluated with the same regressors it will use in production.

## Added requirement — Fiscal Summary monthly, quarterly, and yearly rollups (P1)

Requested by the user after the initial review. Extend the existing Fiscal Summary tab rather than introducing a separate page.

**Behavior:**

- Provide one **Fiscal Month / Fiscal Quarter / Fiscal Year** selector. Both the table and chart must immediately show the same selected granularity across the available historical and forecast date range.
- Retain three distinct value series in every view: **Historical Visits**, **Original Forecast**, and **Adjusted Forecast**. A quarter/year containing both observed and future months must retain both contributions, without replacing actuals with forecasts or counting original and adjusted forecasts together.
- Preserve the Apr–Mar calendar: April is fiscal month 1, March is fiscal month 12; Q1 is Apr–Jun, Q2 Jul–Sep, Q3 Oct–Dec, and Q4 Jan–Mar. Use the fiscal ending year consistently. For example, April 2024 is `FY2025 M01 (Apr 2024)`, January 2025 is `FY2025 M10 (Jan 2025)`, and March 2025 is `FY2025 M12 (Mar 2025)`.
- Roll up from the same underlying monthly actuals and forecast values using full precision. Monthly rows sum exactly to quarter/year totals; round only for display. Chart bars must not include table subtotal rows as extra observations.
- Use chronological fiscal ordering and clear axis/period labels. Preserve the current yearly default. Switching views must reuse the stored forecast and must not retrain the model, alter phases, or add a run-history entry.
- Show the covered historical and forecast months for partial periods. Distinguish a genuinely zero count from an unavailable observation; avoid implying a full quarter/year when only part of it is covered.
- Export the selected view with its period labels, the three value series, and coverage metadata; use a filename that identifies monthly, quarterly, or yearly granularity. Table, chart, and export must reconcile.

**Implementation:** extend `aggregate_summary_by_period` in `src/visit_forecast/fiscal.py` with a monthly mode and shared fiscal period metadata. Extend the selector/rendering in `apps/streamlit/pages/1_Run_Forecast.py` and the labels/order in `apps/streamlit/components/charts.py`. Derive all three views from one aggregation path. Preserve current quarter/year APIs and existing FY subtotal behavior without double-counting totals in charts or downstream rollups. Update README examples to describe the three modes.

**Acceptance checks:** exercise all three modes using a known fixture spanning March/April, December/January, a quarter containing both actuals and forecasts, and partial opening/closing fiscal periods. Verify each value series independently: sums of fiscal monthly detail equal quarterly/yearly detail, table and chart contain matching period/value pairs, and downloads match the selected view. Test original versus adjusted forecasts with nonzero capacity changes, historical-only/future-only periods, missing observations versus real zeros, selector reruns without refitting, and exclusion of subtotal rows from chart sums. Keep point-error validation metrics month-based; fiscal rollups do not change their interpretation.

## Additional enhancements outside explicit README commitments

| Priority | Enhancement | Why / acceptance direction |
| --- | --- | --- |
| P1 | Non-negative finite forecast contract | Input non-negativity does not constrain Prophet/ARIMA/Theta outputs. Decide between a count-aware transformation/model and transparent clipping; reject NaN/inf and inverted intervals. Evaluate point/band behavior after capacity adjustments, including zero baseline and total loss. This is a risk from model behavior, not a reproduced negative output in this review. |
| P1 | Truthful evaluation labels | Tuning/model selection and reported backtests reuse historical evidence. Present selection scores separately from final holdout performance; show folds, dates, zero-actual exclusions, uncertainty sample size, and convergence warnings. Heuristic “Optimal” labels are not deployment approval. |
| P1 | Reproducible run manifest | Include data fingerprint, code/dependency versions, timestamp with timezone, training endpoint, chosen models/variants, settings, warnings, failures, phase definitions, and evaluation protocol in exports. Retain only the minimum identifying data necessary. |
| P2 | Shared data/settings across pages | Department/scenario pages currently duplicate ingestion and keep separate settings/results. Share a validated dataset and reusable controls, with explicit result provenance and stale-result indicators. |
| P2 | Department diagnostics and reconciliation | Expose each department's candidate comparison, history, intervals, failures, and FY summaries. Compare direct aggregate forecast with sum of departments. Define reconciliation before claiming coherence. Do not sum interval endpoints and call that a calibrated aggregate interval. |
| P2 | Run progress, budgets, cancellation | Default Auto + auto tuning can repeatedly fit Prophet; department loops run serially. Cache reusable forecasts/backtests by full configuration and data fingerprint, report progress, and bound concurrency. Measure performance before setting budgets. Never reuse cached results across incompatible settings. |
| P2 | History that can be trusted | Decide whether all three run types enter history; add run IDs, atomic persistence and visible write failures, schema versioning, and isolated storage. Session-only default is already a useful protection. History currently records metadata, not a complete replayable forecast archive. |
| P2 | Local launcher robustness | Verify app health and child-process identity before opening a browser; a pre-existing listener currently satisfies its port check. Localhost does not guarantee per-user isolation on a shared host. Keep persistence explicitly scoped to intended usage. |
| P2 | Domain-specific calendars and interventions | Business days currently means Monday–Friday without holidays. Add configurable holidays, clinic closures, working patterns, and optional external regressors only after validating them against real operational data. Document that percentage scenarios are assumptions, not estimated causal capacity responses. |
| P2 | Additional fiscal planning presentation | The requested three fiscal views and coverage labels are P1 above. Optionally add clearly labeled projected totals as historical + original forecast and historical + adjusted forecast; avoid counting the two forecast alternatives together or implying that partial-period sums cover a full year. |
| P2 | Usability and notebooks | Update Prophet-only Home copy; disable irrelevant tuning controls; provide examples for every public workflow; offer an optional notebook dependency group/kernel instructions. Add accessible charts/tables, progress/error guidance, and mobile verification. |

Authentication, organization-wide shared persistence, production hosting, new deep-learning models, and new data integrations are separate product decisions. They are not prerequisites for the README's local/sample-data workflow.

## Delivery plan

Implement as reviewable changes in this order. Each item includes its regression tests and documentation; do not defer all testing to the final change. Size indicates relative scope, not a time commitment.

| Order | Change | Scope | Dependencies | Completion gate |
| --- | --- | --- | --- | --- |
| 1 | Shared ingestion and validation | Medium | None | CSV/XLSX filename parity; explicit invalid-row diagnostics; department/month coverage; one validation report used by Data Check and forecasting. |
| 2 | Correct department totals and completeness | Medium | 1 | Common forecast calendar; unknown departments handled; per-date completeness/failures in chart and exports; no misleading complete totals. |
| 3 | Correct evaluation and uncertainty contract | Large | 1 | Chronological calibration/selection/evaluation boundaries, requested/effective settings, fixed coverage defaults, model-aware intervals, stored fallback provenance. |
| 4 | Robust Auto selection and efficient tuning | Medium–large | 3 | Zero-safe ranking; candidate-respecting fallbacks; consistent covariates/folds; disclosed variants; no unnecessary excluded-Prophet work. |
| 5 | Capacity and scenario parity | Medium | 3 for bounds; basic UI fixes can start earlier | Shared four-phase editor; 1-month runs; loss/add ranges; safe horizon changes; correct bounds; baseline reuse; stale-result notices. |
| 6 | Fiscal Summary monthly/quarterly/yearly views | Medium | 1 for validation/coverage semantics; presentation can proceed alongside 3–5 | One selector updates table/chart/exports; historical/original/adjusted series reconcile at all granularities; Apr–Mar labels/order; partial-period coverage; no model refit. |
| 7 | Public workflows and documentation | Medium | 1–6 | Public imports/examples, core CLI flags/outputs, agreed page parity, department diagnostics, updated Home/README, run manifests. |
| 8 | Operational quality and release validation | Medium | 1–7 | Performance baseline, robust history/launcher, populated UI/export CI, wheel-isolated API tests, representative approved dataset comparison. |

**Recommended first implementation slice:** CSV reader + shared validation contract, the one-month phase editor fix, and a regression that prevents incorrect complete department totals. Keep the statistical contract work visible as the next release gate; a cosmetic parity release alone would leave the largest planning risks unresolved.

## Verification matrix for the finished work

- **Data:** CSV and XLSX through API/CLI/UI; pasted input; duplicate rows; invalid dates/numbers/infinities; blanks; negatives; all-zero/intermittent demand; missing internal and boundary months; new/closed/unknown departments; consistent rules across pages.
- **Models:** each model and Auto; candidate exclusions; 12/13/24/25/26-month boundaries; 1/12/36-month horizons; covariates on/off; all candidates failing; invalid metrics; explicit fallback reasons. Compare the actual evaluated cutoff sets, not only requested backtest arguments.
- **Uncertainty:** supported methods per model; requested target preserved; absent coverage; full interval pairs; chronological target-date separation; finite-sample/short-history handling; zero/negative point behavior; actual exported fallback bounds; population versus sampled diagnostics clearly labeled.
- **Capacity/scenarios:** four phases; 100% loss; additions above 100%; overlaps; zero and disabled malformed phases; windows partly/fully beyond horizon; horizon reduction; one-month controls; zero baseline; repeated point/bound application; baseline fit count; exact monthly/FY export arithmetic.
- **Departments:** one/multiple departments; missing/unknown/requested/excluded names; unequal history start/end; internal gaps; one/all failures; per-date completeness; identical chart/CSV/XLSX status.
- **Fiscal Summary:** monthly/quarterly/yearly selector updates both table and chart; Apr–Mar fiscal month/quarter/year boundaries; mixed actual/forecast periods; independent historical/original/adjusted totals; partial coverage; zero versus unavailable values; matching exports; no subtotal double counting or forecast refitting.
- **UI/packaging:** real AppTest button runs with deterministic small samples; every populated result section, department/scenario workflows, control changes, failure handling, history isolation, and export content. Run package tests against a built wheel outside the checkout so source path injection cannot mask packaging errors. Use a small browser test for navigation/download/visual behavior that AppTest does not cover.
- **Release:** retain the existing suite and lockfile checks; compile/build; performance comparison on a documented reference case; approved de-identified operational dataset with a predeclared accuracy/coverage acceptance policy. Existing release notes already identify production-data and hosting validation as outstanding.

## Review evidence and limits

Executed focused probes used the existing locked Python environment. They reproduced CSV filename failure, disabled dictionary coercion, silently discarded invalid visits, missing department-month aggregation, misaligned complete department totals, stale scenario bounds, non-Prophet Auto uncertainty mislabeling, calibration/evaluation target overlap, zero-demand candidate fallback, one-month UI failures, and scenario range resets. A normal 36-month six-step ETS/Theta/seasonal-naive comparison used matching fold dates in the probe; differing folds were **not** established as a current defect and are included as a future invariant to test.

The 89-test result comes from the preceding setup on this same unchanged commit, not a new full-suite run during this review. New probes were targeted reproductions outside tracked application/test files. No real clinic data or deployment identity was available, so this review cannot establish production accuracy, reproduce a different deployed version, or assess user-specific hosting behavior.
