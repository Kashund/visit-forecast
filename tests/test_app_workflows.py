from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "apps" / "streamlit"


def test_one_month_forecast_and_all_fiscal_views(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    app = AppTest.from_file(str(APP / "pages" / "1_Run_Forecast.py")).run(timeout=30)
    app.radio(key="source_mode").set_value("Use synthetic sample")
    app.selectbox(key="model_choice").select("Seasonal naive (baseline)")
    app.radio(key="parameter_selection_mode").set_value("Manual")
    app.slider(key="months").set_value(1)
    app.run()
    assert not app.exception and not app.error
    app.button(key="run_btn").click().run(timeout=60)
    assert not app.exception and not app.error
    run_id = app.session_state["forecast_result"].run_manifest["run_id"]
    app.session_state["forecast_results_section"] = "🏛️ Fiscal Summary"
    app.run()
    for mode in ["Fiscal Month", "Fiscal Quarter", "Fiscal Year"]:
        app.radio(key="fiscal_summary_granularity").set_value(mode).run()
        assert not app.exception and not app.error
        assert len(app.get("plotly_chart")) == 1
        assert app.session_state["forecast_result"].run_manifest["run_id"] == run_id
    assert len(app.session_state["forecast_history"]) == 1


def test_scenario_four_phases_safe_ranges_and_baseline_reuse(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    app = AppTest.from_file(str(APP / "pages" / "4_Scenario_Comparison.py")).run()
    app.selectbox(key="cmp_model_choice").select("Seasonal naive (baseline)")
    app.checkbox(key="cmp_a_phase_1_enabled").check().run()
    app.slider(key="cmp_a_phase_1_between").set_value((10, 12)).run()
    app.slider(key="cmp_months").set_value(6).run()
    assert app.slider(key="cmp_a_phase_1_between").value == (6, 6)
    assert app.warning
    app.slider(key="cmp_months").set_value(1).run()
    assert not app.exception
    app.number_input(key="cmp_a_phase_1_percent").set_value(100.0)
    app.button(key="cmp_run_btn").click().run(timeout=60)
    assert not app.exception and not app.error
    baseline_id = app.session_state["cmp_result"].run_manifest["run_id"]
    assert len(app.session_state["cmp_scenarios"]["A"]) == 4
    app.number_input(key="cmp_a_phase_1_percent").set_value(50.0)
    app.button(key="cmp_run_btn").click().run(timeout=60)
    assert not app.exception
    assert app.session_state["cmp_result"].run_manifest["run_id"] == baseline_id
    assert len(app.session_state["forecast_history"]) == 2


def test_department_workflow_exports_and_diagnostics(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    app = AppTest.from_file(str(APP / "pages" / "3_Department_Forecasts.py")).run()
    app.radio(key="dept_source_mode").set_value("Use synthetic sample")
    app.selectbox(key="dept_model_choice").select("Seasonal naive (baseline)")
    app.slider(key="dept_months").set_value(1)
    app.button(key="dept_run_btn").click().run(timeout=60)
    assert not app.exception and not app.error
    bundle = app.session_state["dept_bundle"]
    assert len(bundle.results) == 3
    assert bundle.combined_future_df.aggregate_complete.eq(True).sum() == 1
    assert len(app.session_state["forecast_history"]) == 1


def test_forecast_cache_is_private_and_invalidates_on_settings(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    from state import cached_forecast
    from visit_forecast.io import generate_synthetic_data

    raw = generate_synthetic_data("2020-01-01", "2022-12-01")
    settings = dict(
        timeframe_start="2020-01-01",
        timeframe_end="2022-12-01",
        forecast_periods=1,
        model_choice="seasonal_naive",
        uncertainty_method="conformal",
    )
    session = {}
    a = cached_forecast(session, raw, settings)
    b = cached_forecast(session, raw, settings)
    assert b.run_manifest["cached_from_run"] == a.run_manifest["run_id"]
    assert b.run_manifest["run_id"] != a.run_manifest["run_id"]
    c = cached_forecast(session, raw, {**settings, "forecast_periods": 2})
    assert "cached_from_run" not in c.run_manifest
    other = cached_forecast({}, raw, settings)
    assert "cached_from_run" not in other.run_manifest
