from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "apps" / "streamlit"


def open_app(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    return AppTest.from_file(str(APP / "Home.py")).run(timeout=30)


def visit(app, page):
    app.switch_page(page).run(timeout=30)
    assert not app.exception
    return app


def baseline(app, prefix):
    app.selectbox(key=prefix + "model_choice").select("Seasonal naive (baseline)")
    app.slider(key=prefix + "months").set_value(3)
    app.run()


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_uploaded_dataset_survives_navigation_and_clear(monkeypatch, extension):
    from visit_forecast.io import generate_synthetic_data

    raw = generate_synthetic_data("2021-01-01", "2024-12-01")
    upload = BytesIO()
    upload.name = f"clinic.{extension}"
    if extension == "csv":
        upload.write(raw.to_csv(index=False).encode())
    else:
        raw.to_excel(upload, index=False)
    # AppTest has no file upload setter. Emulate a widget returning its file
    # once, then being discarded on navigation, just as in Streamlit itself.
    pending = [upload]

    def uploader(self, *args, **kwargs):
        if kwargs["key"].startswith("dept_uploader_") and pending:
            return pending.pop()
        return None

    monkeypatch.setattr(DeltaGenerator, "file_uploader", uploader)
    app = open_app(monkeypatch)
    visit(app, "pages/3_Department_Forecasts.py")
    expected = app.session_state["active_dataset"]["frame"].copy()
    assert app.session_state["active_dataset"]["label"] == upload.name
    baseline(app, "dept_")
    app.button(key="dept_run_btn").click().run(timeout=60)
    assert not app.exception and not app.error
    ids = {
        name: r.run_manifest["run_id"]
        for name, r in app.session_state["dept_bundle"].results.items()
    }
    app.selectbox(key="dept_output").select("Oncology").run()
    app.segmented_control(key="dept_forecast_results_section").set_value(
        "🏛️ Fiscal Summary"
    ).run()
    app.radio(key="dept_fiscal_summary_granularity").set_value("Fiscal Quarter").run()

    for page in [
        "pages/4_Scenario_Comparison.py",
        "pages/2_Data_Check.py",
        "pages/1_Run_Forecast.py",
        "Home.py",
    ]:
        visit(app, page)
        pd.testing.assert_frame_equal(
            app.session_state["active_dataset"]["frame"], expected
        )
        if "Scenario" in page:
            assert app.radio(key="cmp_source_mode").value == "Use shared dataset"
            assert app.text_input(key="cmp_start").value == "2021-01-01"
        if "Data_Check" in page:
            pd.testing.assert_frame_equal(app.dataframe[0].value, expected.head(50))

    visit(app, "pages/3_Department_Forecasts.py")
    assert not app.error and not app.warning
    assert app.selectbox(key="dept_output").value == "Oncology"
    assert app.slider(key="dept_months").value == 3
    assert app.radio(key="dept_fiscal_summary_granularity").value == "Fiscal Quarter"
    assert {
        name: r.run_manifest["run_id"]
        for name, r in app.session_state["dept_bundle"].results.items()
    } == ids
    assert len(app.session_state["forecast_history"]) == 1

    app.button(key="clear_session_data").click().run()
    for key in [
        "active_dataset",
        "shared_dataset",
        "current_input_data",
        "last_run_data",
        "forecast_data_snapshots",
        "forecast_cache",
    ]:
        assert key not in app.session_state
    assert app.get("plotly_chart")  # Completed result still visible after clear.
    assert len(app.session_state["forecast_history"]) == 1
    for page in [
        "pages/4_Scenario_Comparison.py",
        "pages/2_Data_Check.py",
        "pages/1_Run_Forecast.py",
        "Home.py",
        "pages/3_Department_Forecasts.py",
    ]:
        visit(app, page)
        assert "active_dataset" not in app.session_state
        assert "shared_dataset" not in app.session_state
    assert {
        name: r.run_manifest["run_id"]
        for name, r in app.session_state["dept_bundle"].results.items()
    } == ids


def test_pasted_data_is_shared_before_any_run_and_new_data_replaces_it(monkeypatch):
    app = open_app(monkeypatch)
    visit(app, "pages/2_Data_Check.py")
    app.radio(key="check_source_mode").set_value("Paste values (CSV)").run()
    first = "Date,Visits,Department\n2024-01-01,111,A\n2024-02-01,222,A"
    app.text_area(key="check_pasted_csv").set_value(first).run()
    expected = app.session_state["active_dataset"]["frame"].copy()
    visit(app, "pages/4_Scenario_Comparison.py")
    assert app.radio(key="cmp_source_mode").value == "Use shared dataset"
    pd.testing.assert_frame_equal(app.dataframe[0].value, expected)
    app.radio(key="cmp_source_mode").set_value("Paste values (CSV)").run()
    second = "Date,Visits,Department\n2025-01-01,333,B\n2025-02-01,444,B"
    app.text_area(key="cmp_pasted_csv").set_value(second).run()
    visit(app, "pages/2_Data_Check.py")
    assert app.dataframe[0].value.Visits.tolist() == [333, 444]
    visit(app, "pages/1_Run_Forecast.py")
    assert app.radio(key="source_mode").value == "Use shared dataset"
    assert app.selectbox(key="dept").options == ["All", "B"]
    assert app.text_input(key="start").value == "2025-01-01"
    app.button(key="clear_session_data").click().run()
    visit(app, "pages/4_Scenario_Comparison.py")
    assert app.radio(key="cmp_source_mode").value == "Upload file"
    app.radio(key="cmp_source_mode").set_value("Paste values (CSV)").run()
    assert app.text_area(key="cmp_pasted_csv").value == ""
    assert "active_dataset" not in app.session_state


def test_forecasts_and_scenario_assumptions_survive_until_successful_replacement(
    monkeypatch,
):
    app = open_app(monkeypatch)
    visit(app, "pages/1_Run_Forecast.py")
    app.radio(key="source_mode").set_value("Use synthetic sample").run()
    baseline(app, "")
    app.button(key="run_btn").click().run(timeout=60)
    original = app.session_state["forecast_result"]
    run_id = original.run_manifest["run_id"]
    app.segmented_control(key="forecast_results_section").set_value(
        "📅 Future Table"
    ).run()
    visit(app, "pages/4_Scenario_Comparison.py")
    baseline(app, "cmp_")
    app.checkbox(key="cmp_a_phase_1_enabled").check().run()
    app.selectbox(key="cmp_a_phase_1_mode").select("Capacity add").run()
    app.number_input(key="cmp_a_phase_1_percent").set_value(25.0)
    app.button(key="cmp_run_btn").click().run(timeout=60)
    assert not app.exception and not app.error
    comparison = app.dataframe[1].value.copy()  # data preview precedes results
    baseline_id = app.session_state["cmp_result"].run_manifest["run_id"]
    comparison_id = app.session_state["cmp_manifest"]["run_id"]
    visit(app, "pages/2_Data_Check.py")
    visit(app, "Home.py")
    visit(app, "pages/4_Scenario_Comparison.py")
    assert not app.error and not app.warning
    assert app.number_input(key="cmp_a_phase_1_percent").value == 25.0
    pd.testing.assert_frame_equal(app.dataframe[1].value, comparison)
    assert app.session_state["cmp_result"].run_manifest["run_id"] == baseline_id
    assert app.session_state["cmp_manifest"]["run_id"] == comparison_id
    assert len(app.session_state["forecast_history"]) == 2
    visit(app, "pages/1_Run_Forecast.py")
    assert (
        app.segmented_control(key="forecast_results_section").value == "📅 Future Table"
    )
    assert app.session_state["forecast_result"].run_manifest["run_id"] == run_id
    app.slider(key="months").set_value(6).run()
    assert app.session_state["forecast_result"].run_manifest["run_id"] == run_id
    app.text_input(key="start").set_value("2025-03-01")
    app.button(key="run_btn").click().run(timeout=60)
    assert app.error and not app.exception
    assert app.session_state["forecast_result"].run_manifest["run_id"] == run_id
    assert any("scenario_summary" in df.value.columns for df in app.dataframe)
    app.text_input(key="start").set_value("2020-04-01")
    app.button(key="run_btn").click().run(timeout=60)
    assert not app.error and not app.exception
    assert app.session_state["forecast_result"].run_manifest["run_id"] != run_id
    assert len(app.session_state["forecast_result"].future_forecast_df) == 6
    app.button(key="clear_history").click().run()
    assert app.session_state["forecast_result"] is not None
    app.run()
    assert app.metric


def test_active_data_and_forecasts_are_private_to_the_browser_session(monkeypatch):
    first = open_app(monkeypatch)
    visit(first, "pages/1_Run_Forecast.py")
    first.radio(key="source_mode").set_value("Use bundled sample").run()
    assert "active_dataset" in first.session_state
    other = open_app(monkeypatch)
    visit(other, "pages/1_Run_Forecast.py")
    assert "active_dataset" not in other.session_state
    assert "forecast_result" not in other.session_state
    assert other.radio(key="source_mode").value == "Upload file"
