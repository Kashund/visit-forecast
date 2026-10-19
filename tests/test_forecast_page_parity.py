from pathlib import Path
from io import StringIO
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "apps" / "streamlit"


def page(name):
    return AppTest.from_file(str(APP / "pages" / name)).run(timeout=30)


def configure_sample(app, prefix="", months=3):
    app.radio(key=f"{prefix}source_mode").set_value("Use synthetic sample")
    app.selectbox(key=f"{prefix}model_choice").select("Seasonal naive (baseline)")
    app.radio(key=f"{prefix}parameter_selection_mode").set_value("Manual")
    app.radio(key=f"{prefix}uncertainty_method").set_value("Conformal calibrated")
    app.slider(key=f"{prefix}months").set_value(months)
    app.slider(key=f"{prefix}target_coverage").set_value(0.85)
    app.run()
    app.checkbox(key=f"{prefix}phase_1_enabled").check().run()
    app.selectbox(key=f"{prefix}phase_1_mode").select("Capacity add").run()
    app.number_input(key=f"{prefix}phase_1_percent").set_value(25.0).run()
    return app


def test_sidebar_fields_options_and_defaults_match(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    single = page("1_Run_Forecast.py")
    departments = page("3_Department_Forecasts.py")
    for kind in [
        "radio",
        "selectbox",
        "slider",
        "checkbox",
        "number_input",
        "multiselect",
        "text_input",
        "button",
        "file_uploader",
    ]:
        left = [
            (
                x.label,
                list(x.options) if hasattr(x, "options") else None,
                x.value if hasattr(x, "value") else None,
            )
            for x in single.sidebar.get(kind)
        ]
        right = [
            (
                x.label,
                list(x.options) if hasattr(x, "options") else None,
                x.value if hasattr(x, "value") else None,
            )
            for x in departments.sidebar.get(kind)
        ]
        assert left == right, (kind, left, right)


def test_department_switch_updates_all_shared_results_without_refit(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    dept = configure_sample(page("3_Department_Forecasts.py"), "dept_")
    dept.button(key="dept_run_btn").click().run(timeout=60)
    assert not dept.exception and not dept.error
    bundle = dept.session_state["dept_bundle"]
    run_ids = {d: r.run_manifest["run_id"] for d, r in bundle.results.items()}
    assert bundle.results["Cardiology"].target_coverage == 0.85
    original = bundle.results["Cardiology"].future_forecast_df
    pd.testing.assert_series_equal(
        original.yhat_adjusted, original.yhat_original * 1.25, check_names=False
    )
    assert dept.selectbox(key="dept_output").value == "Cardiology"
    single = configure_sample(page("1_Run_Forecast.py"))
    single.selectbox(key="dept").select("Cardiology").run()
    single.button(key="run_btn").click().run(timeout=60)
    assert not single.exception and not single.error
    pd.testing.assert_frame_equal(
        single.session_state["forecast_result"].future_forecast_df, original
    )
    assert [(m.label, m.value) for m in single.metric] == [
        (m.label, m.value) for m in dept.metric
    ]
    dept.session_state["dept_forecast_results_section"] = "📅 Future Table"
    dept.run()
    for department in ["Oncology", "Neurology", "Cardiology"]:
        dept.selectbox(key="dept_output").select(department).run()
        assert not dept.exception and not dept.error
        expected = bundle.results[department].future_forecast_df
        future = [
            x.value for x in dept.dataframe if "scenario_summary" in x.value.columns
        ][0]
        assert future.Department.eq(department).all()
        pd.testing.assert_frame_equal(
            future.drop(columns=["scenario_summary", "Department"]), expected
        )
        assert len(dept.session_state["forecast_history"]) == 1
        assert {
            d: r.run_manifest["run_id"]
            for d, r in dept.session_state["dept_bundle"].results.items()
        } == run_ids
    dept.session_state["dept_forecast_results_section"] = "🏛️ Fiscal Summary"
    dept.run()
    for mode in ["Fiscal Month", "Fiscal Quarter", "Fiscal Year"]:
        dept.radio(key="dept_fiscal_summary_granularity").set_value(mode).run()
        assert not dept.exception and not dept.error
        tables = [
            x.value for x in dept.dataframe if "Historical_Visits" in x.value.columns
        ]
        assert tables


@pytest.mark.parametrize(
    "source_mode,label",
    [
        ("Upload file", "visits.xlsx"),
        ("Paste values (CSV)", "Pasted CSV"),
        ("Use synthetic sample", "Generated synthetic sample"),
    ],
)
def test_data_check_previews_the_last_run_source(monkeypatch, source_mode, label):
    monkeypatch.syspath_prepend(str(APP))
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-02-01", "2024-03-01"] * 2,
            "Visits": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "Department": ["A"] * 3 + ["B"] * 3,
        }
    )
    app = AppTest.from_file(str(APP / "pages" / "2_Data_Check.py"))
    app.session_state["last_run_data"] = dict(
        frame=raw,
        workflow="Run Forecast",
        source_mode=source_mode,
        label=label,
        stage="Last run",
        settings=dict(
            timeframe_start="2024-02-01",
            timeframe_end="2024-03-01",
            department="A",
            exclude_departments=[],
        ),
    )
    app.run()
    assert not app.exception and not app.error
    assert app.dataframe[0].value.Visits.tolist() == [2.0, 3.0]
    assert any(label in caption.value for caption in app.caption)
    app.radio(key="check_preview_rows").set_value("Raw input").run()
    pd.testing.assert_frame_equal(app.dataframe[0].value, raw)


def test_data_check_direct_paste_generated_and_bundled_samples(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    app = page("2_Data_Check.py")
    app.radio(key="check_source_mode").set_value("Paste values (CSV)").run()
    text = "Date,Visits,Department\n2024-01-01,10,A\n2024-02-01,20,A"
    app.text_area(key="check_pasted_csv").set_value(text).run()
    assert not app.exception and not app.error
    pd.testing.assert_frame_equal(app.dataframe[0].value, pd.read_csv(StringIO(text)))
    app.radio(key="check_source_mode").set_value("Use synthetic sample").run()
    assert not app.exception and not app.error
    assert len(app.dataframe[0].value) == 50
    assert set(app.dataframe[0].value.columns) == {"Date", "Visits", "Department"}
    app.radio(key="check_source_mode").set_value("Use bundled sample").run()
    for filename in ["synthetic_visits.csv", "synthetic_visits_small.csv"]:
        app.selectbox(key="check_sample_dataset").select(filename).run()
        expected = pd.read_csv(APP.parents[1] / "data" / "sample" / filename).head(50)
        pd.testing.assert_frame_equal(app.dataframe[0].value, expected)
        assert not app.exception and not app.error


def test_data_check_preserves_last_run_after_inputs_change(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    app = AppTest.from_file(str(APP / "Home.py")).run()
    app.switch_page("pages/1_Run_Forecast.py").run()
    configure_sample(app)
    app.button(key="run_btn").click().run(timeout=60)
    expected = app.session_state["last_run_data"]["frame"].copy()
    app.radio(key="source_mode").set_value("Paste values (CSV)").run()
    app.text_area(key="pasted_csv").set_value(
        "Date,Visits,Department\n2024-01-01,777,A\n2024-02-01,888,A"
    ).run()
    assert app.warning
    app.switch_page("pages/2_Data_Check.py").run()
    assert not app.exception and not app.error
    assert app.dataframe[0].value.Visits.tolist() == [777, 888]
    app.selectbox(key="check_snapshot").select("Last run").run()
    pd.testing.assert_frame_equal(app.dataframe[0].value, expected.head(50))
    app.selectbox(key="check_snapshot").select("Current inputs").run()
    assert app.dataframe[0].value.Visits.tolist() == [777, 888]


def test_department_controls_forward_tuning_and_prophet_panels(monkeypatch):
    from dataclasses import replace
    from visit_forecast import service

    monkeypatch.syspath_prepend(str(APP))
    backend = service.forecast_visits
    calls = []

    def fixture_forecast(**kwargs):
        calls.append(kwargs)
        result = backend(
            df=kwargs["df"],
            department=kwargs["department"],
            timeframe_start=kwargs["timeframe_start"],
            timeframe_end=kwargs["timeframe_end"],
            forecast_periods=kwargs["forecast_periods"],
            model_choice="seasonal_naive",
            uncertainty_method="conformal",
        )
        dates = result.future_forecast_df.ds
        native = pd.DataFrame(
            {
                "ds": dates,
                "yhat": [100.0] * len(dates),
                "yhat_lower": [90.0] * len(dates),
                "yhat_upper": [110.0] * len(dates),
                "trend": [100.0] * len(dates),
                "yearly": [1.0] * len(dates),
                "business_days": [2.0] * len(dates),
            }
        )
        cv = pd.DataFrame(
            {
                "horizon": pd.to_timedelta(["30 days", "60 days"]),
                "coverage": [0.85, 0.9],
                "mae": [1.0, 2.0],
                "rmse": [2.0, 3.0],
                "mape": [0.01, 0.02],
                "mdape": [0.01, 0.01],
            }
        )
        return replace(
            result,
            selected_model="prophet",
            requested_model="prophet",
            prophet_forecast_df=native,
            prophet_cv_metrics_df=cv,
            prophet_cv_raw_df=pd.DataFrame(
                {"cutoff": dates, "y": [100.0] * len(dates)}
            ),
        )

    monkeypatch.setattr(service, "forecast_visits", fixture_forecast)
    app = configure_sample(page("3_Department_Forecasts.py"), "dept_")
    app.selectbox(key="dept_model_choice").select("Prophet").run()
    app.radio(key="dept_parameter_selection_mode").set_value("Auto-tune")
    app.radio(key="dept_uncertainty_method").set_value("Auto-select").run()
    app.toggle(key="dept_joint_auto_tuning_enabled").set_value(True)
    app.button(key="dept_run_btn").click().run(timeout=60)
    assert not app.exception and not app.error
    assert len(calls) == 3
    for call in calls:
        assert call["model_choice"] == "prophet"
        assert call["tuning_mode"] == "auto"
        assert call["uncertainty_method"] == "auto"
        assert call["joint_auto_tuning_enabled"] is True
        assert call["target_coverage"] == 0.85
        assert call["capacity_phases"][0]["percent"] == 25.0
    for section in ["📊 Prophet CV", "🧩 Prophet Plots", "🧾 Run History"]:
        app.session_state["dept_forecast_results_section"] = section
        app.run()
        assert not app.exception and not app.error


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_upload_source_is_reusable_for_preview_and_forecast(monkeypatch, extension):
    from io import BytesIO

    monkeypatch.syspath_prepend(str(APP))
    from components.data_sources import select_data_source

    raw = pd.DataFrame({"Date": ["2024-01-01"], "Visits": [17], "Department": ["A"]})
    upload = BytesIO()
    upload.name = f"visits.{extension}"
    if extension == "csv":
        upload.write(raw.to_csv(index=False).encode())
    else:
        raw.to_excel(upload, index=False, engine="openpyxl")

    class UploadUI:
        def radio(self, *args, **kwargs):
            return "Upload file"

        def file_uploader(self, *args, **kwargs):
            return upload

    for _ in range(2):
        source = select_data_source("check_", UploadUI())
        pd.testing.assert_frame_equal(source.resolve("", ""), raw)
        assert source.label == upload.name


def test_data_check_keeps_invalid_current_inputs_visible(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    app = AppTest.from_file(str(APP / "Home.py")).run()
    app.switch_page("pages/1_Run_Forecast.py").run()
    app.radio(key="source_mode").set_value("Paste values (CSV)").run()
    app.text_area(key="pasted_csv").set_value(
        "Date,Visits,Department\n2024-01-01,bad,A\n2024-02-01,17,A"
    ).run()
    app.button(key="run_btn").click().run()
    assert app.error
    app.switch_page("pages/2_Data_Check.py").run()
    assert not app.exception
    assert app.dataframe[0].value.Visits.tolist() == ["bad", "17"]
    assert app.error


def test_data_check_can_preview_preexisting_shared_session_data(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    raw = pd.DataFrame({"Date": ["2024-01-01"], "Visits": [17], "Department": ["A"]})
    app = AppTest.from_file(str(APP / "pages" / "2_Data_Check.py"))
    app.session_state["shared_dataset"] = raw
    app.run()
    assert not app.exception and not app.error
    pd.testing.assert_frame_equal(app.dataframe[0].value, raw)
