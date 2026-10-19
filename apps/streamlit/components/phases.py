"""One capacity editor for forecast and scenario workflows."""

import streamlit as st


def phase_editor(prefix: str, horizon: int, container=None) -> list[dict]:
    container = container or st
    phases = []
    for number in range(1, 5):
        key = f"{prefix}_{number}"
        with container.expander(f"Phase {number}", expanded=False):
            enabled = st.checkbox("Enabled", key=f"{key}_enabled")
            mode_label = st.selectbox(
                "Mode",
                ["Capacity loss", "Capacity add"],
                key=f"{key}_mode",
                disabled=not enabled,
            )
            mode = "loss" if mode_label == "Capacity loss" else "add"
            percent_key = f"{key}_percent"
            if mode == "loss" and st.session_state.get(percent_key, 0) > 100:
                st.session_state[percent_key] = 100.0
                st.warning("Loss was limited to 100%. Review this phase.")
            percent = st.number_input(
                "Percent (%)",
                min_value=0.0,
                max_value=100.0 if mode == "loss" else None,
                value=0.0,
                step=1.0,
                key=percent_key,
                disabled=not enabled,
                help="Loss can be 0–100%. Additions may exceed 100%.",
            )
            range_key = f"{key}_between"
            old = tuple(st.session_state.get(range_key, (1, min(12, horizon))))
            new = (max(1, min(int(old[0]), horizon)), max(1, min(int(old[1]), horizon)))
            if old != new:
                st.session_state[range_key] = new
                if enabled:
                    st.warning(
                        f"Phase {number} month range was clamped from {old} to {new}. Review before applying."
                    )
            if horizon == 1:
                st.caption("Applies to forecast month 1.")
                st.session_state[range_key] = (1, 1)
                month_range = (1, 1)
            else:
                if range_key not in st.session_state:
                    st.session_state[range_key] = new
                month_range = st.slider(
                    "Apply between forecast months",
                    1,
                    horizon,
                    key=range_key,
                    disabled=not enabled,
                )
            phases.append(
                dict(
                    enabled=enabled,
                    mode=mode,
                    percent=float(percent),
                    start_month=int(month_range[0]),
                    end_month=int(month_range[1]),
                )
            )
    return phases
