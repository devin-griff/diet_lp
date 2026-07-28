# =============================================================================
# Diet LP Optimizer: a Streamlit tutorial app.
#
# This file builds an interactive web app around the classic Stigler-style
# diet problem: choose how much of each food to buy so that total cost is
# minimized while every nutrient requirement is met.
#
# It is a Linear Program (LP): variables are continuous, not binary:
#   minimize   sum_f  p_f * x_f                  (cost)
#   subject to sum_f  D_{f,n} * x_f >= r_n  for all nutrients n
#              x_f >= 0
#
# Library roadmap:
#   - streamlit : the UI framework. Each interaction reruns this script
#                  top-to-bottom; persistent values live in `st.session_state`.
#   - pyomo     : algebraic modeling: sets, params, vars, objective,
#                  constraints. Continuous (NonNegativeReals) variables here.
#   - HiGHS     : the LP solver, called via Pyomo's appsi_highs interface.
#                  Ships as a pip wheel (`highspy`).
#   - pandas    : DataFrame shape for Streamlit's data editor and Altair.
#   - altair    : grouped bars + horizontal "min requirement" markers.
#
# File roadmap:
#   1. Solver      : model definition + HiGHS log capture.
#   2. Constants   : defaults, nutrient labels, MAX_FOODS, slider cap.
#   3. State       : session_state init / reset, slider key helpers, user/
#                     canonical/widget-key mirroring callback.
#   4. Utilities   : DataFrame <-> internal-dict conversion, totals, cost.
#   5. LaTeX       : render the current instance as a formatted equation.
#   6. Tabs        : render_data_tab / render_formulation_tab /
#                     render_logs_tab / render_optimizer_tab.
#   7. Main        : page config, corner-logo CSS, header/caption, four
#                     tabs assembled at the module bottom.
# =============================================================================

import base64
import copy
import math
import threading
from pathlib import Path

import altair as alt
import pandas as pd
import pyomo.environ as pyo
import streamlit as st
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.opt import TerminationCondition


# ---------- Solver ----------
#
# Standard Pyomo LP. The only twist is `_solve_capturing`, which redirects
# HiGHS's solver output at the OS file-descriptor level so we can show it
# in the Logs tab.

def build_model(data):
    # ConcreteModel: components bound to data at construction time.
    m = pyo.ConcreteModel()

    # Two index sets: foods (F) and nutrients (N).
    m.FOOD = pyo.Set(initialize=data["foods"])
    m.NUTRIENTS = pyo.Set(initialize=data["nutrients"])

    # Parameters:
    #   needs[n]     = required minimum amount r_n of nutrient n
    #   content[f,n] = how much of nutrient n is in one unit of food f (D_{f,n})
    #   price[f]     = unit cost p_f of food f
    m.needs = pyo.Param(m.NUTRIENTS, initialize=data["needs"])
    m.content = pyo.Param(m.FOOD, m.NUTRIENTS, initialize=data["content"])
    m.price = pyo.Param(m.FOOD, initialize=data["price"])

    # Decision variable x_f: how much of each food to buy. Continuous and
    # non-negative: fractional servings are allowed.
    m.eaten = pyo.Var(m.FOOD, domain=pyo.NonNegativeReals)

    # One nutrient constraint per nutrient n: total nutrient delivered must
    # be at least the required amount. `rule=` builds an indexed constraint.
    def need_def(m, n):
        return sum(m.content[f, n] * m.eaten[f] for f in m.FOOD) >= m.needs[n]

    m.need_constraint = pyo.Constraint(m.NUTRIENTS, rule=need_def)

    # Objective: minimize total cost.
    m.cost = pyo.Objective(
        expr=sum(m.eaten[f] * m.price[f] for f in m.FOOD),
        sense=pyo.minimize,
    )
    return m


# One solve at a time per machine: every Streamlit session runs app.py in
# its own thread of this one process, and the solver-log capture redirects
# process-global stdout. Overlapping captures corrupt each other and fail
# both solves, so every solve serializes behind one process-wide lock.
# The lock must come from st.cache_resource: Streamlit re-executes this
# script per rerun in a fresh namespace, so a bare module-level Lock
# would be a new object every rerun and would serialize nothing.
@st.cache_resource(show_spinner=False)
def _solve_lock():
    return threading.Lock()


def _solve_capturing(m):
    """Run the solver and return (results, log_text). Captures HiGHS's
    stdout via Pyomo's capture_output (FD-level redirect on newer Pyomo,
    plain stdout capture on older). HiGHS via appsi_highs doesn't support
    a logfile= kwarg, so the FD-level capture is the only path."""
    log_text = ""
    try:
        with _solve_lock(), capture_output(capture_fd=True) as buf:
            solver = pyo.SolverFactory("appsi_highs")
            results = solver.solve(m, tee=True)
        log_text = buf.getvalue()
    except TypeError:
        # Older Pyomo without capture_fd: fall back to plain stdout capture.
        with _solve_lock(), capture_output() as buf:
            solver = pyo.SolverFactory("appsi_highs")
            results = solver.solve(m, tee=True)
        log_text = buf.getvalue()
    return results, log_text


def solve(data):
    # Top-level entrypoint used by the UI. Always returns a plain dict so the
    # caller can stash the result in session_state without holding on to a
    # live Pyomo model.

    # Empty problem: bail before constructing a model with no foods.
    if not data["foods"]:
        return {"status": "no_foods", "x": {}, "cost": None, "log": ""}

    m = build_model(data)

    try:
        results, log = _solve_capturing(m)
    except ApplicationError as e:
        # Pyomo raises ApplicationError when the solver isn't available.
        # HiGHS ships as a pip wheel via highspy, so this normally only
        # fires on a broken install.
        return {
            "status": "solver_missing",
            "message": (
                "HiGHS solver not available. Run `pip install highspy` "
                f"in your environment. ({e})"
            ),
            "x": {},
            "cost": None,
            "log": "",
        }

    # Translate Pyomo's TerminationCondition enum into a small set of stable
    # status strings the UI knows how to render.
    tc = results.solver.termination_condition
    if tc == TerminationCondition.optimal:
        # Pull numeric values out of the model before returning.
        x = {f: float(pyo.value(m.eaten[f])) for f in data["foods"]}
        cost = float(pyo.value(m.cost))
        return {"status": "optimal", "x": x, "cost": cost, "log": log}
    if tc in (
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
    ):
        return {"status": "infeasible", "x": {}, "cost": None, "log": log}
    if tc == TerminationCondition.unbounded:
        return {"status": "unbounded", "x": {}, "cost": None, "log": log}
    # Catch-all (solver error, time limit, etc.).
    return {"status": str(tc), "x": {}, "cost": None, "log": log}


# ---------- Constants ----------
#
# The four nutrients are baked into the schema (column ordering, slider
# cap math, etc.). NUTRIENT_LABELS maps the short codes to display names.

NUTRIENTS = ["P", "C", "F", "V"]
NUTRIENT_LABELS = {"P": "Protein", "C": "Carbs", "F": "Fat", "V": "Vitamins"}

# Hard caps used by the UI. MAX_FOODS truncates the data editor; SLIDER_CAP
# is the absolute upper bound for the per-food sliders even if the model
# could theoretically use more.
MAX_FOODS = 10
SLIDER_CAP = 50.0

# Default instance shown on first load and after the "Reset to defaults"
# button. `content` is keyed by (food, nutrient) tuples.
DEFAULT_DATA = {
    "foods": ["fruit", "vegetables", "meat", "bread", "pasta", "eggs"],
    "nutrients": NUTRIENTS,
    "needs": {"P": 20.0, "C": 40.0, "F": 15.0, "V": 20.0},
    "price": {
        "fruit": 2.0, "vegetables": 2.0, "meat": 6.0,
        "bread": 2.0, "pasta": 3.0, "eggs": 4.0,
    },
    "content": {
        ("fruit", "P"): 1.0, ("fruit", "C"): 4.0, ("fruit", "F"): 0.0, ("fruit", "V"): 5.0,
        ("vegetables", "P"): 2.0, ("vegetables", "C"): 3.0, ("vegetables", "F"): 0.0, ("vegetables", "V"): 6.0,
        ("meat", "P"): 8.0, ("meat", "C"): 0.0, ("meat", "F"): 5.0, ("meat", "V"): 1.0,
        ("bread", "P"): 2.0, ("bread", "C"): 6.0, ("bread", "F"): 1.0, ("bread", "V"): 1.0,
        ("pasta", "P"): 3.0, ("pasta", "C"): 8.0, ("pasta", "F"): 1.0, ("pasta", "V"): 0.0,
        ("eggs", "P"): 6.0, ("eggs", "C"): 1.0, ("eggs", "F"): 4.0, ("eggs", "V"): 2.0,
    },
}


# ---------- State ----------
#
# Streamlit re-executes the whole script on every interaction. Anything that
# must persist between runs lives in `st.session_state`. The keys we use:
#   - data:                the current problem instance (foods/needs/...)
#   - optimal:             the most recent solver result, or None
#   - _pending_reset:      one-shot flag to reset on the next run
#   - _bounds_optimal:     cached unconstrained-cost LP, sized per food, used
#                          to scale slider upper bounds
#   - slider_<food>:       canonical (non-widget) slider value for each food
#                         : what cost/chart computations read
#   - v_slider_<food>:     backing widget key for the user-side slider
#   - opt_v_<food>:        backing widget key for the optimal-side slider
#   - need_<nutrient>:     number_input value for each nutrient requirement
#   - data_editor:         backing key for the data editor widget
#   - apply_data_btn / reset_data_btn:  Data tab Apply / Reset buttons
#   - run_btn / set_opt_btn:            Optimizer tab Run / Set-to-optimal

def slider_key(food):
    # Stable, food-keyed CANONICAL identifier. Lives in session_state, drives
    # cost/chart computation, and is mutated by Apply changes (Data tab),
    # Set at Optimum, Reset to defaults, and init. NOT a widget key: the
    # actual st.slider widget uses `slider_widget_key(f)`.
    return f"slider_{food}"


def slider_widget_key(food):
    # Widget-owned key for the user-side st.slider. Distinct from
    # `slider_key` so the canonical can be freely modified by other code
    # paths (which a widget-owned key forbids after instantiation). The
    # widget reads its display value from session_state[widget_key], which
    # we pre-sync from the canonical on every rerun before the slider
    # renders.
    return f"v_{slider_key(food)}"


def _mirror_user_slider(canonical_key, widget_key):
    # on_change callback: push the widget's just-changed value back into
    # the canonical session_state slot. Runs before the script reruns, so
    # by the time render_optimizer_tab's pre-sync loop executes,
    # canonical and widget_key are already in agreement.
    st.session_state[canonical_key] = st.session_state[widget_key]


def init_state():
    # Idempotent init: only seed defaults the first time, otherwise the
    # user's edits would be wiped on every rerun.
    if "data" not in st.session_state:
        st.session_state.data = copy.deepcopy(DEFAULT_DATA)
    if "optimal" not in st.session_state:
        st.session_state.optimal = None
    # Silent LP solve used only to size the sliders (see
    # `slider_upper_bound`). Not shown to the user: the `optimal` key
    # above still drives the visible Optimal column / chart bars and
    # stays None until the user clicks Run Optimizer.
    if "_bounds_optimal" not in st.session_state:
        st.session_state._bounds_optimal = solve(st.session_state.data)
    # New foods added later need a slider key, but existing keys are
    # preserved so the user's selection survives.
    for f in st.session_state.data["foods"]:
        if slider_key(f) not in st.session_state:
            st.session_state[slider_key(f)] = 0.0
    # The reset button can't directly mutate widget-backed keys without
    # raising a Streamlit error, so it sets a flag and reruns. We then
    # apply the reset *before* widgets are instantiated this run.
    if st.session_state.pop("_pending_reset", False):
        apply_reset()


def apply_reset():
    # Restore the default instance and clear all widget-backed keys.
    st.session_state.data = copy.deepcopy(DEFAULT_DATA)
    st.session_state.optimal = None
    # Re-solve the silent bounds LP so slider ranges fit the defaults.
    st.session_state._bounds_optimal = solve(st.session_state.data)
    for f in DEFAULT_DATA["foods"]:
        st.session_state[slider_key(f)] = 0.0
    for n in NUTRIENTS:
        st.session_state[f"need_{n}"] = float(DEFAULT_DATA["needs"][n])


def current_slider_values():
    # Read each slider out of session_state. Missing keys default to 0.
    return {
        f: float(st.session_state.get(slider_key(f), 0.0))
        for f in st.session_state.data["foods"]
    }


# ---------- Utilities ----------
#
# Adapters between two data shapes:
#   - Internal dict shape (used by solver and most of the app).
#   - DataFrame shape (used by Streamlit's data editor widget).
# Plus small helpers for slider bounds, nutrient sums, and cost.

def data_to_df(data):
    # Internal -> DataFrame. One row per food, one column per nutrient + price.
    rows = []
    for f in data["foods"]:
        rows.append({
            "Food": f,
            "P": data["content"][(f, "P")],
            "C": data["content"][(f, "C")],
            "F": data["content"][(f, "F")],
            "V": data["content"][(f, "V")],
            "Price": data["price"][f],
        })
    return pd.DataFrame(rows)


def df_to_data(df, needs):
    # DataFrame -> internal. Normalizes whatever the user typed: strip
    # whitespace, drop blanks/duplicates, coerce numerics, clamp to >= 0.
    df = df.copy()
    df["Food"] = df["Food"].astype("string").str.strip()
    df = df.dropna(subset=["Food"])
    df = df[df["Food"] != ""]
    df = df.drop_duplicates(subset=["Food"], keep="first")
    for col in NUTRIENTS + ["Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    foods = df["Food"].tolist()
    price = {row.Food: float(row.Price) for row in df.itertuples()}
    # `content` is keyed by (food, nutrient) tuples to match Pyomo's
    # multi-indexed parameter style.
    content = {}
    for row in df.itertuples():
        for n in NUTRIENTS:
            content[(row.Food, n)] = float(getattr(row, n))
    return {
        "foods": foods,
        "nutrients": NUTRIENTS,
        "needs": {n: float(needs[n]) for n in NUTRIENTS},
        "price": price,
        "content": content,
    }


def slider_upper_bound(food, data):
    # Slider max for a food. Two layers:
    #
    # 1. Hard ceiling: the smallest amount that single-handedly satisfies
    #    any one nutrient: anything above this is physically wasteful.
    #    Capped by SLIDER_CAP and floored at 1.0 for visibility.
    bounds = []
    for n in NUTRIENTS:
        c = data["content"].get((food, n), 0.0)
        if c > 0:
            bounds.append(data["needs"][n] / c)
    if not bounds:
        hard_max = SLIDER_CAP
    else:
        hard_max = float(min(SLIDER_CAP, max(1.0, math.ceil(max(bounds)))))

    # 2. Data-aware refinement: when we have a silent LP solve (see
    #    init_state), give every food the same slider range: twice the
    #    ceiling of the largest LP optimum across all foods. Visually
    #    uniform, and the busiest food's optimum lands at ~50% on its
    #    slider with everyone else sitting lower. Capped by `hard_max`
    #    so editing nutrient needs into extreme territory doesn't blow
    #    up the slider.
    bo = st.session_state.get("_bounds_optimal")
    if bo and bo.get("status") == "optimal":
        x = bo.get("x", {})
        max_opt = max(x.values()) if x else 0.0
        suggested = 2 * math.ceil(max_opt)
        return float(min(hard_max, max(1.0, suggested)))
    return hard_max


def nutrient_totals(x, data):
    # For a given diet x, compute total amount of each nutrient delivered.
    return {
        n: sum(data["content"].get((f, n), 0.0) * float(x.get(f, 0.0)) for f in data["foods"])
        for n in NUTRIENTS
    }


def cost_of(x, data):
    # Total cost for a given diet x.
    return sum(data["price"][f] * float(x.get(f, 0.0)) for f in data["foods"])


# ---------- LaTeX (instance formulation) ----------
#
# The Formulation tab shows a static general formulation in math notation
# and a dynamic "instance" formulation that substitutes the user's current
# numbers into the equation. The helpers here build LaTeX source that
# `st.latex` then renders.

_LATEX_ESCAPE = [
    ("\\", r"\textbackslash "),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
]


def _latex_text(s):
    # Wrap a food name in \text{...} so it renders upright, with special
    # characters escaped.
    for raw, esc in _LATEX_ESCAPE:
        s = s.replace(raw, esc)
    return f"\\text{{{s}}}"


def _build_lhs(coefs, foods):
    # Build a "c1 * food1 + c2 * food2 + ..." style sum, skipping zero
    # coefficients and avoiding a leading "+".
    parts = []
    first = True
    for f in foods:
        c = float(coefs.get(f, 0.0))
        if c == 0:
            continue
        c_str = f"{c:g}"
        sep = "" if first else " + "
        parts.append(f"{sep}{c_str} \\, {_latex_text(f)}")
        first = False
    return "".join(parts) if parts else "0"


def build_instance_latex(data):
    # Assemble the full instance formulation as a LaTeX `aligned` block:
    # objective row, one constraint row per nutrient, then non-negativity.
    foods = data["foods"]
    obj = _build_lhs(data["price"], foods)
    rows = [r"\min \quad & " + obj + r" \\"]
    for i, n in enumerate(NUTRIENTS):
        coefs = {f: data["content"].get((f, n), 0.0) for f in foods}
        lhs = _build_lhs(coefs, foods)
        rhs = f"{data['needs'][n]:g}"
        label = NUTRIENT_LABELS[n]
        # First constraint row gets the "s.t." prefix; subsequent rows
        # just align under it.
        prefix = r"\text{s.t.} \quad & " if i == 0 else r"& "
        rows.append(f"{prefix}{lhs} \\ge {rhs} \\quad \\text{{({label})}} \\\\")
    bounds_lhs = ", ".join(_latex_text(f) for f in foods)
    rows.append(f"& {bounds_lhs} \\ge 0")
    body = r"\begin{aligned}" + "\n".join(rows) + r"\end{aligned}"
    # With many foods the line gets long; \small keeps it on screen.
    if len(foods) > 7:
        body = r"\small " + body
    return body


def colored_metric(label, value, color, align="left", suffix_html="", inset="0"):
    # st.metric doesn't support arbitrary value coloring, so we render a
    # metric-shaped block via raw HTML. Used to flag matching/mismatching
    # values (green if your cost equals the optimum, red otherwise).
    # `align` controls text-align so the metric can sit flush against
    # either edge of its column. `suffix_html` is appended inside the
    # value div after the number: used to drop a constraint-violation
    # ⚠ glyph next to Your cost when the user's diet misses a nutrient
    # minimum. `inset` adds horizontal padding on the side matching
    # `align` so the metric can be nudged inward from the column edge
    # (Your/Optimal cost under the chart use this to land away from the
    # column gutter).
    style_color = f"color: {color};" if color else ""
    pad_side = "left" if align == "left" else "right"
    st.markdown(
        f"<div style='margin: 0.25rem 0 1rem 0; padding-{pad_side}: {inset}; text-align: {align};'>"
        f"<div style='font-size: 0.875rem; color: rgba(49,51,63,0.6); margin-bottom: 0.25rem;'>{label}</div>"
        f"<div style='font-size: 2rem; font-weight: 600; line-height: 1; {style_color}'>{value}{suffix_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------- Tabs ----------
#
# One render_* function per tab. Data lets the user edit foods and nutrient
# requirements; Formulation shows the math; Logs shows HiGHS output;
# Optimizer is the main interactive view (defined last in this file).

def render_data_tab():
    # The Apply / Reset action row and pending-edits banner live at the top
    # of the tab so they're always visible. They depend on whether the
    # current widget state differs from `st.session_state.data`, which we
    # only know after rendering the input widgets below. `st.container()`
    # reserves the slot now; we fill it once `new_data` is computed.
    top_slot = st.container()

    # Top: four side-by-side number inputs for nutrient minimums (r_n).
    # Each is bound to a `need_<n>` session_state key so `apply_reset` can
    # seed it.
    st.subheader("Nutrient requirements")
    cols = st.columns(4)
    needs = st.session_state.data["needs"]
    new_needs = {}
    for col, n in zip(cols, NUTRIENTS):
        new_needs[n] = col.number_input(
            f"{NUTRIENT_LABELS[n]} ({n})",
            min_value=0.0,
            max_value=100.0,
            value=float(needs[n]),
            step=1.0,
            key=f"need_{n}",
        )

    # Editable foods table. `num_rows="dynamic"` lets the user add/delete
    # rows. Each column has a configured min so users can't enter negatives.
    st.subheader(f"Foods (max {MAX_FOODS})")
    df = data_to_df(st.session_state.data)
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Food": st.column_config.TextColumn("Food"),
            "P": st.column_config.NumberColumn("P (Protein)", min_value=0.0),
            "C": st.column_config.NumberColumn("C (Carbs)", min_value=0.0),
            "F": st.column_config.NumberColumn("F (Fat)", min_value=0.0),
            "V": st.column_config.NumberColumn("V (Vitamins)", min_value=0.0),
            "Price": st.column_config.NumberColumn("Price", min_value=0.0),
        },
        key="data_editor",
    )

    # Validate / report on the edited table.
    warnings = []
    if len(edited) > MAX_FOODS:
        warnings.append(f"Capped at {MAX_FOODS} foods; extra rows ignored.")
        edited = edited.head(MAX_FOODS)

    names = edited["Food"].dropna().astype("string").str.strip()
    if names.duplicated().any():
        warnings.append("Duplicate food names were dropped (kept the first).")

    new_data = df_to_data(edited, new_needs)
    has_pending = new_data != st.session_state.data

    # Fill the top slot now that we know whether edits are pending.
    with top_slot:
        apply_col, reset_col, _ = st.columns([1, 1, 4])
        with apply_col:
            apply_clicked = st.button(
                "Apply changes",
                type="primary" if has_pending else "secondary",
                width="stretch",
                disabled=not has_pending,
                key="apply_data_btn",
            )
        with reset_col:
            reset_clicked = st.button(
                "Reset to defaults",
                width="stretch",
                key="reset_data_btn",
            )
        if has_pending:
            st.info(
                "Edits pending. Click **Apply changes** to update the "
                "Optimizer tab."
            )

    for w in warnings:
        st.warning(w)

    if apply_clicked:
        # Commit buffered edits. Invalidate the visible LP result, re-solve
        # the silent bounds LP so slider ranges follow the new data, and
        # snap pre-existing canonical slider values to 0.1 so the value
        # badge reads cleanly. Writes target only the canonical key
        # (slider_<food>); the user slider's widget key
        # (slider_widget_key(f)) is owned by the widget and would error
        # if modified here. The pre-sync loop in render_optimizer_tab
        # propagates the new canonical into the widget next rerun.
        st.session_state.data = new_data
        st.session_state.optimal = None
        st.session_state._bounds_optimal = solve(new_data)
        for f in new_data["foods"]:
            if slider_key(f) not in st.session_state:
                st.session_state[slider_key(f)] = 0.0
            else:
                v = float(st.session_state[slider_key(f)])
                # Round to slider step (0.1) so the value bubble reads
                # cleanly post-data-edit.
                st.session_state[slider_key(f)] = round(v, 1)
        st.rerun()

    # Reset uses the deferred-flag pattern documented in `init_state`.
    if reset_clicked:
        st.session_state["_pending_reset"] = True
        st.rerun()


def render_formulation_tab():
    # Two sub-tabs (matching pinch-analysis / strip-packing / facility-location):
    # General (static reference math + pedagogical content) and Instance
    # (substitutes the user's current foods into the formulation).
    sub_general, sub_instance = st.tabs(["General", "Instance"])

    with sub_general:
        # 3-column grid: Sets/Parameters/Variables stacked on the left, the
        # centered objective + constraints in the middle, empty right padding
        # so the equation lands at the page midline rather than the right half.
        left, right, _ = st.columns([1, 1, 1])
        with left:
            st.markdown(
                "**Sets**  \n"
                r"$\mathcal{F} = \{\text{foods}\}$" "  \n"
                r"$\mathcal{N} = \{\text{nutrients}\}$"
            )
            st.markdown(
                "**Parameters**  \n"
                r"$p_i$ price for food option $i \in \mathcal{F}$" "  \n"
                r"$r_j$ nutrition requirement for nutrient $j \in \mathcal{N}$" "  \n"
                r"$D_{ij}$ nutrition info for food $i \in \mathcal{F}$ and nutrient $j \in \mathcal{N}$"
            )
            st.markdown(
                "**Variables**  \n"
                r"$x_i$ amount of food $i \in \mathcal{F}$ eaten or purchased"
            )
        with right:
            # Title + display math in one centered block. `$$...$$` inside
            # st.markdown (rather than st.latex) lets us wrap both in a single
            # text-align:center div so the equation lines up under the title.
            st.markdown(
                r"""<div style="text-align: center;">

**Objective and Constraints**

$$
\begin{gathered}
\min_x \sum_{i \in \mathcal{F}} x_i p_i \quad \text{(cost)} \\
\text{s.t.} \quad \sum_{i \in \mathcal{F}} D_{ij} x_i \ge r_j \quad \forall j \in \mathcal{N} \quad \text{(nutrient minimums)} \\
x_i \ge 0 \quad \forall i \in \mathcal{F} \quad \text{(lower bounds)}
\end{gathered}
$$

</div>""",
                unsafe_allow_html=True,
            )

        st.markdown("**Stigler's original computation**")
        st.markdown(
            "In 1945, economist George Stigler asked: what is the cheapest "
            "diet that meets every daily nutritional requirement? With nothing "
            "but paper, pencil, and \"a shrewd hunch about the products that "
            "should appear in the optimal diet\" (his words), he found a five "
            "food combination (wheat flour, evaporated milk, cabbage, spinach, "
            "dried navy beans) costing \\$39.93 per year in 1939 dollars."
        )
        st.markdown(
            "Two years later, George Dantzig invented the simplex method and "
            "the diet problem became one of the first LPs ever solved by "
            "computer. The true LP optimum came out to \\$39.69 per year. "
            "Stigler's hand calculation was off by 0.6%."
        )
        st.markdown("**Solution method**")
        st.markdown(
            "Solved as an LP via the simplex method, the algorithm Dantzig "
            "invented to solve this exact problem. HiGHS uses dual simplex "
            "by default, with an interior point alternative for very large "
            "instances. For a problem this size (a few foods times a few "
            "nutrients), the solve completes in microseconds. Most of what "
            "you see in the UI is Streamlit's render overhead."
        )
        st.markdown(
            "HiGHS is a modern open source LP/MIP solver from Edinburgh's "
            "ERGO group, distributed as a pip wheel via `highspy`."
        )

        st.markdown(
            "See the [companion Jupyter notebook]"
            "(https://github.com/devin-griff/diet-lp/blob/main/diet.ipynb) "
            "for the Pyomo implementation."
        )

        st.markdown("**References**")
        st.markdown(
            "[1] G. J. Stigler, \"The Cost of Subsistence,\" "
            "*Journal of Farm Economics*, vol. 27, no. 2, pp. 303–314, 1945. "
            "[JSTOR](https://www.jstor.org/stable/1231810)"
        )
        st.markdown(
            "[2] G. B. Dantzig, \"The Diet Problem,\" "
            "*Interfaces*, vol. 20, no. 4, pp. 43–47, 1990. "
            "[INFORMS](https://pubsonline.informs.org/doi/abs/10.1287/inte.20.4.43)"
        )
        st.markdown(
            "[3] Q. Huangfu and J. A. J. Hall, \"Parallelizing the dual "
            "revised simplex method,\" *Mathematical Programming Computation*, "
            "vol. 10, no. 1, pp. 119–142, 2018. "
            "[Springer](https://link.springer.com/article/10.1007/s12532-017-0130-5)"
        )
        st.markdown(
            "[4] M. L. Bynum, G. A. Hackebeil, W. E. Hart, C. D. Laird, "
            "B. L. Nicholson, J. D. Siirola, J.-P. Watson, and D. L. Woodruff, "
            "*Pyomo: Optimization Modeling in Python*, 3rd ed. "
            "Cham: Springer, 2021. "
            "[Springer](https://link.springer.com/book/10.1007/978-3-030-68928-5)"
        )

    with sub_instance:
        st.markdown(
            "The current instance, with values configured on the **Data** tab:"
        )
        data = st.session_state.data
        if not data["foods"]:
            st.info("Add at least one food on the Data tab.")
            return
        st.latex(build_instance_latex(data))


def render_logs_tab():
    # Shows whatever HiGHS printed during the last solve. The capture itself
    # happens in `_solve_capturing`; this tab just displays the result.
    optimal = st.session_state.optimal
    if not optimal:
        st.info("Run the optimizer to see solver logs.")
        return
    log = optimal.get("log", "") or ""
    if not log.strip():
        st.info("No solver output captured for the last run.")
        return
    st.code(log, language="text")


def render_optimizer_tab():
    # Layout (top to bottom):
    #   1. Action buttons (Run Optimizer, Set at Optimum)
    #   2. Status banner if the last run was not optimal
    #   3. Three columns: "Your diet" (interactive horizontal sliders) on
    #      the left, the nutrient bar chart in the middle, "Optimal diet"
    #      (read-only horizontal sliders) on the right. One slider per
    #      food in each side column, stacked top to bottom in food order.
    #   4. Cost metrics below each slider bank.
    data = st.session_state.data
    if not data["foods"]:
        st.info("Add at least one food on the Data tab.")
        return

    # Action row. Narrow on the left so the buttons stay compact.
    act1, act2, _ = st.columns([1, 1, 6])
    with act1:
        if st.button("Run Optimizer", width="stretch", key="run_btn"):
            result = solve(data)
            if result["status"] == "optimal":
                # Snap each x_f UP to the nearest slider step so the
                # displayed optimum is reachable via slider drag. The
                # LP solver returns continuous floats (e.g. bread =
                # 5.4545…) that don't fall on the 0.1 step grid;
                # rounding DOWN would lose feasibility on binding
                # constraints (≥-type), so we ceiling-round, which
                # only adds slack to the constraint side. The cost is
                # recomputed from the snapped values so "Optimal cost"
                # is the cost of the diet shown on the sliders: what
                # the user sees and what they can actually reach.
                #
                # PEDAGOGICAL CAVEAT: "Optimal cost" displayed on the
                # page is therefore the cost of the *step-feasible*
                # rounded solution, NOT the true LP minimum. On the
                # default data: ceiling-rounded cost = $24.0, true LP
                # minimum = $23.71: about a 1% premium. If you're
                # walking a student through the LP formulation, point
                # out that the math optimum is fractional (e.g.
                # vegetables = 1.6363, bread = 5.4545, eggs = 2.3864)
                # and the displayed value is the slider-reachable
                # rounding. The Logs tab still shows HiGHS's exact LP
                # output for the real numbers; the Formulation tab
                # documents the continuous LP. This trade was made so
                # "drag your sliders to match the displayed optimum"
                # works as advertised (Your cost meets Optimal cost,
                # no ⚠ violations). Without ceiling-rounding the user
                # could see the LP value 5.4545 displayed but only
                # ever drag to 5.4 or 5.5, neither of which matches
                # the LP cost AND many of which violate constraints.
                #
                # `zero_eps` treats sub-1e-6 LP values as exact zeros.
                # HiGHS sometimes returns tiny positive values like
                # 1e-12 for variables the LP doesn't want to use;
                # without the threshold, naive ceiling would push
                # those to 0.1 and falsely show "eat 0.1 of this food"
                # on the optimal slider.
                zero_eps = 1e-6
                rounded = {}
                for f, v in result["x"].items():
                    if v < zero_eps:
                        rounded[f] = 0.0
                    else:
                        ub_f = slider_upper_bound(f, data)
                        snapped = math.ceil(v / 0.1) * 0.1
                        rounded[f] = min(snapped, ub_f)
                result["x"] = rounded
                result["cost"] = cost_of(rounded, data)
            st.session_state.optimal = result
    optimal = st.session_state.optimal
    with act2:
        set_disabled = not (optimal and optimal["status"] == "optimal")
        if st.button(
            "Set at Optimum",
            width="stretch",
            disabled=set_disabled,
            key="set_opt_btn",
        ):
            # Copy the optimal x_f values into the user sliders. Each slider
            # is clamped to its own upper bound. We store the exact LP value
            # (no rounding) so Your cost matches Optimal cost exactly and
            # binding nutrient constraints don't get rounded below their
            # minimum (which would otherwise trigger the violation glyph
            # spuriously). The slider's step=0.1 may visually snap the
            # thumb to the nearest 0.1, but the canonical session_state
            # value remains exact.
            for f in data["foods"]:
                ub = slider_upper_bound(f, data)
                val = float(optimal["x"].get(f, 0.0))
                st.session_state[slider_key(f)] = max(0.0, min(val, ub))
            st.rerun()

    # Inline status messages for non optimal solver outcomes.
    if optimal:
        if optimal["status"] == "solver_missing":
            st.error(optimal["message"])
        elif optimal["status"] == "infeasible":
            st.error("Infeasible. No diet satisfies the requirements with this data.")
        elif optimal["status"] == "unbounded":
            st.error("Unbounded problem.")
        elif optimal["status"] not in ("optimal", "no_foods"):
            st.error(f"Solver returned: {optimal['status']}")

    # Clamp every canonical key before any st.slider call so the widget
    # never receives an out-of-range value when the user shrinks a
    # requirement (Streamlit errors if `value` is outside
    # [min_value, max_value]).
    for f in data["foods"]:
        ub = slider_upper_bound(f, data)
        key = slider_key(f)
        existing = float(st.session_state.get(key, 0.0))
        preserved = max(0.0, min(existing, ub))
        if existing != preserved:
            st.session_state[key] = preserved

    # Pre-sync canonical -> widget_key. The user-side st.slider owns
    # `slider_widget_key(f)`; once the slider instantiates this run, that
    # key cannot be modified by any other code path. So we copy from the
    # (freely-mutable) canonical into the widget-owned key now, BEFORE
    # the slider renders. This is the channel through which Set at
    # Optimum / Apply changes / Reset updates reach the rendered widget.
    for f in data["foods"]:
        wkey = slider_widget_key(f)
        st.session_state[wkey] = float(st.session_state.get(slider_key(f), 0.0))

    # Per-food hover tooltips: price per unit + nutrient density. Each
    # food's tooltip becomes the `content` of a CSS ::after pseudo-element
    # that fires on :hover of the slider's element-container: no JS,
    # no iframe. Built once outside the loop so user and optimal sides
    # share text.
    nutrient_label_short = {"P": "Prot", "C": "Carb", "F": "Fat", "V": "Vit"}
    user_tooltips = {}
    for f in data["foods"]:
        parts = [f"${data['price'][f]:g}"] + [
            f"{nutrient_label_short.get(n, n)} {data['content'][(f, n)]:g}"
            for n in data["nutrients"]
        ]
        user_tooltips[f] = "  ·  ".join(parts) + "  per unit"

    # Per-food tooltip content rules. Streamlit stamps
    # `st-key-<widget_key>` as a class on every widget's element-
    # container, which gives us a food-stable hook for CSS without any
    # JS injection. Tooltips fire on the USER side only: the optimal
    # side is read-only and already labelled, so hover info there would
    # be redundant.
    tooltip_rules = []
    for f in data["foods"]:
        tooltip_rules.append(
            f'[class~="st-key-{slider_widget_key(f)}"]:hover::after '
            f'{{ content: "{user_tooltips[f]}"; }}'
        )
    tooltip_content_css = "\n".join(tooltip_rules)

    # Optimal sliders use native BaseWeb rendering (active fill segment
    # painted by the slider widget itself, synced with thumb position by
    # design). To make them read-only we render with disabled=False but
    # block interaction via CSS `pointer-events: none`, plus the
    # pre-sync loop here that pins `session_state[widget_key] = LP_value`
    # each rerun so any accidental keyboard nudge bounces back to the LP
    # value on the next render.
    #
    # We then override BaseWeb's blue gradient with an amber gradient
    # per-food, using the same `value/max %` stop position BaseWeb uses
    # (so the amber/gray boundary lands exactly where the thumb is,
    # the same way the blue/gray boundary does on the user side). The
    # gradient stop is set in the *same Streamlit rerun* that BaseWeb
    # places the thumb at its new position, so both arrive on the same
    # paint frame: no "fill leads thumb" lag like the disabled-slider
    # workaround had.
    solved = bool(optimal and optimal["status"] == "optimal")
    opt_x = optimal["x"] if solved else {}
    opt_fill_rules = []
    for f in data["foods"]:
        ub = slider_upper_bound(f, data)
        if solved and ub > 0:
            val = max(0.0, min(float(opt_x.get(f, 0.0)), ub))
            pct = val / ub * 100
        else:
            val = 0.0
            pct = 0.0
        wkey = f"opt_v_{f}"
        st.session_state[wkey] = val
        # Exact class match: widget key is `opt_v_<food>` with no
        # trailing value, so the class on the element-container is
        # exactly `st-key-opt_v_<food>`. Using `:has()` on the
        # element-container picks the right one without risking
        # food-name-prefix collisions (e.g. "fruit" vs "fruitcake")
        # that a substring `[class*=...]` selector could hit.
        opt_fill_rules.append(
            f'[data-testid="stElementContainer"].st-key-opt_v_{f} '
            f'[data-baseweb="slider"] > div > div > div:nth-child(2) '
            f'{{ background-image: linear-gradient(to right, '
            f'#E69F00 0%, #E69F00 {pct:.4f}%, '
            f'rgba(151, 166, 195, 0.25) {pct:.4f}%, '
            f'rgba(151, 166, 195, 0.25) 100%) !important; }}'
        )
    opt_fill_css = "\n".join(opt_fill_rules)

    st.markdown(
        f"""
        <style>
        /* The native st.slider value bubble (floats above each thumb,
         * follows the thumb position) is the value indicator: same
         * pattern as quad-tank. Default Streamlit behavior, no override
         * needed here. */

        /* Kill BaseWeb's default `transition: all` on every slider
         * thumb (both user and optimal). The thumb's inline `transform`
         * animates between value updates via that transition (~200ms
         * ease), while the active-fill gradient and the value bubble
         * snap to their new positions on the same React commit. That
         * mismatch is the "bar leads thumb" effect right after
         * Set at Optimum / Run Optimizer. With transition:none the
         * thumb snaps to its new transform in the same frame as the
         * gradient. User-side drag still feels smooth: drag tracking
         * is mouse-driven, not transition-driven. */
        [data-testid="stSlider"] [role="slider"] {{
            transition: none !important;
        }}

        /* Optimal column: read-only sliders that visually match the
         * "Optimal" amber series in the Constraints chart.
         *
         * Strategy:
         *   - `disabled=False` on the widget so BaseWeb paints the
         *     active-fill gradient + positions the thumb in the SAME
         *     React render cycle (synced by design: no "fill leads
         *     thumb" lag).
         *   - Per-food CSS rules below override BaseWeb's blue gradient
         *     with the Wong amber color, keeping BaseWeb's percentage
         *     stop (so the amber/gray boundary lands under the thumb,
         *     same as the user side's blue/gray boundary).
         *   - Thumb recolored to amber.
         *   - `pointer-events: none` blocks mouse interaction; the
         *     pre-sync loop above pins session_state to the LP value
         *     so any keyboard focus + arrow press snaps back. */
        /* Collapse the invisible marker element-container. The marker
         * div sits inside an stElementContainer that has its own
         * top/bottom padding from Streamlit's vertical block layout,
         * which would push the optimal-side sliders ~14px below their
         * user-side counterparts. `display: none` on the wrapper
         * removes its layout box; the marker child is still in the DOM
         * so the column-level `:has(.optimal-col-marker)` selectors
         * keep matching. */
        [data-testid="stElementContainer"]:has(.optimal-col-marker) {{
            display: none;
        }}
        [data-testid="stColumn"]:has(.optimal-col-marker) [data-testid="stSlider"] {{
            pointer-events: none;
        }}
        [data-testid="stColumn"]:has(.optimal-col-marker)
            [data-testid="stSlider"] [role="slider"] {{
            background-color: #E69F00 !important;
        }}

        /* Tighten the vertical gap between sliders on both sides. Each
         * st.slider's element-container gets a small negative top margin
         * so the food label tucks closer to the previous slider's track.
         * Targets only the slider element-containers (via the st-key
         * class hook) so the surrounding rows (action buttons, cost
         * metrics, column headers) stay at default spacing. */
        [class*="st-key-v_slider_"],
        [class*="st-key-opt_v_"] {{
            margin-top: -0.5rem;
            position: relative;
        }}

        /* Hover tooltip on the constraint-violation icon next to "Your
         * cost". Exactly matches the Vega-tooltip styling below so the
         * "Constraint violated" text bubble looks identical whether the
         * user hovers the cost-icon ⚠ or a chart-mark ⚠. */
        .diet-violation-icon {{
            position: relative;
            display: inline-block;
        }}
        .diet-violation-icon:hover::after {{
            content: attr(data-violation-tooltip);
            position: absolute;
            /* Position the tooltip to the right of (and slightly below)
             * the icon: same relative location Vega-tooltip uses when
             * hovering chart marks (a small offset away from the cursor),
             * instead of dropping straight under the icon where the
             * cursor itself sits. */
            top: 100%;
            left: 100%;
            margin-left: 0.5rem;
            background: #000;
            color: #fff;
            padding: 0.5rem 0.75rem;
            border-radius: 4px;
            font-size: 0.75rem;
            /* Pin font metrics so width/height match Vega-tooltip's
             * "Constraint violated" bubble exactly. The cost-icon
             * pseudo otherwise inherits font-weight: 600 + line-height:1
             * from the parent cost-number div. */
            font-family: inherit;
            font-weight: 400;
            line-height: 1.2;
            white-space: nowrap;
            z-index: 1000;
            pointer-events: none;
        }}

        /* Vega-tooltip element added to <body> by altair_chart's hover
         * plugin. Default styling is white-bg / dark-text / 6px 12px /
         * 8px radius / 1px border: completely different from the
         * cost-icon tooltip. Overriding both color and box metrics to
         * match (.diet-violation-icon:hover::after above) so a hover on
         * any ⚠ in the app: cost-side or chart-side: produces a
         * visually identical popup. */
        #vg-tooltip-element.vg-tooltip {{
            background: #000 !important;
            color: #fff !important;
            border: none !important;
            padding: 0.5rem 0.75rem !important;
            border-radius: 4px !important;
            font-size: 0.75rem !important;
            /* Match the cost-icon tooltip's font metrics so text width
             * and box height are identical between the two. Vega-tooltip
             * otherwise uses its own font stack and may differ. */
            font-family: inherit !important;
            font-weight: 400 !important;
            line-height: 1.2 !important;
            box-shadow: none !important;
        }}
        /* Vega-tooltip renders every tooltip as a <table> with two columns:
         * `td.key` (the field name) and `td.value`. Our violation marks
         * pass title=" " for `status` to suppress the field-name label;
         * the resulting empty `td.key` still occupies width (~25px), so
         * the Vega tooltip bubble ends up wider than the cost-icon's
         * plain-text bubble. Collapsing the .key column on whitespace-
         * only content would be ideal but CSS can't match text content,
         * so we collapse it globally: the chart's bar tooltips lose
         * their field-name labels but still show the values (You / 0.00
         * / Protein), which remain readable in context. */
        #vg-tooltip-element.vg-tooltip td.key {{
            display: none !important;
        }}
        #vg-tooltip-element.vg-tooltip table {{
            border-spacing: 0 !important;
        }}
        #vg-tooltip-element.vg-tooltip td,
        #vg-tooltip-element.vg-tooltip th {{
            color: #fff !important;
            padding: 0 !important;
        }}

        /* Hover tooltip on the user-side slider's element-container.
         * Replaces the "?" help icon. The content (price + nutrient
         * density) is set per-food by the rules emitted below. */
        [class*="st-key-v_slider_"]:hover::after {{
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(17, 24, 39, 0.92);
            color: #f9fafb;
            padding: 0.35rem 0.55rem;
            border-radius: 4px;
            font-size: 0.75rem;
            white-space: pre-line;
            width: max-content;
            z-index: 1000;
            pointer-events: none;
            margin-top: 0.25rem;
        }}
        {tooltip_content_css}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Three column body: Your diet (left), nutrient chart (middle), Optimal
    # diet (right). Sliders are now horizontal and stacked vertically inside
    # each side column, in food order top to bottom.
    your_col, chart_col, opt_col = st.columns([4, 4, 4])

    with your_col:
        st.markdown("**Your diet**")
        for f in data["foods"]:
            ub = slider_upper_bound(f, data)
            canonical = slider_key(f)
            widget_key = slider_widget_key(f)
            # The slider reads its display value from
            # `session_state[widget_key]`, which the pre-sync loop above
            # already populated from the canonical. On user drag,
            # Streamlit writes the new value to widget_key and fires the
            # on_change callback, which mirrors widget_key back into the
            # canonical so other paths (Apply changes, cost computation,
            # chart bars) see it. max_value updates from a data edit are
            # picked up naturally each rerun.
            st.slider(
                label=f,
                key=widget_key,
                min_value=0.0,
                max_value=float(ub),
                step=0.1,
                format="%.1f",
                on_change=_mirror_user_slider,
                args=(canonical, widget_key),
            )

    # Read the user sliders and compute the user cost AFTER the left side
    # widgets have written their values back into session_state above.
    slider_vals = current_slider_values()
    user_cost = cost_of(slider_vals, data)

    # Pre-compute nutrient totals + violation flag here so the cost
    # metric below can render a ⚠ glyph next to the cost number when
    # any nutrient minimum is missed.
    #
    # `violation_eps = 1e-6` is sized purely to absorb solver float
    # noise on the LP solution itself (19.99999998 vs 20.0): NOT to
    # cover slider step rounding. With step=0.1, a user rounding their
    # picks to the displayed LP value can still be ~0.1 short of a
    # binding constraint, which is a genuine constraint violation the
    # user should see (the LP minimum is at the constraint boundary;
    # rounding down loses feasibility). The ⚠ correctly fires in that
    # case, telling the user to bump a slider up; widening the eps to
    # mask it would hide real infeasibility.
    user_totals = nutrient_totals(slider_vals, data)
    violation_eps = 1e-6
    has_violation = any(
        user_totals[n] < data["needs"][n] - violation_eps for n in NUTRIENTS
    )

    # Decide cost-metric colors once so both columns paint consistently.
    # Green when Your cost meets or beats the LP optimum. Beating it is
    # only possible while infeasible (some nutrient minimum is violated)
    # since the LP is a minimization at the constraint boundary; the
    # chart's ⚠ glyph already flags infeasibility separately, so the
    # cost indicator and the feasibility indicator stay orthogonal.
    #
    # Both costs are displayed and compared at 1-decimal precision -
    # the same granularity as the slider step (0.1). This lets the user
    # drag sliders to displayed LP values and see Your cost match
    # Optimal cost, instead of being painted red over a sub-dime
    # rounding gap that the cost numbers also gloss over.
    if optimal and optimal["status"] == "optimal":
        opt_cost = float(optimal["cost"])
        your_color = (
            "#16a34a" if round(user_cost, 1) <= round(opt_cost, 1)
            else "#dc2626"
        )
        opt_color = "#16a34a"
        opt_value = f"{opt_cost:.1f}"
    else:
        your_color = None
        opt_color = None
        opt_value = "-"

    # Your cost / Optimal cost get rendered below the chart in the
    # middle column (see chart_col block below), so they sit under a
    # fixed-height visual and don't drift down as more foods are added
    # to the slider banks on either side. The ⚠ glyph is appended to
    # Your cost when any nutrient minimum is missed: mirrors the
    # chart's per-nutrient ⚠ marks at a glance-friendly summary scale.
    violation_icon_html = (
        '<span class="diet-violation-icon" '
        'data-violation-tooltip="Constraint violated" '
        'style="color: #dc2626; margin-left: 0.5rem; font-size: 1.5rem; '
        'cursor: default; vertical-align: baseline;">⚠</span>'
        if has_violation else ''
    )

    with chart_col:
        # Middle column: grouped bar chart of nutrient totals (You vs
        # Optimal) with red minimum requirement rules. Bars are slim so
        # the four nutrient groups fit in the narrow middle column.
        st.markdown("**Constraints**")

        # user_totals was pre-computed earlier (for the cost-metric
        # violation glyph). Reusing here.
        rows = [
            {"nutrient": NUTRIENT_LABELS[n], "source": "You", "value": user_totals[n]}
            for n in NUTRIENTS
        ]
        if optimal and optimal["status"] == "optimal":
            opt_totals = nutrient_totals(optimal["x"], data)
            rows.extend(
                {"nutrient": NUTRIENT_LABELS[n], "source": "Optimal", "value": opt_totals[n]}
                for n in NUTRIENTS
            )

        chart_df = pd.DataFrame(rows)
        nutrient_order = [NUTRIENT_LABELS[n] for n in NUTRIENTS]

        # Tight You/Optimal pairs inside each nutrient slot:
        # - `paddingInner=0` on the xOffset scale removes the gap between
        #   the two source sub-bands within a nutrient slot.
        # - Bumping `mark_bar(size=20)` so each bar takes up ~all of its
        #   half-slot; the pair effectively touches.
        # - Bumping `paddingInner=0.5` on the x axis preserves space
        #   between adjacent nutrient groups now that the pair itself is
        #   wider.
        bars = (
            alt.Chart(chart_df)
            .mark_bar(size=20)
            .encode(
                x=alt.X(
                    "nutrient:N",
                    sort=nutrient_order,
                    title=None,
                    scale=alt.Scale(paddingInner=0.5),
                ),
                xOffset=alt.XOffset(
                    "source:N",
                    sort=["You", "Optimal"],
                    scale=alt.Scale(paddingInner=0),
                ),
                y=alt.Y("value:Q", title="Total nutrient"),
                color=alt.Color(
                    "source:N",
                    scale=alt.Scale(domain=["You", "Optimal"], range=["#0072B2", "#E69F00"]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("nutrient:N"),
                    alt.Tooltip("source:N"),
                    alt.Tooltip("value:Q", format=".2f"),
                ],
            )
        )
        # Two endpoints per nutrient at pixel offsets that extend slightly
        # past the bar pair: each bar is 20 px wide and they touch, so the
        # pair spans -20 .. +20 px from the band center. We endpoint the
        # rule at -25 / +25 so it overhangs 5 px on each side. Using a
        # quantitative xOffset (`:Q`) feeds those numbers as raw pixel
        # offsets rather than going through a band scale.
        line_df = pd.DataFrame(
            [
                {"nutrient": NUTRIENT_LABELS[n], "value": data["needs"][n],
                 "kind": "Min requirement", "x_off": dx}
                for n in NUTRIENTS for dx in (-25, 25)
            ]
        )
        rules = (
            alt.Chart(line_df)
            .mark_line(strokeWidth=2, strokeDash=[3, 3], strokeCap="butt")
            .encode(
                x=alt.X("nutrient:N", sort=nutrient_order),
                xOffset=alt.XOffset("x_off:Q"),
                y="value:Q",
                detail="nutrient:N",
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(domain=["Min requirement"], range=["#dc2626"]),
                    legend=alt.Legend(
                        title=None,
                        symbolType="stroke",
                        symbolStrokeWidth=2,
                        symbolDash=[3, 3],
                        orient="top",
                    ),
                ),
                tooltip=[alt.Tooltip("value:Q", title="Min requirement")],
            )
        )
        # Layer 3 (conditional): warning glyph above any "You" bar whose
        # nutrient total falls below the min requirement. Same pattern as
        # the knapsack weight-limit ⚠, in the same red (#dc2626) as the
        # dotted min-requirement rules: the two read as a "constraint /
        # you violated it" pair. Hover shows the shortfall. `violation_eps`
        # is the same epsilon (1e-6) defined earlier near the cost-metric
        # violation flag, absorbing float-precision noise so the LP
        # solution itself doesn't trigger the glyph spuriously.
        violation_rows = [
            {
                "nutrient": NUTRIENT_LABELS[n],
                "value": user_totals[n],
                "source": "You",
                "deficit": data["needs"][n] - user_totals[n],
                "status": "Constraint violated",
            }
            for n in NUTRIENTS
            if user_totals[n] < data["needs"][n] - violation_eps
        ]
        violation_marks = None
        if violation_rows:
            violation_df = pd.DataFrame(violation_rows)
            violation_marks = (
                alt.Chart(violation_df)
                .mark_text(
                    text="⚠",
                    fontSize=22,
                    color="#dc2626",
                    dy=-12,
                    baseline="bottom",
                    fontWeight="bold",
                )
                .encode(
                    x=alt.X("nutrient:N", sort=nutrient_order),
                    xOffset=alt.XOffset(
                        "source:N",
                        sort=["You", "Optimal"],
                        scale=alt.Scale(paddingInner=0),
                    ),
                    y="value:Q",
                    # Every violation icon in the app: the cost-metric
                    # ⚠ and these per-nutrient chart marks: shows the
                    # same "Constraint violated" tooltip on hover, with
                    # no extra fields. The `title=" "` (single space)
                    # hides Vega-Lite's default field-name column so
                    # only the value text is visible.
                    tooltip=[alt.Tooltip("status:N", title=" ")],
                )
            )

        # Fixed chart height of 400px. The default instance has 6 foods,
        # where each native st.slider takes ~70px (label + track + value
        # indicator + margin), so 6 × ~70 ≈ 420 spans roughly the same
        # vertical extent as the slider stack. Foods are user-editable
        # (up to MAX_FOODS), so for non-default instances the chart and
        # slider stack heights may diverge.
        if violation_marks is not None:
            chart = (bars + rules + violation_marks).resolve_scale(color="independent").properties(height=400)
        else:
            chart = (bars + rules).resolve_scale(color="independent").properties(height=400)
        st.altair_chart(chart, width="stretch")

        # Cost metrics anchored below the chart (fixed-height block)
        # rather than at the bottom of the slider columns. This keeps
        # them at the same vertical position regardless of how many
        # foods the user adds on the side columns. Left half holds
        # "Your cost" (left-aligned with the chart's left edge); right
        # half holds "Optimal cost" (right-aligned to the chart's
        # right edge).
        cost_left, cost_right = st.columns([1, 1])
        with cost_left:
            colored_metric(
                "Your cost", f"{user_cost:.1f}", your_color,
                align="left", suffix_html=violation_icon_html,
                inset="3rem",
            )
        with cost_right:
            colored_metric(
                "Optimal cost", opt_value, opt_color,
                align="right", inset="1.5rem",
            )

    with opt_col:
        # Right column: always render one slider per food, in the same
        # food order as the user side. Pre-solve, every slider sits at 0;
        # post-solve, each shows the LP's x_f. The widget is rendered
        # without `disabled=True` (BaseWeb's disabled gradient lags the
        # thumb on value updates); instead the CSS injected above sets
        # `pointer-events: none` to block mouse interaction and the pre-
        # sync loop pins session_state to the LP value so any focused
        # keyboard arrow snaps back. The same CSS repaints the active
        # fill and thumb amber so the column still carries the "Optimal"
        # semantic color.
        st.markdown("**Optimal diet**")
        # Marker for the CSS rule that paints the optimal sliders amber.
        # Sits at the top of the column; `:has(.optimal-col-marker)`
        # selects this stColumn so the rule scopes to its descendants.
        st.markdown('<div class="optimal-col-marker"></div>', unsafe_allow_html=True)
        # `solved` and `opt_x` were computed earlier (for the fill CSS).
        for f in data["foods"]:
            ub = slider_upper_bound(f, data)
            if solved:
                val = max(0.0, min(float(opt_x.get(f, 0.0)), ub))
            else:
                val = 0.0
            # Stable widget key (no value stamp). When the LP value
            # changes, BaseWeb updates the existing component's value
            # prop and React commits the new gradient + new thumb
            # transform in the same render pass: no unmount/remount.
            # A value-stamped key forces a remount on every value
            # change, which left a frame where the new container had
            # the new CSS gradient applied but the thumb's inline
            # transform hadn't been set yet ("bar leads thumb"). The
            # pre-sync above pins session_state to the LP value, so
            # this widget always shows the current LP solution.
            st.slider(
                label=f,
                key=f"opt_v_{f}",
                min_value=0.0,
                max_value=float(ub),
                step=0.1,
                format="%.1f",
            )

    # Inject the per-food amber gradient CSS AFTER the slider widgets
    # ship to the client. Streamlit emits elements in Python source
    # order; if the gradient CSS is emitted before the optimal sliders,
    # the new gradient stop ("amber at 45.45%") arrives at the browser
    # at t≈1.2s after Run Optimizer is clicked, but the slider's new
    # thumb position arrives at t≈2.4s: leaving a visible second of
    # "fill at new position, thumb still at old". By emitting the CSS
    # after the slider widgets, the thumb commits to its new transform
    # first, then this style update arrives ~0 frames later and just
    # recolors the inline blue gradient to amber. Thumb and amber-fill
    # stay aligned the entire time.
    st.markdown(
        f"<style>\n{opt_fill_css}\n</style>",
        unsafe_allow_html=True,
    )


# ---------- Main ----------
#
# Module-level code runs on every Streamlit rerun, so this section needs to
# be cheap and idempotent: configure the page, ensure session_state is set
# up, then assemble the four tabs.

# `set_page_config` must be the first Streamlit call; "wide" layout gives
# the three-column optimizer (Your diet | Constraints chart | Optimal diet)
# enough horizontal room.
st.set_page_config(page_title="Diet", page_icon="favicon.png", layout="wide")

# Initialize session_state defaults and apply any pending reset.
init_state()

# Tighten the top of the main block so the title sits closer to the page top
# and the tabs are visible without scrolling. 2.5rem clears the sticky header
# (running-script spinner + «« sidebar toggle in the top-right) without hiding
# the title underneath it. Same value used across the template family: see
# griffith-pse-app-template/app.py.
st.markdown(
    """
    <style>
    .block-container,
    [data-testid="stMainBlockContainer"] {
      padding-top: 2.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Home link: clicking the Griffith PSE logo navigates back to the portfolio
# site. Same-tab navigation since the user is leaving the demo. Pinned to
# the upper-left corner of the page via position:fixed so it stays visible
# while scrolling. Image is embedded from the local favicon.png as a base64
# data URL: the link still navigates to griffith-pse.com when clicked, but
# loading the page itself doesn't make any third-party request.
_FAVICON_DATA_URL = "data:image/png;base64," + base64.b64encode(
    (Path(__file__).parent / "favicon.png").read_bytes()
).decode()
st.markdown(
    """
    <style>
    .home-logo-corner {
        position: fixed;
        top: 0.5rem;
        left: 0.75rem;
        z-index: 999999;
    }
    .home-logo-corner img {
        width: 32px;
        height: 32px;
        border-radius: 4px;
        display: block;
    }
    </style>
    <a href="https://griffith-pse.com" target="_self" class="home-logo-corner">
      <img src="https://griffith-pse.com/images/favicon.png"
           alt="Griffith PSE: home" width="32" height="32" style="width:32px;height:32px;border-radius:4px;display:block" />
    </a>
    """.replace("https://griffith-pse.com/images/favicon.png", _FAVICON_DATA_URL),
    unsafe_allow_html=True,
)
st.markdown(
    "<h2 style='margin: 0 0 0.25rem 0; padding: 0; font-size: 1.5rem; font-weight: 700;'>"
    "Diet LP Optimizer "
    "<a href='https://github.com/devin-griff/diet-lp' target='_blank' "
    "title='View source on GitHub' "
    "style='display: inline-block; vertical-align: 0.02em; margin: 0 0.35rem 0 0.1rem; "
    "color: inherit;'>"
    "<svg viewBox='0 0 16 16' width='20' height='20' fill='currentColor' "
    "aria-label='GitHub'>"
    "<path d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17."
    "55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-"
    ".82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 "
    "2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59."
    "82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27"
    ".68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51"
    ".56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1."
    "07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-"
    "8-8-8z'/></svg></a>"
    "<span style='font-size: 1.15rem; font-weight: 400; color: #6b7280;'>"
    "powered by "
    "<a href='https://github.com/Pyomo/pyomo' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>Pyomo</a>"
    " + "
    "<a href='https://github.com/ERGO-Code/HiGHS' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>HiGHS</a>"
    "</span>"
    "</h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([1, 1])
with _caption_col:
    st.markdown(
        "Try to beat the optimizer: pick food quantities in the **Optimizer** tab "
        "to meet every nutrient minimum at the lowest cost you can, then click "
        "**Run Optimizer** to compare against the solver's cheapest feasible diet. Edit "
        "foods, prices, and nutrient requirements in the **Data** tab; the "
        "**Formulation** and **Logs** tabs show the underlying LP and solver output."
    )

# `st.tabs` returns one container per label, used as a context manager to
# scope subsequent `st.*` calls into that tab.
optimizer_tab, data_tab, formulation_tab, logs_tab = st.tabs(
    ["🎯 Optimizer", "📋 Data", "📐 Formulation", "📜 Logs"]
)
with optimizer_tab:
    render_optimizer_tab()
with data_tab:
    render_data_tab()
with formulation_tab:
    render_formulation_tab()
with logs_tab:
    render_logs_tab()
