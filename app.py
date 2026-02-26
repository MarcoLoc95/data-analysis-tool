"""
Data Analysis & Plotting Tool - Streamlit Web App
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import io, re

def grid_data(response):
    """Extract DataFrame from AgGrid response (compatible across versions)."""
    if hasattr(response, 'data') and isinstance(response.data, pd.DataFrame):
        return response.data
    if isinstance(response, dict) and 'data' in response:
        d = response['data']
        return d if isinstance(d, pd.DataFrame) else pd.DataFrame(d)
    return None

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

st.set_page_config(layout="wide", page_title="Data Analysis Tool", page_icon="\U0001F4CA")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

OKABE_ITO = ["#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7","#000000"]
TOL_BRIGHT = ["#4477AA","#EE6677","#228833","#CCBB44","#66CCEE","#AA3377","#BBBBBB"]
TOL_MUTED = ["#332288","#88CCEE","#44AA99","#117733","#999933","#DDCC77","#CC6677","#882255","#AA4499"]
TOL_VIBRANT = ["#EE7733","#0077BB","#33BBEE","#EE3377","#CC3311","#009988","#BBBBBB"]
TOL_LIGHT = ["#77AADD","#EE8866","#EEDD88","#FFAABB","#99DDFF","#44BB99","#BBCC33","#AAAA00"]
MONO_BLUE = ["#08306b","#2171b5","#4292c6","#6baed6","#9ecae1","#c6dbef","#deebf7"]
MONO_GREY = ["#252525","#525252","#737373","#969696","#bdbdbd","#d9d9d9","#f0f0f0"]
MONO_RED = ["#67000d","#a50f15","#cb181d","#ef3b2c","#fb6a4a","#fc9272","#fcbba1"]

QUAL_PALETTES = {
    "Okabe-Ito": OKABE_ITO, "Tol Bright": TOL_BRIGHT, "Tol Muted": TOL_MUTED,
    "Tol Vibrant": TOL_VIBRANT, "Tol Light": TOL_LIGHT,
    "Blue (mono)": MONO_BLUE, "Grey (mono)": MONO_GREY, "Red (mono)": MONO_RED,
}
SEQ_CMAPS = ["viridis","plasma","inferno","magma","cividis"]

MARKERS = {"o": "\u25CF Circle", "s": "\u25A0 Square", "^": "\u25B2 Triangle",
           "D": "\u25C6 Diamond", "v": "\u25BC Tri-down", "P": "\u271A Plus",
           "X": "\u2716 Cross", "*": "\u2605 Star"}

# ═══════════════════════════════════════════════════════════════════════════════
#  FIT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _lin(x,a,b): return a*x+b
def _quad(x,a,b,c): return a*x**2+b*x+c
def _cubic(x,a,b,c,d): return a*x**3+b*x**2+c*x+d
def _exp(x,a,b,c): return a*np.exp(b*x)+c
def _log(x,a,b): return a*np.log(x)+b
def _pow(x,a,b): return a*np.power(x,b)
def _gauss(x,a,mu,s): return a*np.exp(-0.5*((x-mu)/s)**2)
def _sigm(x,L,k,x0,b): return L/(1+np.exp(-k*(x-x0)))+b
def _sin(x,A,w,p,c): return A*np.sin(w*x+p)+c

FIT_FUNCTIONS = {
    "Linear (ax+b)":              (_lin,   ["a","b"],           "$y = ax + b$"),
    "Quadratic":                  (_quad,  ["a","b","c"],       "$y = ax^2 + bx + c$"),
    "Cubic":                      (_cubic, ["a","b","c","d"],   "$y = ax^3 + bx^2 + cx + d$"),
    "Exponential (a*exp(bx)+c)":  (_exp,   ["a","b","c"],       "$y = a\\,e^{bx} + c$"),
    "Logarithmic (a*ln(x)+b)":   (_log,   ["a","b"],           "$y = a\\,\\ln(x) + b$"),
    "Power Law (a*x^b)":         (_pow,   ["a","b"],           "$y = a\\,x^{b}$"),
    "Gaussian":                   (_gauss, ["a","mu","sigma"],  "$y = a\\,e^{-(x-\\mu)^2/2\\sigma^2}$"),
    "Sigmoid":                    (_sigm,  ["L","k","x0","b"],  "$y = L/(1+e^{-k(x-x_0)}) + b$"),
    "Sine":                       (_sin,   ["A","w","phi","c"], "$y = A\\sin(\\omega x + \\varphi) + c$"),
}

_CUSTOM_NS = {"np": np, "pi": np.pi, "e": np.e,
    "sqrt": np.sqrt, "log": np.log, "log10": np.log10,
    "exp": np.exp, "abs": np.abs,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh}

def build_custom_func(expr, params_str):
    if not expr or not params_str:
        raise ValueError("Enter both expression and parameter names.")
    pn = [p.strip() for p in params_str.split(",") if p.strip()]
    if not pn:
        raise ValueError("No parameter names provided.")
    safe_expr = expr.replace("^", "**")
    def fn(x, *args):
        ns = dict(_CUSTOM_NS); ns["x"] = x
        for name, val in zip(pn, args): ns[name] = val
        return eval(safe_expr, {"__builtins__": {}}, ns)
    return fn, pn, f"$y = {expr}$"

def parse_bounds(text, n):
    if not text.strip(): return None
    parts = [s.strip().lower() for s in text.split(",")]
    if len(parts) != n: raise ValueError(f"Expected {n} bounds, got {len(parts)}")
    vals = []
    for p in parts:
        if p in ("inf","+inf",""): vals.append(np.inf)
        elif p == "-inf": vals.append(-np.inf)
        else: vals.append(float(p))
    return vals

def confidence_band(func, xf, popt, pcov, n_data, confidence=0.95):
    from scipy.stats import t as t_dist
    k = len(popt)
    dof = max(n_data - k, 1)
    t_val = t_dist.ppf((1 + confidence) / 2, dof)
    eps = np.sqrt(np.finfo(float).eps)
    J = np.zeros((len(xf), k))
    y0 = func(xf, *popt)
    for i in range(k):
        dp = np.zeros(k)
        dp[i] = max(abs(popt[i]) * eps, eps)
        J[:, i] = (func(xf, *(popt + dp)) - y0) / dp[i]
    var_y = np.sum((J @ pcov) * J, axis=1)
    delta = t_val * np.sqrt(np.maximum(var_y, 0))
    return y0, delta

def sig_str(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."

def col_letter(n):
    """Convert 0-based index to Origin-style column name: A, B, ..., Z, AA, AB, ..."""
    result = ""
    while True:
        result = chr(ord('A') + n % 26) + result
        n = n // 26 - 1
        if n < 0:
            break
    return "Col" + result

def make_default_df(ncols=3, nrows=10):
    return pd.DataFrame({col_letter(i): pd.array([np.nan]*nrows, dtype=float) for i in range(ncols)})

# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "df": make_default_df(),
        "formulas": {},
        "col_meta": {},  # {col: {"name":"", "units":"", "comments":""}}
        "analysis_log": [],
        "table_ver": 0,  # increment to force data_editor re-creation
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def get_df():
    return st.session_state.df

def set_df(df):
    st.session_state.df = df

def eval_formulas():
    df = get_df()
    for col, formula in st.session_state.formulas.items():
        if col not in df.columns or not formula: continue
        expr = formula.lstrip("=").strip().replace("^", "**")
        if not expr: continue
        try:
            def _shift(arr, n=1):
                out = np.empty_like(arr, dtype=float); out[:] = np.nan
                if n > 0: out[n:] = arr[:-n] if n < len(arr) else np.nan
                elif n < 0: out[:n] = arr[-n:] if -n < len(arr) else np.nan
                else: out[:] = arr
                return out
            def _diff(arr):
                out = np.empty_like(arr, dtype=float); out[0] = np.nan
                out[1:] = np.diff(arr.astype(float)); return out
            ns = dict(_CUSTOM_NS)
            ns.update({"shift": _shift, "diff": _diff})
            for c in df.columns:
                if c != col: ns[c] = pd.to_numeric(df[c], errors='coerce').values
            result = eval(expr, {"__builtins__": {}}, ns)
            if np.isscalar(result): df[col] = result
            else: df[col] = pd.to_numeric(pd.Series(result), errors='coerce')
        except Exception:
            pass

def get_col_label(col):
    meta = st.session_state.col_meta.get(col, {})
    name = meta.get("name", "")
    units = meta.get("units", "")
    if name and units: return f"{name} ({units})"
    if name: return name
    if units: return f"{col} ({units})"
    return col

def add_log(text, header=False):
    if header:
        st.session_state.analysis_log.append(f"\n{'='*60}\n{text}\n{'='*60}")
    else:
        st.session_state.analysis_log.append(text)

def clear_log():
    st.session_state.analysis_log = []

def get_pal(name, n):
    colors = QUAL_PALETTES.get(name, OKABE_ITO)
    return [colors[i % len(colors)] for i in range(n)]

def num_cols():
    df = get_df()
    return [c for c in df.columns if pd.to_numeric(df[c], errors='coerce').notna().any()]

def col_fmt(c):
    """Format column for dropdown display: ColA: Temperature (K)"""
    if c == "(none)": return "(none)"
    lbl = get_col_label(c)
    return f"{c}: {lbl}" if lbl != c else c

def bump_table():
    """Increment table version to force data_editor widget re-creation."""
    st.session_state.table_ver = st.session_state.get("table_ver", 0) + 1

def load_data(df_raw):
    """Load a dataframe: rename columns to ColA..ColZ..ColAA, move original headers to display names."""
    n_cols = len(df_raw.columns)
    new_names = [col_letter(i) for i in range(n_cols)]
    old_names = list(df_raw.columns)

    # Check if headers look like text (not just integers from headerless CSV)
    has_text_headers = any(isinstance(h, str) and not h.replace('.','',1).replace('-','',1).isdigit()
                          for h in old_names)

    df_new = df_raw.copy()
    df_new.columns = new_names

    st.session_state.formulas = {}
    meta = {}
    for i, nn in enumerate(new_names):
        if has_text_headers:
            meta[nn] = {"name": str(old_names[i]), "units": "", "comments": ""}
        else:
            meta[nn] = {"name": "", "units": "", "comments": ""}
    st.session_state.col_meta = meta
    set_df(df_new)
    bump_table()

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR - FILE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("\U0001F4CA Data Analysis Tool")
    st.markdown("---")

    st.subheader("Load Data")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "tsv", "xlsx", "xls"],
                                 key="file_uploader")
    if uploaded is not None:
        # Use a flag to only load once per file
        fkey = f"loaded_{uploaded.name}_{uploaded.size}"
        if fkey not in st.session_state:
            try:
                if uploaded.name.endswith((".xlsx", ".xls")):
                    df_raw = pd.read_excel(uploaded)
                else:
                    df_raw = pd.read_csv(uploaded)
                load_data(df_raw)
                st.session_state[fkey] = True
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")

    pasted = st.text_area("Or paste data (tab/comma separated)", height=80, key="paste_area")
    if st.button("Load pasted data") and pasted.strip():
        try:
            sep = "\t" if "\t" in pasted else ","
            df_raw = pd.read_csv(io.StringIO(pasted), sep=sep)
            load_data(df_raw)
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    st.subheader("Table Operations")

    # +Column | -Column
    c1, c2 = st.columns(2)
    with c1:
        if st.button("+Column", use_container_width=True):
            df = get_df()
            new_col = col_letter(len(df.columns))
            df[new_col] = np.nan
            st.session_state.col_meta[new_col] = {"name": "", "units": "", "comments": ""}
            set_df(df)
            bump_table()
            st.rerun()
    with c2:
        if st.button("\u2212Column", use_container_width=True):
            df = get_df()
            if len(df.columns) > 1:
                drop_col = df.columns[-1]
                df = df.drop(columns=[drop_col])
                st.session_state.col_meta.pop(drop_col, None)
                st.session_state.formulas.pop(drop_col, None)
                set_df(df)
                bump_table()
                st.rerun()

    # +Row | -Row
    c3, c4 = st.columns(2)
    with c3:
        if st.button("+Row", use_container_width=True):
            df = get_df()
            new = pd.DataFrame({c: [np.nan] for c in df.columns})
            set_df(pd.concat([df, new], ignore_index=True))
            bump_table()
            st.rerun()
    with c4:
        if st.button("\u2212Row", use_container_width=True):
            df = get_df()
            if len(df) > 1:
                set_df(df.iloc[:-1].reset_index(drop=True))
                bump_table()
                st.rerun()

    # +10 Rows | +50 Rows
    c5, c6 = st.columns(2)
    with c5:
        if st.button("+10 Rows", use_container_width=True):
            df = get_df()
            new = pd.DataFrame({c: [np.nan]*10 for c in df.columns})
            set_df(pd.concat([df, new], ignore_index=True))
            bump_table()
            st.rerun()
    with c6:
        if st.button("+50 Rows", use_container_width=True):
            df = get_df()
            new = pd.DataFrame({c: [np.nan]*50 for c in df.columns})
            set_df(pd.concat([df, new], ignore_index=True))
            bump_table()
            st.rerun()

    # Clear All (centered)
    _, cc, _ = st.columns([1, 2, 1])
    with cc:
        if st.button("Clear All", use_container_width=True):
            set_df(make_default_df())
            st.session_state.formulas = {}
            st.session_state.col_meta = {}
            bump_table()
            st.rerun()

    st.markdown("---")
    st.subheader("Download")
    df = get_df()
    # Export CSV with display names as headers
    export_df = df.copy()
    export_df.columns = [get_col_label(c) for c in df.columns]
    csv_data = export_df.to_csv(index=False)
    st.download_button("Download CSV", csv_data, "data.csv", "text/csv", use_container_width=True)

    if st.session_state.analysis_log:
        log_parts = []
        if st.session_state.get("notes", "").strip():
            log_parts.append(f"NOTES:\n{st.session_state.notes}\n{'='*60}\n")
        log_parts.extend(st.session_state.analysis_log)
        log_text = "\n".join(log_parts)
        st.download_button("Download Analysis (.txt)", log_text, "analysis.txt", "text/plain",
                          use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_data, tab_graph, tab_analysis = st.tabs(["\U0001F4CB Data", "\U0001F4C8 Graph", "\U0001F9EA Analysis"])

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_data:
    df = get_df()
    cols_list = list(df.columns)
    n_c = len(cols_list)
    ver = st.session_state.get("table_ver", 0)

    # ── Metadata grid (one row per column) ──
    with st.expander("Column metadata", expanded=False):
        meta_data = []
        for col in cols_list:
            m = st.session_state.col_meta.get(col, {"name":"","units":"","comments":""})
            meta_data.append({
                "Column": col,
                "Name": m.get("name", ""),
                "Units": m.get("units", ""),
                "Formula": st.session_state.formulas.get(col, ""),
                "Comment": m.get("comments", ""),
            })
        meta_df = pd.DataFrame(meta_data)

        gb_meta = GridOptionsBuilder.from_dataframe(meta_df)
        gb_meta.configure_column("Column", editable=False, minWidth=90, maxWidth=110, pinned="left",
                                  cellStyle={"fontWeight": "bold", "backgroundColor": "#f0f0f0"})
        gb_meta.configure_column("Name", editable=True, minWidth=140, flex=2)
        gb_meta.configure_column("Units", editable=True, minWidth=90, flex=1)
        gb_meta.configure_column("Formula", editable=True, minWidth=180, flex=2,
                                  cellStyle={"fontFamily": "monospace", "backgroundColor": "#fdf6f0"})
        gb_meta.configure_column("Comment", editable=True, minWidth=140, flex=2)
        gb_meta.configure_grid_options(
            domLayout='autoHeight',
            suppressMovableColumns=True,
            enterNavigatesVertically=True,
            enterNavigatesVerticallyAfterEdit=True,
        )

        meta_response = AgGrid(
            meta_df,
            gridOptions=gb_meta.build(),
            update_mode=GridUpdateMode.VALUE_CHANGED,
            theme="streamlit",
            key=f"meta_grid_{ver}",
        )

        # Write back
        edited_meta = grid_data(meta_response)
        if edited_meta is not None:
            for _, row in edited_meta.iterrows():
                col = str(row["Column"])
                if col not in cols_list:
                    continue
                st.session_state.col_meta[col] = {
                    "name": str(row["Name"]) if pd.notna(row["Name"]) else "",
                    "units": str(row["Units"]) if pd.notna(row["Units"]) else "",
                    "comments": str(row["Comment"]) if pd.notna(row["Comment"]) else "",
                }
                f_val = str(row["Formula"]).strip() if pd.notna(row["Formula"]) else ""
                if f_val:
                    st.session_state.formulas[col] = f_val
                elif col in st.session_state.formulas:
                    del st.session_state.formulas[col]

    # ── Evaluate formulas ──
    if st.session_state.formulas:
        eval_formulas()

    # ── Data info ──
    nn_missing = int(df.isna().sum().sum())
    nc_num = len(df.select_dtypes(include=[np.number]).columns)
    st.caption(f"{len(df)} rows x {n_c} cols ({nc_num} numeric"
               f"{f', {nn_missing} missing' if nn_missing else ''})")

    # ── Main data grid ──
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        editable=True,
        resizable=True,
        sortable=True,
        filter=True,
        type=["numericColumn"],
    )
    # Set column headers to display name + units
    for col in cols_list:
        lbl = get_col_label(col)
        header = f"{col}: {lbl}" if lbl != col else col
        gb.configure_column(col, headerName=header)

    gb.configure_grid_options(
        enterNavigatesVertically=True,
        enterNavigatesVerticallyAfterEdit=True,
        undoRedoCellEditing=True,
        undoRedoCellEditingLimit=20,
        suppressMovableColumns=True,
        stopEditingWhenCellsLoseFocus=True,
    )

    grid_response = AgGrid(
        df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.VALUE_CHANGED,
        height=500,
        fit_columns_on_grid_load=(n_c <= 8),
        theme="streamlit",
        allow_unsafe_jscode=True,
        key=f"data_grid_{ver}",
    )

    # Write back edited data
    edited_df = grid_data(grid_response)
    if edited_df is not None and len(edited_df) > 0:
        # Try to coerce each column to numeric; leave as-is if it fails
        for col in edited_df.columns:
            converted = pd.to_numeric(edited_df[col], errors='coerce')
            # Only convert if most values are numeric (not all NaN from failed coercion)
            if converted.notna().any() or edited_df[col].isna().all():
                edited_df[col] = converted
        set_df(edited_df)

    # ── Notes ──
    if "notes" not in st.session_state:
        st.session_state.notes = ""
    st.session_state.notes = st.text_area("\U0001F4DD Notes", st.session_state.notes, height=80,
                                           placeholder="Experiment notes...")

# ═══════════════════════════════════════════════════════════════════════════════
#  GRAPH TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_graph:
    df = get_df()
    cols = list(df.columns)
    ncols = num_cols()

    # Controls in left column, plot in right
    ctrl_col, plot_col = st.columns([1, 2.5])

    with ctrl_col:
        plot_type = st.selectbox("Plot type", ["Scatter", "Line", "Line+Scatter",
                                                "Histogram", "Box Plot", "Violin Plot", "Error Bar"])

        multi_stat = plot_type in ("Box Plot", "Violin Plot", "Histogram")

        if multi_stat:
            sel_cols = st.multiselect("Columns", ncols, default=ncols[:2] if len(ncols)>=2 else ncols, format_func=col_fmt)
        else:
            xc = st.selectbox("X column", ncols, index=0 if ncols else 0, format_func=col_fmt)
            yc = st.selectbox("Y column", ncols, index=min(1, len(ncols)-1) if len(ncols)>1 else 0, format_func=col_fmt)

        if plot_type == "Error Bar":
            err_opts = ["(none)"] + ncols
            yerr_col = st.selectbox("Y error column", err_opts, format_func=col_fmt)
            xerr_col = st.selectbox("X error column", err_opts, format_func=col_fmt)

        # Style
        with st.expander("Style", expanded=False):
            palette = st.selectbox("Palette", list(QUAL_PALETTES.keys()))
            marker_code = st.selectbox("Marker", list(MARKERS.keys()),
                                       format_func=lambda x: MARKERS[x])
            c1, c2 = st.columns(2)
            with c1:
                marker_size = st.slider("Marker size", 1, 20, 6)
                line_width = st.slider("Line width", 0.5, 5.0, 1.5, 0.5)
            with c2:
                alpha = st.slider("Alpha", 0.1, 1.0, 0.8, 0.05)
                plot_scale = st.slider("Text scale", 0.5, 3.0, 1.0, 0.25)
            show_grid = st.checkbox("Grid", True)
            show_legend = st.checkbox("Legend", True)
            show_units = st.checkbox("Show units in labels", True)

        # Labels
        with st.expander("Labels", expanded=False):
            title = st.text_input("Title", "")
            xlabel = st.text_input("X label", "")
            ylabel = st.text_input("Y label", "")

        # Advanced - only show relevant options
        with st.expander("Advanced", expanded=False):
            # Axis options - always shown
            st.markdown("**Axis**")
            ac1, ac2 = st.columns(2)
            with ac1:
                xmin = st.text_input("X min", "", placeholder="auto")
                ymin = st.text_input("Y min", "", placeholder="auto")
            with ac2:
                xmax = st.text_input("X max", "", placeholder="auto")
                ymax = st.text_input("Y max", "", placeholder="auto")
            log_x = st.checkbox("Log X")
            log_y = st.checkbox("Log Y")
            minor_ticks = st.checkbox("Minor ticks")

            # Fit - only for XY plot types
            do_fit = False; fit_model = ""; bounds_lo = ""; bounds_hi = ""
            fit_extend = False; fit_confidence = False; fit_ls = "--"; fit_lw = 2.0
            custom_expr = ""; custom_params = ""
            if plot_type in ("Scatter", "Line", "Line+Scatter", "Error Bar"):
                st.markdown("---")
                st.markdown("**Fit**")
                do_fit = st.checkbox("Overlay fit")
                fit_model = st.selectbox("Fit model", list(FIT_FUNCTIONS.keys()) + ["Custom..."])
                if fit_model == "Custom...":
                    custom_expr = st.text_input("f(x) =", "", placeholder="Vmax*x/(Km+x)")
                    custom_params = st.text_input("Parameters", "", placeholder="Vmax, Km")
                bounds_lo = st.text_input("Lower bounds", "", placeholder="-inf, 0, -inf")
                bounds_hi = st.text_input("Upper bounds", "", placeholder="inf, 100, inf")
                fit_extend = st.checkbox("Extend fit to axis range")
                fit_confidence = st.checkbox("95% confidence band")
                fit_ls = st.selectbox("Fit line style", ["--", "-", "-.", ":"],
                                      format_func=lambda x: {"--":"Dashed","-":"Solid","-.":"Dash-dot",":":"Dotted"}[x])
                fit_lw = st.slider("Fit line width", 0.5, 5.0, 2.0, 0.5)

            # Histogram options
            bin_mode = "Auto"; n_bins = 30; bin_width = 1.0
            hist_density = False; hist_stack = False
            if plot_type == "Histogram":
                st.markdown("---")
                st.markdown("**Histogram**")
                bin_mode = st.selectbox("Bin mode", ["Auto", "Bin count", "Bin width"])
                if bin_mode == "Bin count":
                    n_bins = st.number_input("Number of bins", 2, 500, 30)
                elif bin_mode == "Bin width":
                    bin_width = st.number_input("Bin width", 0.001, 1e6, 1.0, format="%.4f")
                hist_density = st.checkbox("Normalize")
                hist_stack = st.checkbox("Stacked")

            # Box Plot options
            box_mean = False; whisker_mode = "IQR (1.5x)"; box_points = True
            if plot_type == "Box Plot":
                st.markdown("---")
                st.markdown("**Box Plot**")
                box_mean = st.checkbox("Show mean")
                whisker_mode = st.selectbox("Whiskers", ["IQR (1.5x)", "Min/Max", "1 SD", "1 SEM"])
                box_points = st.checkbox("Show data points", True)

            # Violin Plot options
            violin_extend = False
            if plot_type == "Violin Plot":
                st.markdown("---")
                st.markdown("**Violin Plot**")
                violin_extend = st.checkbox("Extend to zero")

            # Color mapping - only for scatter types
            cmap = "viridis"; ccol = "(none)"
            if plot_type in ("Scatter", "Line+Scatter"):
                st.markdown("---")
                st.markdown("**Color Mapping**")
                cmap = st.selectbox("Colormap", SEQ_CMAPS)
                ccol = st.selectbox("Color column", ["(none)"] + ncols, format_func=col_fmt)

            # Export DPI - always shown
            st.markdown("---")
            st.markdown("**Export**")
            export_dpi = st.slider("Export DPI", 72, 600, 300, step=50)

    # ── PLOTTING ──
    with plot_col:
        sc = plot_scale
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=100)
        ax.tick_params(labelsize=10*sc)
        plot_ok = True
        fit_output = []

        try:
            colors = get_pal(palette, 10)
            ms = marker_size * sc
            mk = marker_code
            lw = line_width * sc

            if plot_type in ("Scatter", "Line", "Line+Scatter"):
                xf = pd.to_numeric(df[xc], errors='coerce')
                yf = pd.to_numeric(df[yc], errors='coerce')
                mask = xf.notna() & yf.notna()
                xp, yp = xf[mask].values, yf[mask].values

                if ccol != "(none)" and ccol in df.columns:
                    cf = pd.to_numeric(df[ccol], errors='coerce')
                    mask = mask & cf.notna()
                    xp, yp, cp = xf[mask].values, yf[mask].values, cf[mask].values
                    scat = ax.scatter(xp, yp, c=cp, s=ms**2, marker=mk,
                                      alpha=alpha, cmap=cmap, edgecolors='none')
                    fig.colorbar(scat, ax=ax, label=get_col_label(ccol))
                else:
                    if "Scatter" in plot_type:
                        ax.scatter(xp, yp, c=colors[0], s=ms**2, marker=mk, alpha=alpha,
                                   edgecolors='white', linewidths=0.3*sc, label=get_col_label(yc))
                    if "Line" in plot_type:
                        order = np.argsort(xp)
                        ax.plot(xp[order], yp[order], c=colors[0], lw=lw, alpha=alpha,
                                label=get_col_label(yc) if "Scatter" not in plot_type else None)

            elif plot_type == "Error Bar":
                xf = pd.to_numeric(df[xc], errors='coerce')
                yf = pd.to_numeric(df[yc], errors='coerce')
                mask = xf.notna() & yf.notna()
                ye = None; xe = None
                if yerr_col != "(none)" and yerr_col in df.columns:
                    ye_s = pd.to_numeric(df[yerr_col], errors='coerce')
                    mask = mask & ye_s.notna()
                if xerr_col != "(none)" and xerr_col in df.columns:
                    xe_s = pd.to_numeric(df[xerr_col], errors='coerce')
                    mask = mask & xe_s.notna()
                xp, yp = xf[mask].values, yf[mask].values
                if yerr_col != "(none)" and yerr_col in df.columns:
                    ye = pd.to_numeric(df[yerr_col], errors='coerce')[mask].values
                if xerr_col != "(none)" and xerr_col in df.columns:
                    xe = pd.to_numeric(df[xerr_col], errors='coerce')[mask].values
                ax.errorbar(xp, yp, yerr=ye, xerr=xe, fmt=mk, color=colors[0],
                           markersize=ms, capsize=3*sc, elinewidth=0.8*sc, lw=lw, alpha=alpha,
                           label=get_col_label(yc))

            elif plot_type == "Histogram":
                if not sel_cols:
                    st.warning("Select columns.")
                    plot_ok = False
                else:
                    arrs = []
                    for s in sel_cols:
                        v = pd.to_numeric(df[s], errors='coerce').dropna().values
                        if len(v): arrs.append(v)
                    if arrs:
                        hcolors = get_pal(palette, len(arrs))
                        if bin_mode == "Auto":
                            bins = "auto"
                        elif bin_mode == "Bin count":
                            bins = n_bins
                        else:
                            all_v = np.concatenate(arrs)
                            lo, hi = np.nanmin(all_v), np.nanmax(all_v)
                            bins = np.arange(lo, hi + bin_width, bin_width)
                        lbls = [get_col_label(s) for s in sel_cols]
                        ax.hist(arrs, bins=bins, color=hcolors[:len(arrs)], alpha=alpha,
                                label=lbls, density=hist_density, stacked=hist_stack,
                                edgecolor='white', linewidth=0.5)

            elif plot_type == "Box Plot":
                if not sel_cols:
                    st.warning("Select columns.")
                    plot_ok = False
                else:
                    arrs, lbls = [], []
                    for s in sel_cols:
                        v = pd.to_numeric(df[s], errors='coerce').dropna().values
                        if len(v): arrs.append(v); lbls.append(get_col_label(s))
                    if arrs:
                        bcolors = get_pal(palette, len(arrs))
                        kw = dict(patch_artist=True, labels=lbls, widths=0.5, showmeans=box_mean,
                                  meanprops=dict(marker='D', markerfacecolor='black',
                                                 markeredgecolor='black', markersize=5*sc),
                                  medianprops=dict(color='black', linewidth=1.5*sc))
                        if whisker_mode == "Min/Max":
                            bp = ax.boxplot(arrs, whis=(0,100), showfliers=False, **kw)
                        elif whisker_mode in ("1 SD","1 SEM"):
                            bp = ax.boxplot(arrs, whis=(25,75), showfliers=False, **kw)
                            for i, arr in enumerate(arrs):
                                mean_v = arr.mean()
                                delta = arr.std(ddof=1) if whisker_mode=="1 SD" else arr.std(ddof=1)/np.sqrt(len(arr))
                                wlo, whi = mean_v - delta, mean_v + delta
                                q1, q3 = np.percentile(arr, [25,75])
                                bp['whiskers'][2*i].set_ydata([q1, wlo])
                                bp['whiskers'][2*i+1].set_ydata([q3, whi])
                                bp['caps'][2*i].set_ydata([wlo, wlo])
                                bp['caps'][2*i+1].set_ydata([whi, whi])
                        else:
                            bp = ax.boxplot(arrs, flierprops=dict(marker=mk, markersize=max(ms*0.5,3), alpha=0.4), **kw)
                        for patch, col in zip(bp['boxes'], bcolors):
                            patch.set_facecolor(col); patch.set_alpha(alpha)
                        if box_points:
                            for i, arr in enumerate(arrs):
                                jit = np.random.normal(0, 0.04, len(arr))
                                ax.scatter(np.full(len(arr), i+1)+jit, arr, c='black',
                                           s=ms**1.5, marker=mk, alpha=0.3, zorder=3)

            elif plot_type == "Violin Plot":
                if not sel_cols:
                    st.warning("Select columns.")
                    plot_ok = False
                else:
                    arrs, lbls = [], []
                    for s in sel_cols:
                        v = pd.to_numeric(df[s], errors='coerce').dropna().values
                        if len(v) > 1: arrs.append(v); lbls.append(get_col_label(s))
                    if arrs:
                        vcolors = get_pal(palette, len(arrs))
                        if violin_extend:
                            from scipy.stats import gaussian_kde
                            positions = list(range(1, len(arrs)+1))
                            for i, arr in enumerate(arrs):
                                kde = gaussian_kde(arr)
                                vmin, vmax = arr.min(), arr.max()
                                span = vmax - vmin
                                grid = np.linspace(vmin - span*0.5, vmax + span*0.5, 200)
                                density = kde(grid)
                                density = density / density.max() * 0.4
                                ax.fill_betweenx(grid, positions[i]-density, positions[i]+density,
                                                 color=vcolors[i%len(vcolors)], alpha=alpha,
                                                 edgecolor='black', linewidth=0.5*sc)
                            for i, arr in enumerate(arrs):
                                ax.hlines(arr.mean(), positions[i]-0.15, positions[i]+0.15,
                                          color='black', linewidth=1.5*sc, zorder=4)
                                ax.hlines(np.median(arr), positions[i]-0.15, positions[i]+0.15,
                                          color='#555', linewidth=1*sc, linestyle='--', zorder=4)
                        else:
                            parts = ax.violinplot(arrs, showmeans=True, showmedians=True, showextrema=False)
                            for i, pc in enumerate(parts['bodies']):
                                pc.set_facecolor(vcolors[i%len(vcolors)]); pc.set_alpha(alpha)
                                pc.set_edgecolor('black'); pc.set_linewidth(0.5*sc)
                            parts['cmeans'].set_color('black'); parts['cmeans'].set_linewidth(1.5*sc)
                            parts['cmedians'].set_color('#555'); parts['cmedians'].set_linestyle('--')
                        ax.set_xticks(range(1, len(lbls)+1))
                        ax.set_xticklabels(lbls, rotation=30, ha='right')
                        for i, arr in enumerate(arrs):
                            jit = np.random.normal(0, 0.06, len(arr))
                            ax.scatter(np.full(len(arr), i+1)+jit, arr, c=vcolors[i%len(vcolors)],
                                       s=max(ms*3,14), marker=mk, alpha=0.5,
                                       edgecolors='white', linewidths=0.3*sc, zorder=3)

            # ── FIT OVERLAY ──
            if do_fit and plot_type not in ("Histogram","Box Plot","Violin Plot") and plot_ok:
                try:
                    xf = pd.to_numeric(df[xc], errors='coerce')
                    yf = pd.to_numeric(df[yc], errors='coerce')
                    mask = xf.notna() & yf.notna()
                    x_fit, y_fit = xf[mask].values, yf[mask].values

                    if fit_model == "Custom...":
                        func, pn, formula_tex = build_custom_func(custom_expr, custom_params)
                    else:
                        func, pn, formula_tex = FIT_FUNCTIONS[fit_model]

                    n_p = len(pn)
                    lo_b = parse_bounds(bounds_lo, n_p) if bounds_lo.strip() else [-np.inf]*n_p
                    hi_b = parse_bounds(bounds_hi, n_p) if bounds_hi.strip() else [np.inf]*n_p

                    popt, pcov = curve_fit(func, x_fit, y_fit, p0=[1.0]*n_p,
                                           bounds=(lo_b, hi_b), maxfev=50000)
                    perr = np.sqrt(np.diag(pcov))

                    # Stats
                    yp = func(x_fit, *popt)
                    residuals = y_fit - yp
                    ss_r = np.sum(residuals**2)
                    ss_t = np.sum((y_fit - y_fit.mean())**2)
                    n = len(y_fit); k = len(popt); dof = n - k
                    r2 = 1 - ss_r/ss_t if ss_t else float('nan')
                    red_chi2 = ss_r/dof if dof > 0 else float('nan')

                    # Plot fit line
                    if fit_extend:
                        xlims = ax.get_xlim()
                        xf_line = np.linspace(xlims[0], xlims[1], 300)
                    else:
                        xf_line = np.linspace(x_fit.min(), x_fit.max(), 300)

                    lbl = formula_tex + "\n" + "\n".join(f"${p} = {v:.4g}$" for p,v in zip(pn, popt))
                    lbl += f"\n$R^2 = {r2:.4f}$"
                    ax.plot(xf_line, func(xf_line, *popt), color='#CC0000', ls=fit_ls,
                            lw=fit_lw*sc, label=lbl, zorder=5)

                    if fit_confidence:
                        y_band, delta = confidence_band(func, xf_line, popt, pcov, n)
                        ax.fill_between(xf_line, y_band-delta, y_band+delta,
                                        color='#CC0000', alpha=0.12, zorder=4)

                    # Log to analysis
                    disp = fit_model if fit_model != "Custom..." else custom_expr
                    fit_output.append(f"Curve Fit (from Graph): {disp}")
                    fit_output.append(f"  N = {n}")
                    fit_output.append("  Parameters:")
                    for p, v, e_ in zip(pn, popt, perr):
                        fit_output.append(f"    {p:>8s} = {v:>14.6g} +/- {e_:.6g}")
                    fit_output.append(f"  R2         = {r2:.6g}")
                    fit_output.append(f"  RMSE       = {np.sqrt(ss_r/n):.6g}")
                    fit_output.append(f"  Res. Var.  = {red_chi2:.6g}  (dof = {dof})")
                    if ss_r > 0:
                        _log_ssr = n * np.log(ss_r/n)
                        fit_output.append(f"  AIC        = {_log_ssr + 2*k:.4g}")
                        fit_output.append(f"  BIC        = {_log_ssr + np.log(n)*k:.4g}")

                except Exception as e:
                    st.warning(f"Fit failed: {e}")

            # ── AXIS FORMATTING ──
            if plot_ok:
                if log_x: ax.set_xscale('log')
                if log_y: ax.set_yscale('log')
                if minor_ticks: ax.minorticks_on()

                # Labels
                if not multi_stat:
                    xl = xlabel if xlabel else (get_col_label(xc) if show_units else xc)
                    yl = ylabel if ylabel else (get_col_label(yc) if show_units else yc)
                    ax.set_xlabel(xl, fontsize=12*sc)
                    ax.set_ylabel(yl, fontsize=12*sc)
                else:
                    if xlabel: ax.set_xlabel(xlabel, fontsize=12*sc)
                    if ylabel: ax.set_ylabel(ylabel, fontsize=12*sc)
                if title: ax.set_title(title, fontsize=14*sc)

                # Ranges
                try:
                    if xmin: ax.set_xlim(left=float(xmin))
                    if xmax: ax.set_xlim(right=float(xmax))
                    if ymin: ax.set_ylim(bottom=float(ymin))
                    if ymax: ax.set_ylim(top=float(ymax))
                except ValueError:
                    pass

                ax.grid(show_grid, alpha=0.3)
                if show_legend and ax.get_legend_handles_labels()[1]:
                    ax.legend(fontsize=8*sc)
                fig.tight_layout()

        except Exception as e:
            st.error(f"Plotting error: {e}")
            plot_ok = False

        st.pyplot(fig)

        # Export buttons
        c_exp1, c_exp2, c_exp3 = st.columns(3)
        for fmt, col in [("png", c_exp1), ("pdf", c_exp2), ("svg", c_exp3)]:
            with col:
                buf = io.BytesIO()
                try:
                    fig.savefig(buf, format=fmt, dpi=export_dpi, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(f"Download .{fmt}", buf, f"plot.{fmt}",
                                      use_container_width=True)
                except Exception:
                    pass

        plt.close(fig)

        # Fit output display
        if fit_output:
            with st.expander("Fit Results", expanded=True):
                st.code("\n".join(fit_output))

# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_analysis:
    df = get_df()
    ncols_a = num_cols()

    ctrl_a, out_a = st.columns([1, 2])

    with ctrl_a:
        analysis_type = st.selectbox("Analysis", [
            "Descriptive Statistics", "Correlation Matrix", "Curve Fit",
            "t-Test", "ANOVA (one-way)", "Normality", "Non-Parametric"])

        if analysis_type == "Descriptive Statistics":
            ds_col = st.selectbox("Column", ncols_a, key="ds_col", format_func=col_fmt)
            if st.button("Run", key="btn_ds"):
                d = pd.to_numeric(df[ds_col], errors='coerce').dropna().values
                add_log(f"Descriptive Statistics: {get_col_label(ds_col)}", header=True)
                add_log(f"  N        = {len(d)}")
                add_log(f"  Mean     = {np.mean(d):.6g}")
                add_log(f"  Median   = {np.median(d):.6g}")
                add_log(f"  SD       = {np.std(d, ddof=1):.6g}")
                add_log(f"  SEM      = {np.std(d, ddof=1)/np.sqrt(len(d)):.6g}")
                add_log(f"  Min      = {np.min(d):.6g}")
                add_log(f"  Max      = {np.max(d):.6g}")
                add_log(f"  Range    = {np.ptp(d):.6g}")
                q1, q3 = np.percentile(d, [25, 75])
                add_log(f"  Q1       = {q1:.6g}")
                add_log(f"  Q3       = {q3:.6g}")
                add_log(f"  IQR      = {q3-q1:.6g}")
                add_log(f"  Skewness = {stats.skew(d):.6g}")
                add_log(f"  Kurtosis = {stats.kurtosis(d):.6g}")

        elif analysis_type == "Correlation Matrix":
            corr_cols = st.multiselect("Columns", ncols_a, default=ncols_a[:3] if len(ncols_a)>=3 else ncols_a,
                                       key="corr_cols", format_func=col_fmt)
            corr_method = st.selectbox("Method", ["Pearson", "Spearman"], key="corr_method")
            if st.button("Run", key="btn_corr") and len(corr_cols) >= 2:
                sub = df[corr_cols].apply(pd.to_numeric, errors='coerce')
                add_log(f"Correlation ({corr_method})", header=True)
                for i, c1 in enumerate(corr_cols):
                    for c2 in corr_cols[i+1:]:
                        d1 = sub[c1].dropna(); d2 = sub[c2].dropna()
                        idx = d1.index.intersection(d2.index)
                        if len(idx) < 3: continue
                        if corr_method == "Pearson":
                            r, p = stats.pearsonr(d1[idx], d2[idx])
                        else:
                            r, p = stats.spearmanr(d1[idx], d2[idx])
                        add_log(f"  {get_col_label(c1)} vs {get_col_label(c2)}: r={r:.4f}, p={p:.4g} {sig_str(p)}")

        elif analysis_type == "Curve Fit":
            cf_x = st.selectbox("X column", ncols_a, key="cf_x", format_func=col_fmt)
            cf_y = st.selectbox("Y column", ncols_a, index=min(1, len(ncols_a)-1) if len(ncols_a)>1 else 0, key="cf_y", format_func=col_fmt)
            cf_w = st.selectbox("Weight column (sigma)", ["(none)"] + ncols_a, key="cf_w", format_func=col_fmt)
            cf_model = st.selectbox("Model", list(FIT_FUNCTIONS.keys()) + ["Custom..."], key="cf_model")
            if cf_model == "Custom...":
                cf_expr = st.text_input("f(x) =", "", key="cf_expr", placeholder="Vmax*x/(Km+x)")
                cf_params = st.text_input("Parameters", "", key="cf_params", placeholder="Vmax, Km")
            cf_blo = st.text_input("Lower bounds", "", key="cf_blo")
            cf_bhi = st.text_input("Upper bounds", "", key="cf_bhi")

            if st.button("Run", key="btn_cf"):
                try:
                    x = pd.to_numeric(df[cf_x], errors='coerce')
                    y = pd.to_numeric(df[cf_y], errors='coerce')
                    m = x.notna() & y.notna()
                    sigma = None
                    if cf_w != "(none)" and cf_w in df.columns:
                        ye = pd.to_numeric(df[cf_w], errors='coerce')
                        m = m & ye.notna() & (ye > 0)
                        sigma = ye[m].values
                    xv, yv = x[m].values, y[m].values

                    if cf_model == "Custom...":
                        func, pn, ftex = build_custom_func(cf_expr, cf_params)
                    else:
                        func, pn, ftex = FIT_FUNCTIONS[cf_model]

                    n_p = len(pn)
                    lo_b = parse_bounds(cf_blo, n_p) if cf_blo.strip() else [-np.inf]*n_p
                    hi_b = parse_bounds(cf_bhi, n_p) if cf_bhi.strip() else [np.inf]*n_p

                    popt, pcov = curve_fit(func, xv, yv, p0=[1.0]*n_p, sigma=sigma,
                                           absolute_sigma=(sigma is not None),
                                           bounds=(lo_b, hi_b), maxfev=50000)
                    perr = np.sqrt(np.diag(pcov))

                    yp = func(xv, *popt)
                    residuals = yv - yp
                    ss_r = np.sum(residuals**2)
                    ss_t = np.sum((yv - yv.mean())**2)
                    n, k = len(yv), len(popt)
                    dof = n - k
                    r2 = 1 - ss_r/ss_t if ss_t else float('nan')
                    ar2 = 1 - (1-r2)*(n-1)/(n-k-1) if n>k+1 else float('nan')

                    if sigma is not None:
                        chi2_val = np.sum((residuals/sigma)**2)
                    else:
                        chi2_val = ss_r
                    red_chi2 = chi2_val/dof if dof > 0 else float('nan')

                    disp = cf_model if cf_model != "Custom..." else cf_expr
                    add_log(f"Curve Fitting: {disp}", header=True)
                    add_log(f"  X: {get_col_label(cf_x)} | Y: {get_col_label(cf_y)} | N: {n}")
                    if sigma is not None:
                        add_log(f"  Weights: {get_col_label(cf_w)}")
                    add_log("  Parameters:")
                    for p, v, e_ in zip(pn, popt, perr):
                        add_log(f"    {p:>8s} = {v:>14.6g} +/- {e_:.6g}")
                    add_log(f"  R2         = {r2:.6g}")
                    add_log(f"  Adj R2     = {ar2:.6g}")
                    add_log(f"  RMSE       = {np.sqrt(ss_r/n):.6g}")
                    if sigma is not None:
                        add_log(f"  Chi2       = {chi2_val:.6g}")
                        add_log(f"  Red. Chi2  = {red_chi2:.6g}  (dof = {dof})")
                        from scipy.stats import chi2 as chi2_dist
                        p_chi2 = 1 - chi2_dist.cdf(chi2_val, dof) if dof > 0 else float('nan')
                        add_log(f"  p(Chi2)    = {p_chi2:.6g}")
                    else:
                        add_log(f"  SS_res     = {ss_r:.6g}")
                        add_log(f"  Res. Var.  = {red_chi2:.6g}  (dof = {dof})")
                    if ss_r > 0:
                        _l = n * np.log(ss_r/n)
                        add_log(f"  AIC        = {_l + 2*k:.4g}")
                        add_log(f"  BIC        = {_l + np.log(n)*k:.4g}")
                except Exception as e:
                    add_log(f"Fitting Error: {e}", header=True)

        elif analysis_type == "t-Test":
            tt_type = st.selectbox("Type", ["Independent", "Paired", "One-sample"], key="tt_type")
            tt_c1 = st.selectbox("Column 1", ncols_a, key="tt_c1", format_func=col_fmt)
            if tt_type != "One-sample":
                tt_c2 = st.selectbox("Column 2", ncols_a, index=min(1, len(ncols_a)-1) if len(ncols_a)>1 else 0, key="tt_c2", format_func=col_fmt)
            else:
                tt_mu = st.number_input("Test value (mu)", value=0.0, key="tt_mu")
            tt_alt = st.selectbox("Alternative", ["two-sided", "less", "greater"], key="tt_alt")

            if st.button("Run", key="btn_tt"):
                d1 = pd.to_numeric(df[tt_c1], errors='coerce').dropna().values
                add_log(f"t-Test ({tt_type})", header=True)
                try:
                    if tt_type == "One-sample":
                        res = stats.ttest_1samp(d1, tt_mu, alternative=tt_alt)
                        add_log(f"  {get_col_label(tt_c1)}: N={len(d1)}, mean={d1.mean():.6g}")
                        add_log(f"  mu = {tt_mu}")
                    elif tt_type == "Independent":
                        d2 = pd.to_numeric(df[tt_c2], errors='coerce').dropna().values
                        res = stats.ttest_ind(d1, d2, alternative=tt_alt)
                        add_log(f"  {get_col_label(tt_c1)}: N={len(d1)}, mean={d1.mean():.6g}, SD={d1.std(ddof=1):.6g}")
                        add_log(f"  {get_col_label(tt_c2)}: N={len(d2)}, mean={d2.mean():.6g}, SD={d2.std(ddof=1):.6g}")
                    else:
                        d2 = pd.to_numeric(df[tt_c2], errors='coerce').dropna().values
                        n = min(len(d1), len(d2))
                        if len(d1) != len(d2):
                            add_log(f"  Warning: unequal lengths ({len(d1)} vs {len(d2)}), using first {n} rows.")
                        res = stats.ttest_rel(d1[:n], d2[:n], alternative=tt_alt)
                        add_log(f"  Paired N={n}")
                    add_log(f"  t = {res.statistic:.6g}")
                    add_log(f"  p = {res.pvalue:.6g} {sig_str(res.pvalue)}")
                except Exception as e:
                    add_log(f"  Error: {e}")

        elif analysis_type == "ANOVA (one-way)":
            anova_cols = st.multiselect("Groups (columns)", ncols_a, key="anova_cols", format_func=col_fmt)
            if st.button("Run", key="btn_anova") and len(anova_cols) >= 2:
                groups = [pd.to_numeric(df[c], errors='coerce').dropna().values for c in anova_cols]
                add_log("ANOVA (one-way)", header=True)
                for c, g in zip(anova_cols, groups):
                    add_log(f"  {get_col_label(c)}: N={len(g)}, mean={g.mean():.6g}, SD={g.std(ddof=1):.6g}")
                F, p = stats.f_oneway(*groups)
                add_log(f"  F = {F:.6g}")
                add_log(f"  p = {p:.6g} {sig_str(p)}")

        elif analysis_type == "Normality":
            nc = st.selectbox("Column", ncols_a, key="nc_col", format_func=col_fmt)
            if st.button("Run", key="btn_norm"):
                d = pd.to_numeric(df[nc], errors='coerce').dropna().values
                add_log(f"Normality: {get_col_label(nc)}", header=True)
                add_log(f"  N = {len(d)}")
                if len(d) < 3:
                    add_log("  Need >= 3 values.")
                else:
                    verdicts = []
                    s, p = stats.shapiro(d)
                    add_log(f"  Shapiro-Wilk: W={s:.6g}, p={p:.6g} {sig_str(p)}")
                    add_log(f"    W close to 1 = normal. {'Data appears normal.' if p>=0.05 else 'Departure from normality.'}")
                    nd = len(d)
                    add_log(f"    Best for N<50. (N={nd}, {'appropriate' if nd<50 else 'consider other tests'})")
                    verdicts.append(p >= 0.05)

                    if len(d) >= 20:
                        s, p = stats.normaltest(d)
                        add_log(f"  D'Agostino:   k2={s:.6g}, p={p:.6g} {sig_str(p)}")
                        add_log(f"    {'No evidence against normality.' if p>=0.05 else 'Skewness/kurtosis significantly non-normal.'}")
                        verdicts.append(p >= 0.05)

                    s, p = stats.kstest(d, 'norm', args=(d.mean(), d.std(ddof=1)))
                    add_log(f"  K-S test:     D={s:.6g}, p={p:.6g} {sig_str(p)}")
                    add_log(f"    {'Consistent with normality.' if p>=0.05 else 'Differs significantly from normal.'}")
                    verdicts.append(p >= 0.05)

                    r = stats.anderson(d, dist='norm')
                    add_log(f"  Anderson:     A2={r.statistic:.6g}")
                    and_normal = True
                    for sl, cv in zip(r.significance_level, r.critical_values):
                        rej = r.statistic > cv
                        if rej and sl == 5.0: and_normal = False
                        add_log(f"    {sl}%: crit={cv:.4g} -> {'REJECT' if rej else 'fail to reject'}")
                    verdicts.append(and_normal)

                    n_pass = sum(verdicts)
                    add_log(f"\n  Summary: {n_pass}/{len(verdicts)} tests consistent with normality.")
                    if n_pass == len(verdicts):
                        add_log("  Interpretation: No evidence against normality. Parametric tests appropriate.")
                    elif n_pass == 0:
                        add_log("  Interpretation: All tests reject normality. Consider non-parametric tests or transformation.")
                    else:
                        add_log(f"  Interpretation: Mixed results. Skewness={stats.skew(d):.3g}, Kurtosis={stats.kurtosis(d):.3g}.")

        elif analysis_type == "Non-Parametric":
            np_type = st.selectbox("Test", ["Mann-Whitney U", "Wilcoxon Signed-Rank",
                                             "Kruskal-Wallis"], key="np_type")
            np_c1 = st.selectbox("Column 1", ncols_a, key="np_c1", format_func=col_fmt)
            if np_type != "Kruskal-Wallis":
                np_c2 = st.selectbox("Column 2", ncols_a, index=min(1, len(ncols_a)-1) if len(ncols_a)>1 else 0, key="np_c2", format_func=col_fmt)
            else:
                np_multi = st.multiselect("Columns", ncols_a, key="np_multi", format_func=col_fmt)

            if st.button("Run", key="btn_np"):
                add_log(f"Non-Parametric: {np_type}", header=True)
                try:
                    d1 = pd.to_numeric(df[np_c1], errors='coerce').dropna().values
                    if np_type == "Mann-Whitney U":
                        d2 = pd.to_numeric(df[np_c2], errors='coerce').dropna().values
                        s, p = stats.mannwhitneyu(d1, d2, alternative='two-sided')
                        add_log(f"  U = {s:.6g}, p = {p:.6g} {sig_str(p)}")
                    elif np_type == "Wilcoxon Signed-Rank":
                        d2 = pd.to_numeric(df[np_c2], errors='coerce').dropna().values
                        n = min(len(d1), len(d2))
                        s, p = stats.wilcoxon(d1[:n], d2[:n])
                        add_log(f"  W = {s:.6g}, p = {p:.6g} {sig_str(p)}")
                    else:
                        groups = [pd.to_numeric(df[c], errors='coerce').dropna().values for c in np_multi]
                        s, p = stats.kruskal(*groups)
                        add_log(f"  H = {s:.6g}, p = {p:.6g} {sig_str(p)}")
                except Exception as e:
                    add_log(f"  Error: {e}")

    # Output display
    with out_a:
        st.subheader("Output")
        if st.button("Clear", key="btn_clear_log"):
            clear_log()
            st.rerun()
        if st.session_state.analysis_log:
            st.code("\n".join(st.session_state.analysis_log), language=None)
        else:
            st.info("Run an analysis to see results here.")
