import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import os
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Constants ───
BASE_RC = 0.34
BASE_COVERAGE = 1.25
BASE_PREMIUM = 0.0125
BASE_DISCOUNT = 0.06
BASE_ANNUITY = sum(1 / (1 + 0.06) ** y for y in range(1, 41))
ANGLES = [52, 60, 70, 75, 77]
ANGLE_COLORS = {
    52: [59, 130, 246], 60: [239, 68, 68],
    70: [234, 179, 8],  75: [168, 85, 247], 77: [34, 197, 94],
}
ANGLE_LABELS = {52: '52°', 60: '60°', 70: '70°', 75: '75°', 77: '77°'}

def angle_legend(angles):
    """Small HTML color legend for tilt-angle-colored maps."""
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px;">'
        f'<span style="width:12px;height:12px;border-radius:3px;'
        f'background:rgb({ANGLE_COLORS[a][0]},{ANGLE_COLORS[a][1]},{ANGLE_COLORS[a][2]});'
        f'display:inline-block;margin-right:5px;"></span>{a}°</span>'
        for a in sorted(angles))
    return (f'<div style="margin:4px 0 10px 2px;font-size:0.85rem;color:#475569;">'
            f'<b>Tilt angle:</b> {chips}</div>')

# ─── Stow-Reliability machinery ───
# Anchor angles where per-location PML/AAL data exists (77 carries the 75° VDE data,
# consistent with the naming convention of the original tool).
ANCHOR_ANGLES = [0, 45, 50, 52, 60, 70, 75, 77]

from scipy.interpolate import PchipInterpolator

@st.cache_resource
def load_shape_functions():
    """Canonical P50 into-wind damage-shape curves derived from the ATI/VDE
    site studies (Ft. Stockton TX, Snyder TX, Stuttgart AR). Normalized to the
    0-degree (flat) value; peak damage sits ~10-20 degrees into the wind.
    Used ONLY to shape the interpolation between the per-location anchor
    values -- anchors themselves are always respected exactly."""
    sdf = pd.read_csv(os.path.join(SCRIPT_DIR, 'shape_curves.csv'))
    # Stored in ABSOLUTE magnitudes (%, site-average): 2.0 mm sits above 3.2 mm.
    # The interpolation below only uses ratios of differences, so scale is irrelevant there.
    return {key: PchipInterpolator(sdf['theta'].values, sdf[key].values)
            for key in ['pml_20', 'pml_32', 'aal_20', 'aal_32']}, sdf


def apply_curve_correction(values, flat_values, floor_pct, shape_pow):
    """Commercial correction of the damage-vs-angle discount.

    Works on the damage ratio r = V(theta) / V(0):   r' = max(r ** shape_pow, floor)
      - shape_pow < 1  → damage declines LESS sharply with tilt (smaller risk
        discount for steeper stow); > 1 → sharper decline. 1.0 = VDE baseline.
      - floor (fraction of flat damage) → minimum damage retained at any tilt.
    Baseline (floor=0, shape_pow=1) reproduces the given VDE data exactly.
    Zero-risk locations (flat damage = 0) always stay zero.
    """
    if floor_pct <= 0 and abs(shape_pow - 1.0) < 1e-12:
        return np.asarray(values, dtype=float)
    v = np.asarray(values, dtype=float)
    v0 = np.asarray(flat_values, dtype=float)
    safe0 = np.maximum(v0, 1e-12)
    r = np.clip(v / safe0, 0.0, None)
    r_corr = np.maximum(np.power(r, shape_pow), floor_pct / 100.0)
    return np.where(v0 > 1e-12, safe0 * r_corr, 0.0)

def curve_value(df, theta, metric, suffix):
    """Vectorized PML/AAL value at an arbitrary tilt angle theta for every location.

    Between consecutive anchors [t_i, t_j] the value follows the canonical shape f:
        V(theta) = v_i + (v_j - v_i) * (f(theta) - f(t_i)) / (f(t_j) - f(t_i))
    This hits every anchor exactly and reproduces the into-wind damage bump
    (max near 10-20 deg) inside the 0-45 gap, scaled to each location's own data.
    """
    shape_fns, _ = load_shape_functions()
    f = shape_fns[f'{metric}_{suffix}']
    theta = float(np.clip(theta, 0.0, 77.0))
    for i in range(len(ANCHOR_ANGLES) - 1):
        if ANCHOR_ANGLES[i] <= theta <= ANCHOR_ANGLES[i + 1]:
            t_i, t_j = ANCHOR_ANGLES[i], ANCHOR_ANGLES[i + 1]
            break
    v_i = df[f'{metric}_{t_i}'].values
    v_j = df[f'{metric}_{t_j}'].values
    f_i, f_j, f_t = float(f(t_i)), float(f(t_j)), float(f(theta))
    if abs(f_j - f_i) > 1e-12:
        w = (f_t - f_i) / (f_j - f_i)
    else:
        w = (theta - t_i) / (t_j - t_i) if t_j != t_i else 0.0
    return v_i + (v_j - v_i) * w

# Demand data files
ORENNIA_FILE = 'orennia_market_demand_05.18.26_v2.csv'
WOODMAC_FILE = 'woodmac_demand_05.18.26.csv'


def annuity_factor(r, n=40):
    if r <= 0:
        return float(n)
    return sum(1 / (1 + r) ** y for y in range(1, n + 1))


# ─── Page Config ───
st.set_page_config(page_title="Hail Reliability Tool v5", page_icon="🌨️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
    .stApp { font-family: 'DM Sans', sans-serif; }
    .main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; letter-spacing: -0.5px; }
    .subtitle { font-size: 1rem; color: #6b7280; margin-top: 0; margin-bottom: 1.5rem; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); color: #f1f5f9; }
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #f8fafc; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 1.5rem; font-weight: 700; }
    section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] label p,
    section[data-testid="stSidebar"] label div, section[data-testid="stSidebar"] label span {
        color: #f8fafc !important; font-weight: 500 !important; opacity: 1 !important;
    }
    section[data-testid="stSidebar"] .stCheckbox label p, section[data-testid="stSidebar"] .stRadio label p,
    section[data-testid="stSidebar"] .stCheckbox div, section[data-testid="stSidebar"] .stRadio div {
        color: #f8fafc !important; opacity: 1 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #f8fafc !important; opacity: 1 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #cbd5e1 !important; }
    .metric-card { background: linear-gradient(135deg, #f8fafc, #f1f5f9); border-radius: 12px; padding: 1.2rem; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    .metric-card h4 { margin: 0; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; color: #64748b; }
    .metric-card .value { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; margin: 0.3rem 0; }
    .metric-card .unit { font-size: 0.75rem; color: #94a3b8; }
    .winner-banner { background: linear-gradient(135deg, #059669, #10b981); color: white; padding: 1rem 1.5rem; border-radius: 12px; text-align: center; font-size: 1.1rem; font-weight: 600; margin: 1rem 0; }
    .lookup-result { background: linear-gradient(135deg, #1e293b, #334155); color: #e2e8f0; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; }
    .lookup-result h3 { color: #38bdf8; margin-top: 0; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ───
@st.cache_data
def load_data(suffix):
    df = pd.read_csv(os.path.join(SCRIPT_DIR, f'hail_data_{suffix}_ext.csv'))
    # The 75° product uses the same VDE 75° damage data that the 77° label carries.
    for m in ['pml', 'aal', 'ins', 'risk']:
        df[f'{m}_75'] = df[f'{m}_77']
    return df

@st.cache_data
def load_demand_data(source):
    """Load demand CSV for the selected source."""
    filename = ORENNIA_FILE if source == 'Orennia' else WOODMAC_FILE
    path = os.path.join(SCRIPT_DIR, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        required = ['hail_lat', 'hail_lon', 'Year', 'DC Capacity (MW)']
        if all(col in df.columns for col in required):
            return df
        st.sidebar.warning(f"⚠️ {filename} missing required columns. Using uniform distribution.")
    return None

@st.cache_data
def derive_market_defaults(demand_df):
    """Derive default market sizes (GWdc) per year from the demand data."""
    if demand_df is None:
        return {2026: 36, 2027: 44, 2028: 50, 2029: 55, 2030: 60, 2031: 65, 2032: 70}
    yearly = demand_df.groupby('Year')['DC Capacity (MW)'].sum() / 1000
    counts = demand_df.groupby('Year').size()
    valid_years = counts[counts >= 20].index
    return {int(yr): round(yearly.get(yr, 0), 1) for yr in sorted(valid_years)}


def compute_costs(df, suffix, replacement_cost, coverage_ratio, annual_premium,
                  interest_rate, risk_pct, capex_dict, ins_on, risk_on, capex_on,
                  active_angles=None, reliability=100.0, stuck_pct=0.0,
                  curve_floor=None, curve_shape=None):
    """Costs per design angle with stow-reliability blending.

    ins(theta)  = PML(theta) * RC(cents/W) * coverage * premium * annuity
    risk(theta) = AAL(theta) * RC(cents/W) * annuity * risk_pct
    Effective   = R * value(design angle) + (1-R) * value(stuck angle),
    stuck angle = stuck_pct% of each design angle.
    At reliability = 100% this reproduces the original tool exactly (design-angle
    anchors are the true, unrounded PML/AAL implied by the given cost columns).
    """
    if active_angles is None:
        active_angles = ANGLES
    if curve_floor is None:
        curve_floor = damage_floor_pct     # module-level sidebar values
    if curve_shape is None:
        curve_shape = damage_shape_pow
    new_annuity = annuity_factor(interest_rate / 100.0)
    rc_cents = replacement_cost * 100.0
    k_ins = rc_cents * (coverage_ratio / 100.0) * (annual_premium / 100.0) * new_annuity
    k_risk = rc_cents * new_annuity * (risk_pct / 100.0)
    R = reliability / 100.0

    computed = df[['lat', 'lon']].copy()
    pml_flat = df['pml_0'].values
    aal_flat = df['aal_0'].values
    for angle in ANGLES:
        theta_stuck = (stuck_pct / 100.0) * angle
        pml_d = apply_curve_correction(df[f'pml_{angle}'].values, pml_flat, curve_floor, curve_shape)
        aal_d = apply_curve_correction(df[f'aal_{angle}'].values, aal_flat, curve_floor, curve_shape)
        pml_s = apply_curve_correction(curve_value(df, theta_stuck, 'pml', suffix), pml_flat, curve_floor, curve_shape)
        aal_s = apply_curve_correction(curve_value(df, theta_stuck, 'aal', suffix), aal_flat, curve_floor, curve_shape)
        pml_eff = R * pml_d + (1 - R) * pml_s
        aal_eff = R * aal_d + (1 - R) * aal_s
        ins_val = pml_eff * k_ins if ins_on else 0.0
        risk_val = aal_eff * k_risk if risk_on else 0.0
        capex_val = capex_dict[angle] if capex_on else 0.0
        computed[f'total_{angle}'] = ins_val + risk_val + capex_val
        computed[f'ins_{angle}'] = ins_val if ins_on else 0.0
        computed[f'risk_{angle}'] = risk_val if risk_on else 0.0
        computed[f'capex_{angle}'] = capex_val if capex_on else 0.0

    active_cols = [f'total_{a}' for a in active_angles]
    computed['best_angle'] = computed[active_cols].idxmin(axis=1).str.replace('total_', '').astype(int)
    computed['best_cost'] = computed[active_cols].min(axis=1)
    return computed

def compute_blended(df20, df32, pct_20, replacement_cost, coverage_ratio, annual_premium,
                    interest_rate, risk_pct, capex_dict, ins_on, risk_on, capex_on,
                    active_angles=None, reliability=100.0, stuck_pct=0.0):
    """Pro-rata blend of 2.0mm and 3.2mm glass results.

    For each angle: blended_total = (pct_20/100) * total_20 + (1-pct_20/100) * total_32
    Best angle is then selected from blended totals.
    """
    if active_angles is None:
        active_angles = ANGLES
    c20 = compute_costs(df20, '20', replacement_cost, coverage_ratio, annual_premium,
                        interest_rate, risk_pct, capex_dict, ins_on, risk_on, capex_on,
                        active_angles, reliability, stuck_pct)
    c32 = compute_costs(df32, '32', replacement_cost, coverage_ratio, annual_premium,
                        interest_rate, risk_pct, capex_dict, ins_on, risk_on, capex_on,
                        active_angles, reliability, stuck_pct)

    # Merge on lat/lon
    blended = c20[['lat', 'lon']].copy()
    w20 = pct_20 / 100.0
    w32 = 1 - w20
    for col_prefix in ['total_', 'ins_', 'risk_', 'capex_']:
        for angle in ANGLES:
            col = f'{col_prefix}{angle}'
            blended[col] = w20 * c20[col].values + w32 * c32[col].values

    active_cols = [f'total_{a}' for a in active_angles]
    blended['best_angle'] = blended[active_cols].idxmin(axis=1).str.replace('total_', '').astype(int)
    blended['best_cost'] = blended[active_cols].min(axis=1)
    return blended


def compute_luce_probabilities(computed, sigma, active_angles):
    """Compute Luce/Logit probability of each tilt angle winning at each location.

    Uses per-location cost normalization: subtract min cost across active angles so
    the cheapest angle has utility 0. Then P_a = exp(-sigma * C_a_norm) / sum_j exp(-sigma * C_j_norm).
    """
    if len(active_angles) == 0:
        return computed.copy()

    cost_cols = [f'total_{a}' for a in active_angles]
    costs = computed[cost_cols].values  # shape (n_locations, n_active)

    # Per-location normalization: subtract min so cheapest has cost 0
    min_costs = costs.min(axis=1, keepdims=True)
    costs_norm = costs - min_costs  # shape (n_locations, n_active)

    # Softmax with negative utility: lower cost -> higher probability
    exponents = -sigma * costs_norm  # negative because lower cost = higher utility
    # Numerical stability: subtract max (but since min cost is 0, max exponent is 0 -> already stable)
    exp_vals = np.exp(exponents)
    probs = exp_vals / exp_vals.sum(axis=1, keepdims=True)

    result = computed.copy()
    for i, angle in enumerate(active_angles):
        result[f'prob_{angle}'] = probs[:, i]
    # For angles not in active set, probability = 0
    for angle in ANGLES:
        if angle not in active_angles:
            result[f'prob_{angle}'] = 0.0

    return result


def interpolate_best_product(lats, lons, best_angles):
    """Create a dense grid of best-product assignments, tightly clipped to data footprint."""
    from scipy.interpolate import NearestNDInterpolator
    from scipy.spatial import Delaunay, cKDTree

    points = np.column_stack([lats, lons])
    interp = NearestNDInterpolator(points, best_angles)
    hull = Delaunay(points)
    tree = cKDTree(points)

    # Finer grid for smoother edges
    lat_range = np.arange(lats.min(), lats.max() + 0.2, 0.2)
    lon_range = np.arange(lons.min(), lons.max() + 0.2, 0.2)
    grid_lat, grid_lon = np.meshgrid(lat_range, lon_range)
    grid_pts = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])

    # Tight clip: inside convex hull AND within 0.55° of a real data point
    # (data grid spacing is ~0.5°, so this hugs coverage without leaving coastal gaps)
    inside = hull.find_simplex(grid_pts) >= 0
    dists, _ = tree.query(grid_pts)
    mask = inside & (dists < 0.55)

    clipped_pts = grid_pts[mask]
    grid_vals = interp(clipped_pts[:, 0], clipped_pts[:, 1])

    grid_df = pd.DataFrame({
        'lat': clipped_pts[:, 0], 'lon': clipped_pts[:, 1],
        'best_angle': grid_vals.astype(int),
    })
    grid_df['color'] = grid_df['best_angle'].map(ANGLE_COLORS)
    grid_df['best_angle_display'] = grid_df['best_angle'].astype(str) + '°'
    return grid_df

# US States GeoJSON URL (loaded at runtime by pydeck from public CDN)
US_STATES_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"


# ─── Sidebar ───
st.sidebar.markdown("## 🌨️ Hail Sensitivity Tool")

st.sidebar.markdown("### Tilt Angles in Analysis")
st.sidebar.caption("Toggle which stow angles are considered.")
# Visibility is controlled by the "Angle Visibility Filter" at the bottom of this panel.
VISIBLE_ANGLES = [a for a in ANGLES if st.session_state.get(f"vis_{a}", a != 70)]
if not VISIBLE_ANGLES:
    VISIBLE_ANGLES = [52]
ang_cols = st.sidebar.columns(max(1, len(VISIBLE_ANGLES)))
angle_enabled = {}
defaults = {52: True, 60: True, 70: True, 75: True, 77: True}
for i, a in enumerate(VISIBLE_ANGLES):
    with ang_cols[i]:
        angle_enabled[a] = st.checkbox(f"{a}°", value=defaults[a], key=f"ang_{a}")
ACTIVE_ANGLES = [a for a in VISIBLE_ANGLES if angle_enabled[a]]
if not ACTIVE_ANGLES:
    st.sidebar.error("Select at least one tilt angle.")
    ACTIVE_ANGLES = [VISIBLE_ANGLES[0]]

st.sidebar.markdown("### Glass Type")
glass_choice = st.sidebar.radio("Module glass thickness",
                                 ["3.2 mm", "2.0 mm", "Blended", "Compare Both"],
                                 index=2, horizontal=True)
if glass_choice == "Blended":
    glass_pct_20 = st.sidebar.slider("2.0 mm Share (%)", 0, 100, 50, 5,
                                      help="Pro-rata mix of 2.0mm and 3.2mm glass. Default 50/50.")
else:
    glass_pct_20 = 50  # unused

st.sidebar.markdown("### Stow Reliability")
rel_mode = st.sidebar.radio("Mode", ["Single Portfolio", "Compare Two Portfolios"],
                            index=0, key="rel_mode",
                            help="Compare mode runs two reliability scenarios side by side "
                                 "with an A-vs-B comparison sheet.")
if rel_mode == "Single Portfolio":
    stow_reliability = st.sidebar.slider(
        "Trackers Reaching Full Stow (%)", 0.0, 100.0, 100.0, 0.1,
        help="Share of trackers that reach the design stow angle before hail arrives. "
             "100% = original tool behavior.")
    stuck_pct = st.sidebar.slider(
        "Failed Trackers At (% of design angle)", 0, 100, 0, 1,
        help="Where the failed trackers end up, as a % of each design angle. "
             "0% = stuck flat. 15% puts a 52° tracker at 7.8°, a 60° tracker at 9.0°.")
    if stow_reliability < 100:
        _stuck_str = ", ".join(f"{a}°→{(stuck_pct/100.0)*a:.1f}°" for a in ANGLES)
        st.sidebar.caption(f"Failed-tracker angles: {_stuck_str}")
    rel_a, rel_b, stuck_a, stuck_b = stow_reliability, stow_reliability, stuck_pct, stuck_pct
    _rel_caption = (f"Stow reliability: **{stow_reliability:.1f}%**"
                    + (f" (failed at **{stuck_pct}%** of design angle)" if stow_reliability < 100 else ""))
else:
    def _portfolio_angle_select(label, key, default_angles):
        if key in st.session_state:
            st.session_state[key] = [v for v in st.session_state[key] if v in VISIBLE_ANGLES]
        sel = st.sidebar.multiselect(label, VISIBLE_ANGLES,
                                     default=[a for a in default_angles if a in VISIBLE_ANGLES],
                                     format_func=lambda a: f"{a}°", key=key)
        if not sel:
            st.sidebar.error("Select at least one angle — using fallback.")
            sel = [VISIBLE_ANGLES[-1]]
        return sorted(sel)

    st.sidebar.markdown("**🅰 Portfolio A**")
    angles_a = _portfolio_angle_select("A — Tracker Angles", "angles_a", [52, 77])
    rel_a = st.sidebar.slider("A — Reaching Full Stow (%)", 0.0, 100.0, 99.7, 0.1, key="rel_a")
    stuck_a = st.sidebar.slider("A — Failed Trackers At (% of design angle)", 0, 100, 20, 1, key="stuck_a")
    st.sidebar.markdown("**🅱 Portfolio B**")
    angles_b = _portfolio_angle_select("B — Tracker Angles", "angles_b", [60, 75])
    rel_b = st.sidebar.slider("B — Reaching Full Stow (%)", 0.0, 100.0, 90.0, 0.1, key="rel_b")
    stuck_b = st.sidebar.slider("B — Failed Trackers At (% of design angle)", 0, 100, 20, 1, key="stuck_b")
    stow_reliability, stuck_pct = rel_a, stuck_a  # defaults for shared code paths
    _ang_a_str = "/".join(f"{a}°" for a in angles_a)
    _ang_b_str = "/".join(f"{a}°" for a in angles_b)
    _rel_caption = (f"Portfolios: **A [{_ang_a_str}] {rel_a:.1f}%** vs "
                    f"**B [{_ang_b_str}] {rel_b:.1f}%** stow reliability")

st.sidebar.markdown("### Damage Curve Correction")
st.sidebar.caption("Commercial adjustment of the damage-vs-tilt discount. "
                   "Defaults = VDE curves exactly.")
damage_shape_pow = st.sidebar.slider(
    "Decay Sharpness (power)", 0.30, 2.00, 1.00, 0.05,
    help="Applied to the damage ratio r = V(θ)/V(flat) as r^power. "
         "< 1.0: damage declines less sharply with tilt (smaller risk discount for steeper "
         "stow — the commercial-feedback direction). > 1.0: sharper decline. 1.0 = VDE baseline.")
damage_floor_pct = st.sidebar.slider(
    "Minimum Damage Floor (% of flat 0° damage)", 0.0, 50.0, 0.0, 0.5,
    help="Damage at any tilt is at least this share of the site's flat (0°) damage — "
         "e.g. 10 means even a fully-stowed tracker retains ≥10% of flat PML/AAL. "
         "0 = VDE baseline. Zero-risk sites stay zero.")
if damage_floor_pct > 0 or abs(damage_shape_pow - 1.0) > 1e-9:
    st.sidebar.caption(f"⚠️ Correction active: r → max(r^{damage_shape_pow:.2f}, "
                       f"{damage_floor_pct:.1f}% floor). Design-angle costs now deviate "
                       f"from the given VDE data (by design).")

st.sidebar.markdown("### Economics")
replacement_cost = st.sidebar.slider("Module Replacement Cost ($/W)", 0.20, 0.60, 0.34, 0.01,
                                      help="Base: $0.34/W")
interest_rate = st.sidebar.slider("Discount Rate (%)", 0.0, 15.0, 6.0, 0.25,
                                   help="Base: 6.0% → 15.05× annuity over 40 years")

st.sidebar.markdown("### Insurance")
coverage_ratio = st.sidebar.slider("Coverage Ratio (%)", 100, 150, 125, 5,
                                    help="Multiplier on PML for insured value. Base: 125%")
annual_premium = st.sidebar.slider("Annual Premium Rate (%)", 0.25, 5.0, 1.25, 0.05,
                                    help="Annual insurance premium as % of insured value. Base: 1.25%")

st.sidebar.markdown("### Developer Risk")
risk_pct = st.sidebar.slider("Risk Considered (%)", 0, 100, 50, 5,
                              help="% of below-deductible risk to price in. Default: 50%")

st.sidebar.markdown("### CapEx Premium (¢/W)")
capex_52 = st.sidebar.slider("52° CapEx", 0.0, 10.0, 0.0, 0.1)
capex_60 = st.sidebar.slider("60° CapEx", 0.0, 10.0, 0.0, 0.1)
capex_70 = st.sidebar.slider("70° CapEx", 0.0, 10.0, 1.7, 0.1)
capex_75 = st.sidebar.slider("75° CapEx", 0.0, 10.0, 2.5, 0.1)
capex_77 = st.sidebar.slider("77° CapEx", 0.0, 10.0, 2.5, 0.1)
capex_dict = {52: capex_52, 60: capex_60, 70: capex_70, 75: capex_75, 77: capex_77}

st.sidebar.markdown("### Cost Layers")
ins_on = st.sidebar.checkbox("Insurance", value=True)
risk_on = st.sidebar.checkbox("Developer Risk", value=True)
capex_on = st.sidebar.checkbox("CapEx Premium", value=True)

st.sidebar.markdown("### Choice Model (Luce / Logit)")
sigma = st.sidebar.slider(
    "Sigma (cost sensitivity)", 0.0, 5.0, 2.5, 0.1,
    help="0 = no sensitivity (all equally likely). Higher = more decisive. Default 2.5."
)

st.sidebar.markdown("### Demand Shape (geographic distribution)")
demand_shape = st.sidebar.radio(
    "Where is demand located",
    ["Orennia", "Wood Mackenzie"],
    index=0, horizontal=True, key="shape_src",
    help="Orennia = project-level pipeline locations. WoodMac = state-level evenly distributed."
)

st.sidebar.markdown("### Demand Magnitude (total GW)")
magnitude_source = st.sidebar.radio(
    "Total GW per year from",
    ["Orennia", "Wood Mackenzie", "Manual"],
    index=0, horizontal=True, key="mag_src",
    help="Defaults from the chosen forecast. 'Manual' lets you set everything yourself."
)

# Load both demand sources
_shape_df = load_demand_data(demand_shape)
_orennia_df_full = load_demand_data("Orennia")
_woodmac_df_full = load_demand_data("Wood Mackenzie")

# Derive defaults from the magnitude source (or use Orennia's if Manual)
if magnitude_source == "Orennia":
    _market_defaults = derive_market_defaults(_orennia_df_full)
elif magnitude_source == "Wood Mackenzie":
    _market_defaults = derive_market_defaults(_woodmac_df_full)
else:  # Manual
    _market_defaults = {2026: 36, 2027: 44, 2028: 50, 2029: 55, 2030: 60, 2031: 65, 2032: 70}

st.sidebar.markdown("### Market Size by Year (GWdc)")
st.sidebar.caption(f"Shape: **{demand_shape}**  |  Magnitude: **{magnitude_source}**")

market_sizes = {}
for yr, default_gw in _market_defaults.items():
    market_sizes[yr] = st.sidebar.number_input(
        f"{yr} (data: {default_gw:.1f})", min_value=0.0, max_value=500.0,
        value=float(default_gw), step=1.0, key=f"mkt_{yr}_{magnitude_source}")

layers_active = []
if ins_on: layers_active.append("Ins")
if risk_on: layers_active.append("Risk")
if capex_on: layers_active.append("CapEx")
layers_str = " + ".join(layers_active) if layers_active else "None"

# ─── Angle Visibility Filter (gates which angles are offered above) ───
st.sidebar.markdown("---")
st.sidebar.markdown("### Angle Visibility Filter")
st.sidebar.caption("Controls which tilt angles appear as options at the top of this panel "
                   "(and in the portfolio pickers). Hidden angles stay fully functional in the "
                   "code — just not offered. 70° is hidden by default.")
vis_cols = st.sidebar.columns(len(ANGLES))
for i, a in enumerate(ANGLES):
    with vis_cols[i]:
        st.checkbox(f"{a}°", value=(a != 70), key=f"vis_{a}")


# ═══════════════════════════════════════════
# RENDERING FUNCTIONS
# ═══════════════════════════════════════════

def render_single(df, label="", precomputed=None, suffix='32', angles=None):
    _angles = ACTIVE_ANGLES if angles is None else sorted(angles)
    if precomputed is not None:
        computed = precomputed
    else:
        computed = compute_costs(df, suffix, replacement_cost, coverage_ratio, annual_premium,
                                 interest_rate, risk_pct, capex_dict, ins_on, risk_on, capex_on,
                                 active_angles=_angles,
                                 reliability=stow_reliability, stuck_pct=stuck_pct)

    if label:
        st.markdown(f"#### {label}")

    # ─── Metric Cards (only active angles) ───
    cols = st.columns(max(1, len(_angles)))
    for i, angle in enumerate(_angles):
        avg = computed[f'total_{angle}'].mean()
        c = ANGLE_COLORS[angle]
        with cols[i]:
            st.markdown(f'<div class="metric-card"><h4>{angle}° Avg Total</h4>'
                        f'<div class="value" style="color:rgb({c[0]},{c[1]},{c[2]})">{avg:.3f}</div>'
                        f'<div class="unit">¢/W</div></div>', unsafe_allow_html=True)

    win_counts = computed['best_angle'].value_counts()
    overall_winner = win_counts.idxmax()
    win_pct = win_counts.max() / len(computed) * 100
    angles_str = ", ".join([f"{a}°" for a in _angles])
    st.markdown(f'<div class="winner-banner">🏆 {overall_winner}° wins at {win_counts.max()} of '
                f'{len(computed)} locations ({win_pct:.1f}%) — Layers: {layers_str} — Angles: {angles_str}</div>',
                unsafe_allow_html=True)

    # ─── 3D Column Map (only active angles) ───
    st.markdown("#### 🗺️ Total Cost by Angle (3D Columns)")
    map_rows = []
    for _, row in computed.iterrows():
        for angle in _angles:
            val = row[f'total_{angle}']
            if val > 0:
                map_rows.append({
                    'lat': row['lat'],
                    'lon': row['lon'] + (_angles.index(angle) - (len(_angles)-1)/2) * 0.15,
                    'elevation': val * 50000,
                    'color': ANGLE_COLORS[angle],
                    'angle': angle,
                    'cost_display': f"{val:.3f}",
                })
    map_df = pd.DataFrame(map_rows)
    if len(map_df) > 0:
        col_layer = pdk.Layer("ColumnLayer", data=map_df, get_position='[lon, lat]',
                              get_elevation='elevation', elevation_scale=1, radius=12000,
                              get_fill_color='color', pickable=True, auto_highlight=True)
        states_3d = pdk.Layer("GeoJsonLayer", data=US_STATES_URL,
                              stroked=True, filled=False, pickable=False,
                              get_line_color=[100, 100, 100, 140], line_width_min_pixels=1)
        st.pydeck_chart(pdk.Deck(
            layers=[states_3d, col_layer],
            initial_view_state=pdk.ViewState(latitude=39.0, longitude=-98.0, zoom=3.8, pitch=45),
            map_style="light",
            tooltip={"html": "<b>{angle}°</b>: {cost_display} ¢/W",
                     "style": {"backgroundColor": "#1e293b", "color": "#e2e8f0",
                                "fontSize": "13px", "padding": "8px 12px", "borderRadius": "8px"}},
        ), use_container_width=True, height=500)
        st.markdown(angle_legend(_angles), unsafe_allow_html=True)

    # ─── Interpolated Optimal Angle Map ───
    st.markdown("#### 🎯 Optimal Angle by Location")
    computed['color'] = computed['best_angle'].map(ANGLE_COLORS)
    computed['best_cost_display'] = computed['best_cost'].round(3).astype(str)

    grid_df = interpolate_best_product(computed['lat'].values, computed['lon'].values,
                                        computed['best_angle'].values)

    # Background: interpolated fill (not pickable — tooltip comes from foreground only)
    bg_layer = pdk.Layer("ScatterplotLayer", data=grid_df, get_position='[lon, lat]',
                         get_fill_color='color', get_radius=9000, pickable=False,
                         opacity=0.45)

    # Foreground: actual data points (pickable for tooltip)
    fg_layer = pdk.Layer("ScatterplotLayer", data=computed, get_position='[lon, lat]',
                         get_fill_color='color', get_radius=7000, pickable=True,
                         auto_highlight=True, opacity=0.85)

    # State borders drawn last so they sit crisply on top
    states_layer = pdk.Layer(
        "GeoJsonLayer", data=US_STATES_URL,
        stroked=True, filled=False, pickable=False,
        get_line_color=[70, 70, 70, 200], line_width_min_pixels=1,
    )

    st.pydeck_chart(pdk.Deck(
        layers=[bg_layer, fg_layer, states_layer],
        initial_view_state=pdk.ViewState(latitude=39.0, longitude=-96.0, zoom=4.0, pitch=0),
        map_style="light",
        tooltip={"html": "<b>Best:</b> {best_angle}° — {best_cost_display} ¢/W",
                 "style": {"backgroundColor": "#1e293b", "color": "#e2e8f0",
                            "fontSize": "13px", "padding": "8px 12px", "borderRadius": "8px"}},
    ), use_container_width=True, height=520)
    st.markdown(angle_legend(_angles), unsafe_allow_html=True)

    # ─── Win Count Table (active angles only) ───
    st.markdown("#### 📊 Win Counts")
    wc = pd.DataFrame({
        'Angle': [f'{a}°' for a in _angles],
        'Wins': [win_counts.get(a, 0) for a in _angles],
        'Win %': [f"{win_counts.get(a, 0)/len(computed)*100:.1f}%" for a in _angles],
        'Avg Total (¢/W)': [f"{computed[f'total_{a}'].mean():.4f}" for a in _angles],
    })
    st.dataframe(wc, hide_index=True, use_container_width=True)

    return computed


# ─── Market Share & Demand Analysis ───
def render_market_share(computed, shape_df, label=""):
    """Market share by tilt angle with multi-year select, shape/magnitude split, and export."""
    st.markdown("---")
    st.markdown("### 📈 Market Share by Tilt Angle"
                + (f" — {label}" if label else "")
                + f"  [shape: {demand_shape}, magnitude: {magnitude_source}]")

    has_shape = shape_df is not None
    years = sorted(market_sizes.keys())

    # ─── Multi-year selector ───
    st.markdown("**Years to include** (select one or more):")
    yr_cols = st.columns(min(len(years), 8))
    year_selected = {}
    for i, yr in enumerate(years):
        with yr_cols[i % len(yr_cols)]:
            year_selected[yr] = st.checkbox(str(yr), value=True, key=f"yrsel_{yr}_{label}")
    selected_years = [yr for yr in years if year_selected[yr]]

    if not selected_years:
        st.warning("Select at least one year.")
        return

    total_gw = sum(market_sizes[yr] for yr in selected_years)
    total_mw = total_gw * 1000
    yr_label = (f"{min(selected_years)}–{max(selected_years)}"
                if len(selected_years) > 1 else str(selected_years[0]))

    # ─── Build location-level scaled demand from shape source ───
    merged = None
    use_shape = False
    if has_shape:
        shape_sub = shape_df[shape_df['Year'].isin(selected_years)].copy()
        if len(shape_sub) > 0:
            demand_by_loc = shape_sub.groupby(['hail_lat', 'hail_lon']).agg(
                total_mw=('DC Capacity (MW)', 'sum')).reset_index()
            merged = demand_by_loc.merge(
                computed[['lat', 'lon', 'best_angle']],
                left_on=['hail_lat', 'hail_lon'], right_on=['lat', 'lon'], how='inner')
            if len(merged) > 0:
                shape_total = merged['total_mw'].sum()
                scale = total_mw / shape_total if shape_total > 0 else 1.0
                merged['scaled_mw'] = merged['total_mw'] * scale
                angle_summary = merged.groupby('best_angle')['scaled_mw'].sum().reset_index()
                angle_summary.columns = ['Best Product', 'MWdc']
                data_source = f"Shape: {demand_shape} (scaled to {total_gw:.1f} GW)"
                use_shape = True

    if not use_shape:
        merged = None
        n_locations = len(computed)
        mw_per_loc = total_mw / n_locations if n_locations > 0 else 0
        angle_summary = computed.groupby('best_angle').size().reset_index(name='count')
        angle_summary['MWdc'] = angle_summary['count'] * mw_per_loc
        angle_summary = angle_summary[['best_angle', 'MWdc']]
        angle_summary.columns = ['Best Product', 'MWdc']
        data_source = "Uniform distribution"

    # Ensure all active angles appear
    for a in ACTIVE_ANGLES:
        if a not in angle_summary['Best Product'].values:
            angle_summary = pd.concat([angle_summary,
                                       pd.DataFrame({'Best Product': [a], 'MWdc': [0]})],
                                      ignore_index=True)
    angle_summary = angle_summary[angle_summary['Best Product'].isin(ACTIVE_ANGLES)].copy()
    angle_summary = angle_summary.sort_values('Best Product')
    angle_summary['GWdc'] = (angle_summary['MWdc'] / 1000).round(2)
    total_for_share = angle_summary['MWdc'].sum()
    angle_summary['Share (%)'] = (angle_summary['MWdc'] / total_for_share * 100).round(1) if total_for_share > 0 else 0
    angle_summary['Best Product'] = angle_summary['Best Product'].astype(str) + '°'

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"**{yr_label}**  —  Total Market: **{total_gw:.1f} GWdc**")
        st.caption(f"Source: {data_source}")
        display_df = angle_summary[['Best Product', 'GWdc', 'MWdc', 'Share (%)']].copy()
        display_df['MWdc'] = display_df['MWdc'].round(0).astype(int)
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    with c2:
        chart_df = angle_summary.set_index('Best Product')[['GWdc']]
        st.bar_chart(chart_df, use_container_width=True, height=300)

    # ─── Export button ───
    export_df = build_export(display_df, selected_years, total_gw, label)
    csv_bytes = export_df.to_csv(index=False).encode()
    st.download_button(
        "📥 Export Summary CSV", csv_bytes,
        file_name=f"hail_summary_{label.replace(' ', '_')}_{yr_label}.csv",
        mime="text/csv", key=f"export_{label}",
    )

    # ─── 3D Demand Map ───
    st.markdown("#### 🏗️ Demand by Location (MWdc)")
    if merged is not None and len(merged) > 0:
        map_demand = merged[['lat', 'lon', 'best_angle', 'scaled_mw']].copy()
        map_demand.rename(columns={'scaled_mw': 'mw'}, inplace=True)
    else:
        map_demand = computed[['lat', 'lon', 'best_angle']].copy()
        n_locs = len(map_demand)
        map_demand['mw'] = total_mw / n_locs if n_locs > 0 else 0

    map_demand = map_demand[(map_demand['mw'] > 0) &
                             (map_demand['best_angle'].isin(ACTIVE_ANGLES))].copy()

    if len(map_demand) > 0:
        map_demand['color'] = map_demand['best_angle'].map(ANGLE_COLORS)
        map_demand['elevation'] = map_demand['mw'] * 200
        map_demand['mw_display'] = map_demand['mw'].round(0).astype(int).astype(str)
        map_demand['angle_display'] = map_demand['best_angle'].astype(str) + '°'

        demand_col_layer = pdk.Layer(
            "ColumnLayer", data=map_demand, get_position='[lon, lat]',
            get_elevation='elevation', elevation_scale=1, radius=18000,
            get_fill_color='color', pickable=True, auto_highlight=True,
        )
        states_demand = pdk.Layer("GeoJsonLayer", data=US_STATES_URL,
                                  stroked=True, filled=False, pickable=False,
                                  get_line_color=[100, 100, 100, 140], line_width_min_pixels=1)
        st.pydeck_chart(pdk.Deck(
            layers=[states_demand, demand_col_layer],
            initial_view_state=pdk.ViewState(latitude=39.0, longitude=-98.0, zoom=3.8, pitch=45),
            map_style="light",
            tooltip={"html": "<b>{angle_display}</b> — {mw_display} MWdc",
                     "style": {"backgroundColor": "#1e293b", "color": "#e2e8f0",
                                "fontSize": "13px", "padding": "8px 12px", "borderRadius": "8px"}},
        ), use_container_width=True, height=500)
        st.markdown(angle_legend(ACTIVE_ANGLES), unsafe_allow_html=True)
    else:
        st.info("No demand data to display for the selected years.")

    # ─── Year-by-Year Breakdown ───
    if len(selected_years) > 1:
        st.markdown("#### Year-by-Year Breakdown")
        yoy_rows = []
        for yr in selected_years:
            yr_mw = market_sizes[yr] * 1000
            if has_shape:
                yr_shape = shape_df[shape_df['Year'] == yr]
                if len(yr_shape) > 0:
                    dbl = yr_shape.groupby(['hail_lat', 'hail_lon']).agg(
                        total_mw=('DC Capacity (MW)', 'sum')).reset_index()
                    mrg = dbl.merge(computed[['lat', 'lon', 'best_angle']],
                                    left_on=['hail_lat', 'hail_lon'],
                                    right_on=['lat', 'lon'], how='inner')
                    if len(mrg) > 0:
                        sc = yr_mw / mrg['total_mw'].sum() if mrg['total_mw'].sum() > 0 else 1
                        mrg['scaled_mw'] = mrg['total_mw'] * sc
                        for a in ACTIVE_ANGLES:
                            sub = mrg[mrg['best_angle'] == a]
                            yoy_rows.append({'Year': yr, 'Angle': f'{a}°',
                                             'GWdc': round(sub['scaled_mw'].sum() / 1000, 2)})
                        continue
            # Uniform fallback for this year
            counts = computed['best_angle'].value_counts()
            n = len(computed)
            mw_per = yr_mw / n if n > 0 else 0
            for a in ACTIVE_ANGLES:
                cnt = counts.get(a, 0)
                yoy_rows.append({'Year': yr, 'Angle': f'{a}°',
                                 'GWdc': round(cnt * mw_per / 1000, 2)})
        if yoy_rows:
            yoy_df = pd.DataFrame(yoy_rows)
            pivot = yoy_df.pivot_table(index='Angle', columns='Year', values='GWdc',
                                       aggfunc='sum').fillna(0)
            st.dataframe(pivot, use_container_width=True)


def build_export(market_table, selected_years, total_gw, label):
    """Build a single CSV-ready export of inputs and outputs."""
    rows = []
    rows.append({'Section': 'Inputs', 'Key': 'Glass Type', 'Value': glass_choice})
    if glass_choice == "Blended":
        rows.append({'Section': 'Inputs', 'Key': '2.0 mm Share (%)', 'Value': glass_pct_20})
    rows.append({'Section': 'Inputs', 'Key': 'Tilt Angles Active',
                 'Value': ', '.join([f'{a}°' for a in ACTIVE_ANGLES])})
    rows.append({'Section': 'Inputs', 'Key': 'Damage Decay Sharpness (power)', 'Value': damage_shape_pow})
    rows.append({'Section': 'Inputs', 'Key': 'Damage Floor (% of flat)', 'Value': damage_floor_pct})
    rows.append({'Section': 'Inputs', 'Key': 'Stow Reliability (%)', 'Value': stow_reliability})
    rows.append({'Section': 'Inputs', 'Key': 'Failed Trackers At (% of design angle)', 'Value': stuck_pct})
    rows.append({'Section': 'Inputs', 'Key': 'Replacement Cost ($/W)', 'Value': replacement_cost})
    rows.append({'Section': 'Inputs', 'Key': 'Discount Rate (%)', 'Value': interest_rate})
    rows.append({'Section': 'Inputs', 'Key': 'Coverage Ratio (%)', 'Value': coverage_ratio})
    rows.append({'Section': 'Inputs', 'Key': 'Annual Premium (%)', 'Value': annual_premium})
    rows.append({'Section': 'Inputs', 'Key': 'Dev Risk Considered (%)', 'Value': risk_pct})
    rows.append({'Section': 'Inputs', 'Key': '52° CapEx (¢/W)', 'Value': capex_52})
    rows.append({'Section': 'Inputs', 'Key': '60° CapEx (¢/W)', 'Value': capex_60})
    rows.append({'Section': 'Inputs', 'Key': '70° CapEx (¢/W)', 'Value': capex_70})
    rows.append({'Section': 'Inputs', 'Key': '77° CapEx (¢/W)', 'Value': capex_77})
    rows.append({'Section': 'Inputs', 'Key': 'Insurance Layer',
                 'Value': 'On' if ins_on else 'Off'})
    rows.append({'Section': 'Inputs', 'Key': 'Dev Risk Layer',
                 'Value': 'On' if risk_on else 'Off'})
    rows.append({'Section': 'Inputs', 'Key': 'CapEx Layer',
                 'Value': 'On' if capex_on else 'Off'})
    rows.append({'Section': 'Inputs', 'Key': 'Demand Shape', 'Value': demand_shape})
    rows.append({'Section': 'Inputs', 'Key': 'Demand Magnitude Source', 'Value': magnitude_source})
    rows.append({'Section': 'Inputs', 'Key': 'Sigma (Luce sensitivity)', 'Value': sigma})
    rows.append({'Section': 'Inputs', 'Key': 'Years Selected',
                 'Value': ', '.join(str(y) for y in selected_years)})
    rows.append({'Section': 'Inputs', 'Key': 'Total Market (GWdc)', 'Value': round(total_gw, 2)})
    for yr in selected_years:
        rows.append({'Section': 'Market Size', 'Key': f'{yr} (GWdc)',
                     'Value': market_sizes[yr]})

    for _, mr in market_table.iterrows():
        rows.append({'Section': 'Market Share',
                     'Key': f"{mr['Best Product']} GWdc", 'Value': mr['GWdc']})
        rows.append({'Section': 'Market Share',
                     'Key': f"{mr['Best Product']} MWdc", 'Value': mr['MWdc']})
        rows.append({'Section': 'Market Share',
                     'Key': f"{mr['Best Product']} Share (%)", 'Value': mr['Share (%)']})

    return pd.DataFrame(rows)


# ─── Value Gap Analysis (portfolio baseline) ───
def render_value_gap(computed, label=""):
    st.markdown("---")
    st.markdown("### 💰 Value Gap Analysis" + (f" — {label}" if label else ""))
    st.markdown("The **base case** is the full set of active tilt angles (always picks cheapest). "
                "Selecting fewer products shows the added cost of limiting your portfolio.")

    gap_cols = st.columns(max(1, len(ACTIVE_ANGLES)))
    selected = []
    for i, angle in enumerate(ACTIVE_ANGLES):
        with gap_cols[i]:
            if st.checkbox(f"{angle}°", value=True, key=f"vg_{angle}_{label}"):
                selected.append(angle)

    if len(selected) == 0:
        st.warning("Select at least one product angle.")
        return

    # Base case: full active portfolio
    computed['base_cost'] = computed[[f'total_{a}' for a in ACTIVE_ANGLES]].min(axis=1)
    # Subset
    computed['subset_cost'] = computed[[f'total_{a}' for a in selected]].min(axis=1)
    computed['gap'] = computed['subset_cost'] - computed['base_cost']

    max_gap = computed['gap'].quantile(0.95) if computed['gap'].max() > 0 else 1.0
    computed['gap_norm'] = (computed['gap'] / max_gap).clip(0, 1) if max_gap > 0 else 0.0
    computed['gap_color'] = computed['gap_norm'].apply(
        lambda g: [int(239 * (1 - g) + 34 * g), int(68 * (1 - g) + 197 * g),
                    int(68 * (1 - g) + 94 * g)])
    computed['gap_display'] = computed['gap'].round(3).astype(str)
    computed['base_display'] = computed['base_cost'].round(3).astype(str)
    computed['subset_display'] = computed['subset_cost'].round(3).astype(str)

    states_layer = pdk.Layer("GeoJsonLayer", data=US_STATES_URL,
                             stroked=True, filled=False, pickable=False,
                             get_line_color=[100, 100, 100, 180], line_width_min_pixels=1)

    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer", data=computed, get_position='[lon, lat]',
                          get_fill_color='gap_color', get_radius=30000,
                          pickable=True, auto_highlight=True),
                states_layer],
        initial_view_state=pdk.ViewState(latitude=39.0, longitude=-98.0, zoom=3.8, pitch=0),
        map_style="light",
        tooltip={"html": ("<b>Full portfolio:</b> {base_display} ¢/W<br>"
                           "<b>Subset:</b> {subset_display} ¢/W<br>"
                           "<b>Added cost:</b> {gap_display} ¢/W"),
                 "style": {"backgroundColor": "#1e293b", "color": "#e2e8f0",
                            "fontSize": "13px", "padding": "8px 12px", "borderRadius": "8px"}},
    ), use_container_width=True, height=500)

    st.markdown(f"**Avg added cost:** {computed['gap'].mean():.4f} ¢/W  |  "
                f"**Median:** {computed['gap'].median():.4f} ¢/W  |  "
                f"**Max:** {computed['gap'].max():.4f} ¢/W")
    st.markdown("🟢 Green = large gap (clear full-portfolio advantage) → 🔴 Red = small gap (subset nearly as good)")


# ─── Site Lookup ───
def render_lookup(df, computed, label="", suffix_lk='32', rel_lk=None, stuck_lk=None, angles_lk=None):
    _rel = stow_reliability if rel_lk is None else rel_lk
    _stk = stuck_pct if stuck_lk is None else stuck_lk
    _angles_lk = ACTIVE_ANGLES if angles_lk is None else sorted(angles_lk)
    st.markdown("---")
    st.markdown("### 📍 Site Lookup" + (f" — {label}" if label else ""))

    lk1, lk2 = st.columns(2)
    with lk1:
        input_lat = st.number_input("Latitude", 24.0, 50.0, 33.0, 0.1, key=f"lat_{label}")
    with lk2:
        input_lon = st.number_input("Longitude", -125.0, -66.0, -97.0, 0.1, key=f"lon_{label}")

    if st.button("🔍 Lookup", key=f"lookup_{label}"):
        points = df[['lat', 'lon']].values
        results, breakdowns = {}, {}
        for angle in ANGLES:
            try:
                interp = CloughTocher2DInterpolator(points, computed[f'total_{angle}'].values)
                val = interp(input_lat, input_lon)
                if np.isnan(val):
                    val = LinearNDInterpolator(points, computed[f'total_{angle}'].values)(input_lat, input_lon)
                results[angle] = float(val) if not np.isnan(val) else None
            except Exception:
                results[angle] = None

            bd = {}
            for layer_name, prefix in [('Insurance', 'ins'), ('Dev Risk', 'risk')]:
                try:
                    interp = CloughTocher2DInterpolator(points, computed[f'{prefix}_{angle}'].values)
                    v = interp(input_lat, input_lon)
                    if np.isnan(v):
                        v = LinearNDInterpolator(points, computed[f'{prefix}_{angle}'].values)(input_lat, input_lon)
                    bd[layer_name] = float(v) if not np.isnan(v) else 0.0
                except Exception:
                    bd[layer_name] = 0.0
            bd['CapEx'] = capex_dict[angle] if capex_on else 0.0
            breakdowns[angle] = bd

        valid = {a: v for a, v in results.items() if v is not None}
        if not valid:
            st.warning("⚠️ Outside data coverage. Try coordinates within the contiguous US.")
        else:
            best_angle = min(valid, key=valid.get)
            best_cost = valid[best_angle]
            sorted_angles = sorted(valid.items(), key=lambda x: x[1])

            html = f'<div class="lookup-result"><h3>📍 ({input_lat:.1f}, {input_lon:.1f})</h3>'
            html += f'<p style="color:#34d399;font-size:1.2rem;font-weight:700;">🏆 Best: {best_angle}° at {best_cost:.4f} ¢/W</p>'
            html += '<table style="width:100%;border-collapse:collapse;color:#e2e8f0;">'
            html += '<tr style="border-bottom:2px solid #475569;"><th style="text-align:left;padding:8px;">Angle</th>'
            html += '<th style="text-align:right;padding:8px;">Insurance</th><th style="text-align:right;padding:8px;">Dev Risk</th>'
            html += '<th style="text-align:right;padding:8px;">CapEx</th><th style="text-align:right;padding:8px;">Total</th>'
            html += '<th style="text-align:right;padding:8px;">vs Best</th></tr>'
            for angle, cost in sorted_angles:
                is_best = angle == best_angle
                s = 'color:#34d399;font-weight:700;' if is_best else ''
                bd = breakdowns[angle]
                delta = "—" if is_best else f"+{cost - best_cost:.4f}"
                mark = " ✓" if is_best else ""
                html += f'<tr style="border-bottom:1px solid #334155;{s}"><td style="padding:8px;">{angle}°{mark}</td>'
                html += f'<td style="text-align:right;padding:8px;">{bd["Insurance"]:.4f}</td>'
                html += f'<td style="text-align:right;padding:8px;">{bd["Dev Risk"]:.4f}</td>'
                html += f'<td style="text-align:right;padding:8px;">{bd["CapEx"]:.1f}</td>'
                html += f'<td style="text-align:right;padding:8px;">{cost:.4f}</td>'
                html += f'<td style="text-align:right;padding:8px;">{delta}</td></tr>'
            html += '</table></div>'
            st.markdown(html, unsafe_allow_html=True)

            # ── Continuous damage curve at this site ──
            if suffix_lk in ('20', '32'):
                st.markdown("##### 📉 Site Damage Curves (continuous, into-wind)")
                thetas = np.arange(0, 77.5, 0.5)
                anchor_pts = {}
                for m in ['pml', 'aal']:
                    for a in ANCHOR_ANGLES:
                        try:
                            it = CloughTocher2DInterpolator(points, df[f'{m}_{a}'].values)
                            v = it(input_lat, input_lon)
                            if np.isnan(v):
                                v = LinearNDInterpolator(points, df[f'{m}_{a}'].values)(input_lat, input_lon)
                            anchor_pts[(m, a)] = max(float(v), 0.0) if not np.isnan(v) else 0.0
                        except Exception:
                            anchor_pts[(m, a)] = 0.0
                site_df = pd.DataFrame({f'{m}_{a}': [anchor_pts[(m, a)]]
                                        for m in ['pml', 'aal'] for a in ANCHOR_ANGLES})
                _p_flat = np.array([anchor_pts[('pml', 0)]])
                _a_flat = np.array([anchor_pts[('aal', 0)]])
                pml_c = [float(apply_curve_correction(curve_value(site_df, th, 'pml', suffix_lk),
                                                      _p_flat, damage_floor_pct, damage_shape_pow)[0]) * 100
                         for th in thetas]
                aal_c = [float(apply_curve_correction(curve_value(site_df, th, 'aal', suffix_lk),
                                                      _a_flat, damage_floor_pct, damage_shape_pow)[0]) * 100
                         for th in thetas]
                curve_df = pd.DataFrame({'Tilt Angle (°)': thetas,
                                         'PML (%)': pml_c, 'AAL (%)': aal_c}).set_index('Tilt Angle (°)')
                st.line_chart(curve_df, height=300)
                if _rel < 100:
                    stuck_rows = []
                    for a in _angles_lk:
                        th_s = (_stk / 100.0) * a
                        stuck_rows.append({
                            'Design Angle': f'{a}°',
                            'Failed-Tracker Angle': f'{th_s:.1f}°',
                            'PML @ stuck (%)': round(float(apply_curve_correction(
                                curve_value(site_df, th_s, 'pml', suffix_lk), _p_flat,
                                damage_floor_pct, damage_shape_pow)[0]) * 100, 2),
                            'PML @ design (%)': round(float(apply_curve_correction(
                                np.array([anchor_pts[('pml', a)]]), _p_flat,
                                damage_floor_pct, damage_shape_pow)[0]) * 100, 2),
                            'AAL @ stuck (%)': round(float(apply_curve_correction(
                                curve_value(site_df, th_s, 'aal', suffix_lk), _a_flat,
                                damage_floor_pct, damage_shape_pow)[0]) * 100, 3),
                            'AAL @ design (%)': round(float(apply_curve_correction(
                                np.array([anchor_pts[('aal', a)]]), _a_flat,
                                damage_floor_pct, damage_shape_pow)[0]) * 100, 3),
                        })
                    st.markdown(f"**Failed trackers ({100 - _rel:.1f}% of fleet) "
                                f"at {_stk}% of design angle:**")
                    st.dataframe(pd.DataFrame(stuck_rows), hide_index=True, use_container_width=True)


# ─── Luce/Logit Probabilistic Market Share ───
def render_luce_market_share(computed, shape_df, label=""):
    """Probability-weighted market share using the Luce/Logit choice model.

    Shares the same multi-year selection and shape/magnitude controls as the
    deterministic market share section (read from module-level state).
    """
    st.markdown("---")
    st.markdown("### 🎲 Probabilistic Market Share (Luce Model)"
                + (f" — {label}" if label else "")
                + f"  [σ = {sigma:.2f}, shape: {demand_shape}, magnitude: {magnitude_source}]")
    st.caption("Each location splits its demand probabilistically across active tilt angles using "
               "P(a) = exp(−σ·ΔCost_a) / Σ exp(−σ·ΔCost_j), where ΔCost is normalized to the "
               "per-location minimum. Higher σ = more decisive; σ=0 = uniform.")

    # Reuse the same year selection state as deterministic market share
    has_shape = shape_df is not None
    years = sorted(market_sizes.keys())
    selected_years = [yr for yr in years if st.session_state.get(f"yrsel_{yr}_{label}", True)]

    if not selected_years:
        st.warning("Select at least one year in the deterministic market share section above.")
        return None

    total_gw = sum(market_sizes[yr] for yr in selected_years)
    total_mw = total_gw * 1000
    yr_label = (f"{min(selected_years)}–{max(selected_years)}"
                if len(selected_years) > 1 else str(selected_years[0]))

    # Compute Luce probabilities
    luce = compute_luce_probabilities(computed, sigma, ACTIVE_ANGLES)

    # Build location-level scaled demand using the shape source
    merged_demand = None
    if has_shape:
        shape_sub = shape_df[shape_df['Year'].isin(selected_years)]
        if len(shape_sub) > 0:
            demand_by_loc = shape_sub.groupby(['hail_lat', 'hail_lon']).agg(
                total_mw=('DC Capacity (MW)', 'sum')).reset_index()
            merge_cols = ['lat', 'lon'] + [f'prob_{a}' for a in ANGLES]
            merged_demand = demand_by_loc.merge(
                luce[merge_cols],
                left_on=['hail_lat', 'hail_lon'], right_on=['lat', 'lon'], how='inner')
            if len(merged_demand) > 0:
                shape_total = merged_demand['total_mw'].sum()
                scale = total_mw / shape_total if shape_total > 0 else 1.0
                merged_demand['scaled_mw'] = merged_demand['total_mw'] * scale

    if merged_demand is None or len(merged_demand) == 0:
        # Uniform fallback
        mw_per_loc = total_mw / len(luce) if len(luce) > 0 else 0
        merged_demand = luce[['lat', 'lon'] + [f'prob_{a}' for a in ANGLES]].copy()
        merged_demand['scaled_mw'] = mw_per_loc
        data_source = "Uniform distribution"
    else:
        data_source = f"Shape: {demand_shape} (scaled to {total_gw:.1f} GW)"

    # Probability-adjusted market share by angle: sum across locations of (scaled_mw * P_a)
    angle_rows = []
    for a in ACTIVE_ANGLES:
        mw_a = (merged_demand['scaled_mw'] * merged_demand[f'prob_{a}']).sum()
        angle_rows.append({'Best Product': f'{a}°', 'MWdc': mw_a})
    angle_summary = pd.DataFrame(angle_rows)
    angle_summary['GWdc'] = (angle_summary['MWdc'] / 1000).round(2)
    tot = angle_summary['MWdc'].sum()
    angle_summary['Share (%)'] = (angle_summary['MWdc'] / tot * 100).round(1) if tot > 0 else 0
    angle_summary['MWdc'] = angle_summary['MWdc'].round(0).astype(int)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"**{yr_label}**  —  Total Market: **{total_gw:.1f} GWdc**")
        st.caption(f"Source: {data_source}")
        st.dataframe(angle_summary[['Best Product', 'GWdc', 'MWdc', 'Share (%)']],
                     hide_index=True, use_container_width=True)
    with c2:
        chart_df = angle_summary.set_index('Best Product')[['GWdc']]
        st.bar_chart(chart_df, use_container_width=True, height=300)

    # Year-by-year probabilistic breakdown
    if len(selected_years) > 1:
        st.markdown("#### Year-by-Year Probabilistic Breakdown")
        yoy_rows = []
        for yr in selected_years:
            yr_mw = market_sizes[yr] * 1000
            if has_shape:
                yr_shape = shape_df[shape_df['Year'] == yr]
                if len(yr_shape) > 0:
                    dbl = yr_shape.groupby(['hail_lat', 'hail_lon']).agg(
                        total_mw=('DC Capacity (MW)', 'sum')).reset_index()
                    mrg = dbl.merge(luce[['lat', 'lon'] + [f'prob_{a}' for a in ANGLES]],
                                    left_on=['hail_lat', 'hail_lon'],
                                    right_on=['lat', 'lon'], how='inner')
                    if len(mrg) > 0:
                        sc = yr_mw / mrg['total_mw'].sum() if mrg['total_mw'].sum() > 0 else 1
                        mrg['scaled_mw'] = mrg['total_mw'] * sc
                        for a in ACTIVE_ANGLES:
                            mw_a = (mrg['scaled_mw'] * mrg[f'prob_{a}']).sum()
                            yoy_rows.append({'Year': yr, 'Angle': f'{a}°',
                                             'GWdc': round(mw_a / 1000, 2)})
                        continue
            # Uniform fallback
            mw_per = yr_mw / len(luce)
            for a in ACTIVE_ANGLES:
                mw_a = (mw_per * luce[f'prob_{a}']).sum()
                yoy_rows.append({'Year': yr, 'Angle': f'{a}°', 'GWdc': round(mw_a / 1000, 2)})
        if yoy_rows:
            yoy_df = pd.DataFrame(yoy_rows)
            pivot = yoy_df.pivot_table(index='Angle', columns='Year', values='GWdc',
                                       aggfunc='sum').fillna(0)
            st.dataframe(pivot, use_container_width=True)

    return {'luce': luce, 'merged_demand': merged_demand,
            'angle_summary': angle_summary, 'selected_years': selected_years,
            'total_gw': total_gw}


def render_luce_demand_map(luce_result, label=""):
    """3D column map of win probabilities — bar height = P(win), color = tilt angle."""
    if luce_result is None:
        return
    st.markdown("---")
    st.markdown("### 🎲 Win Probability by Location (3D Columns)" + (f" — {label}" if label else ""))
    st.markdown("Each location shows a column per active tilt angle. **Bar height** = Luce probability "
                "that the angle wins at that location (against the full active-angle market). "
                "**Color** = the tilt angle. Toggle angles below to focus the view.")

    # Checkboxes select which angles' columns to display
    cb_cols = st.columns(max(1, len(ACTIVE_ANGLES)))
    selected_angles = []
    for i, angle in enumerate(ACTIVE_ANGLES):
        with cb_cols[i]:
            if st.checkbox(f"{angle}°", value=True, key=f"luce_{angle}_{label}"):
                selected_angles.append(angle)

    if len(selected_angles) == 0:
        st.info("Select at least one tilt angle to display.")
        return

    luce = luce_result['luce']  # has prob_{angle} columns

    # Build one column per location per selected angle, offset side-by-side
    map_rows = []
    n_sel = len(selected_angles)
    for _, row in luce.iterrows():
        for j, angle in enumerate(selected_angles):
            p = row[f'prob_{angle}']
            if p > 0:
                map_rows.append({
                    'lat': row['lat'],
                    'lon': row['lon'] + (j - (n_sel - 1) / 2) * 0.15,
                    'elevation': p * 800000,  # scale probability (0-1) for visibility
                    'color': ANGLE_COLORS[angle],
                    'angle': angle,
                    'prob_display': f"{p*100:.1f}%",
                })
    map_df = pd.DataFrame(map_rows)

    if len(map_df) == 0:
        st.info("No probability data to display.")
        return

    col_layer = pdk.Layer(
        "ColumnLayer", data=map_df, get_position='[lon, lat]',
        get_elevation='elevation', elevation_scale=1, radius=12000,
        get_fill_color='color', pickable=True, auto_highlight=True,
    )
    states_layer = pdk.Layer("GeoJsonLayer", data=US_STATES_URL,
                             stroked=True, filled=False, pickable=False,
                             get_line_color=[100, 100, 100, 140], line_width_min_pixels=1)
    st.pydeck_chart(pdk.Deck(
        layers=[states_layer, col_layer],
        initial_view_state=pdk.ViewState(latitude=39.0, longitude=-98.0, zoom=3.8, pitch=45),
        map_style="light",
        tooltip={"html": "<b>{angle}°</b> — win probability {prob_display}",
                 "style": {"backgroundColor": "#1e293b", "color": "#e2e8f0",
                            "fontSize": "13px", "padding": "8px 12px", "borderRadius": "8px"}},
    ), use_container_width=True, height=500)

    # Legend + summary
    legend = "  ".join(
        f'<span style="color:rgb({ANGLE_COLORS[a][0]},{ANGLE_COLORS[a][1]},{ANGLE_COLORS[a][2]});'
        f'font-weight:700">■ {a}°</span>'
        for a in selected_angles)
    st.markdown(legend, unsafe_allow_html=True)
    avg_str = "  |  ".join(
        f"{a}° avg P: {luce[f'prob_{a}'].mean()*100:.1f}%" for a in selected_angles)
    st.markdown(f"**{avg_str}**")



def render_portfolio_comparison(compA, compB, relA, relB, stuckA, stuckB, shape_df,
                                glass_lbl="", anglesA=None, anglesB=None,
                                comp100=None, base_df=None):
    """A-vs-B comparison sheet: the value of stow reliability across portfolio angle sets."""
    anglesA = sorted(ACTIVE_ANGLES if anglesA is None else anglesA)
    anglesB = sorted(ACTIVE_ANGLES if anglesB is None else anglesB)
    angA_str = "/".join(f"{a}°" for a in anglesA)
    angB_str = "/".join(f"{a}°" for a in anglesB)

    st.markdown(f"### ⚖️ Value of Stow Reliability — A [{angA_str}] {relA:.1f}% "
                f"vs B [{angB_str}] {relB:.1f}%"
                + (f" — {glass_lbl}" if glass_lbl else ""))
    st.caption(f"Portfolio A: **{angA_str}** trackers, **{relA:.1f}%** reach full stow "
               f"(failures at **{stuckA}%** of design angle). "
               f"Portfolio B: **{angB_str}** trackers, **{relB:.1f}%** (failures at **{stuckB}%**). "
               f"**A Advantage = B cost − A cost**: positive means A is cheaper.")

    # ── Rank-matched head-to-head pairs (highest vs highest, lowest vs lowest) ──
    sA, sB = sorted(anglesA, reverse=True), sorted(anglesB, reverse=True)
    npairs = min(len(sA), len(sB))
    pairs = [(sA[i], sB[i]) for i in range(npairs)][::-1]   # display low → high
    extraA, extraB = sorted(sA[npairs:]), sorted(sB[npairs:])

    cols = st.columns(max(1, len(pairs)))
    for i, (aA, aB) in enumerate(pairs):
        adv_p = (compB[f'total_{aB}'] - compA[f'total_{aA}']).mean()
        color = '#059669' if adv_p >= 0 else '#dc2626'
        with cols[i]:
            st.markdown(f'<div class="metric-card"><h4>A {aA}° vs B {aB}°</h4>'
                        f'<div class="value" style="color:{color}">{adv_p:+.4f}</div>'
                        f'<div class="unit">¢/W A advantage (B − A)</div></div>',
                        unsafe_allow_html=True)

    # ── Portfolio-weighted (best-within-portfolio) advantage ──
    bestA = compA['best_cost']
    bestB = compB['best_cost']
    adv = bestB - bestA
    st.markdown(f'<div class="winner-banner">⚖️ Portfolio advantage — A [{angA_str}] over '
                f'B [{angB_str}] (each location buys its cheapest tracker from each portfolio) — '
                f'avg: {adv.mean():+.4f} ¢/W  |  median: {adv.median():+.4f}  |  '
                f'max: {adv.max():+.4f}  |  A cheaper at {(adv > 1e-12).sum()} of {len(adv)} '
                f'locations</div>', unsafe_allow_html=True)

    # ── Advantage map ──
    st.markdown("#### 🗺️ Where Reliability Matters (portfolio cost advantage)")
    mp = compA[['lat', 'lon']].copy()
    mp['adv'] = adv.values
    mp['bestA'] = bestA.round(3).astype(str)
    mp['bestB'] = bestB.round(3).astype(str)
    mp['adv_disp'] = mp['adv'].round(3).astype(str)
    scale = max(float(mp['adv'].abs().quantile(0.95)), 1e-9)

    def _adv_color(v):
        g = min(abs(v) / scale, 1.0)
        if v >= 0:   # A cheaper → green
            return [int(226 - 192 * g), int(232 - 35 * g), int(240 - 146 * g)]
        return [int(226 + 13 * g), int(232 - 164 * g), int(240 - 172 * g)]  # B cheaper → red

    mp['color'] = mp['adv'].apply(_adv_color)
    states_cmp = pdk.Layer("GeoJsonLayer", data=US_STATES_URL, stroked=True, filled=False,
                           pickable=False, get_line_color=[100, 100, 100, 180],
                           line_width_min_pixels=1)
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer", data=mp, get_position='[lon, lat]',
                          get_fill_color='color', get_radius=30000,
                          pickable=True, auto_highlight=True),
                states_cmp],
        initial_view_state=pdk.ViewState(latitude=39.0, longitude=-98.0, zoom=3.4, pitch=0),
        map_style="light",
        tooltip={"html": "<b>A best:</b> {bestA} ¢/W<br><b>B best:</b> {bestB} ¢/W<br>"
                         "<b>A advantage:</b> {adv_disp} ¢/W",
                 "style": {"backgroundColor": "#1e293b", "color": "white",
                           "fontSize": "12px", "borderRadius": "8px"}}),
        use_container_width=True, height=480)
    st.markdown("🟢 **Green** = Portfolio A advantaged &nbsp;·&nbsp; "
                "🔴 **Red** = Portfolio B advantaged")

    # ── Demand by location ──
    st.markdown("#### 🌍 Demand by Location")
    st.caption("Where demand and building are happening. Dot size and color intensity = "
               "forecast DC capacity at that hail-grid location (no product implication).")
    years_all = sorted(market_sizes.keys())
    d_years = st.multiselect("Demand Years", years_all, default=years_all, key="cmp_dem_years")
    if d_years and shape_df is not None:
        d_sub = shape_df[shape_df['Year'].isin(d_years)]
        if len(d_sub) > 0:
            d_mw = d_sub.groupby(['hail_lat', 'hail_lon']).agg(
                mw=('DC Capacity (MW)', 'sum')).reset_index()
            d_total_mw = sum(market_sizes[yr] for yr in d_years) * 1000
            if d_mw['mw'].sum() > 0:
                d_mw['mw'] = d_mw['mw'] * (d_total_mw / d_mw['mw'].sum())
            d_mw = d_mw[d_mw['mw'] > 0].copy()
            d_scale = max(float(d_mw['mw'].quantile(0.95)), 1e-9)

            def _dem_color(v):
                g = min(v / d_scale, 1.0)
                return [int(219 - 190 * g), int(234 - 132 * g), int(254 - 79 * g)]  # light→dark blue

            d_mw['color'] = d_mw['mw'].apply(_dem_color)
            d_mw['radius'] = 8000 + 32000 * np.sqrt(d_mw['mw'] / d_scale).clip(0, 1.4)
            d_mw['mw_display'] = d_mw['mw'].round(0).astype(int).astype(str)
            states_dem = pdk.Layer("GeoJsonLayer", data=US_STATES_URL, stroked=True, filled=False,
                                   pickable=False, get_line_color=[100, 100, 100, 180],
                                   line_width_min_pixels=1)
            st.pydeck_chart(pdk.Deck(
                layers=[pdk.Layer("ScatterplotLayer", data=d_mw,
                                  get_position='[hail_lon, hail_lat]',
                                  get_fill_color='color', get_radius='radius',
                                  pickable=True, auto_highlight=True),
                        states_dem],
                initial_view_state=pdk.ViewState(latitude=39.0, longitude=-98.0, zoom=3.4, pitch=0),
                map_style="light",
                tooltip={"html": "<b>Demand:</b> {mw_display} MWdc",
                         "style": {"backgroundColor": "#1e293b", "color": "white",
                                   "fontSize": "12px", "borderRadius": "8px"}}),
                use_container_width=True, height=450)
        else:
            st.caption("No demand data for the selected years.")

    species = ([('A', a, compA, relA) for a in anglesA] +
               [('B', a, compB, relB) for a in anglesB])

    # ── Head-to-head averages ──
    st.markdown("#### 📊 Per-Angle Averages (rank-matched head-to-head)")
    rows = []
    for aA, aB in pairs:
        tA, tB = compA[f'total_{aA}'].mean(), compB[f'total_{aB}'].mean()
        rows.append({'Matchup': f'A {aA}° vs B {aB}°',
                     'A Avg Total (¢/W)': round(tA, 4),
                     'B Avg Total (¢/W)': round(tB, 4),
                     'A Advantage (¢/W)': round(tB - tA, 4),
                     'B Premium vs A': f"{(tB / tA - 1) * 100:+.1f}%" if tA > 0 else "—"})
    for aA in extraA:
        rows.append({'Matchup': f'A {aA}° (unmatched)',
                     'A Avg Total (¢/W)': round(compA[f'total_{aA}'].mean(), 4),
                     'B Avg Total (¢/W)': None, 'A Advantage (¢/W)': None, 'B Premium vs A': '—'})
    for aB in extraB:
        rows.append({'Matchup': f'B {aB}° (unmatched)',
                     'A Avg Total (¢/W)': None,
                     'B Avg Total (¢/W)': round(compB[f'total_{aB}'].mean(), 4),
                     'A Advantage (¢/W)': None, 'B Premium vs A': '—'})
    angle_rows = rows
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── Stacked cost composition bars (portfolio averages) ──
    STACK_COLS = ['CapEx', 'Insurance (base)', 'Insurance (unreliability)',
                  'Dev Risk (base)', 'Dev Risk (unreliability)']
    STACK_COLORS = ['#3b82f6', '#f97316', '#fdba74', '#7c3aed', '#c4b5fd']

    def _species_stack(comp, a, loc_idx=None):
        """Cost components for one species; averaged, or at a single location index."""
        def _v(frame, col):
            return float(frame[col].mean()) if loc_idx is None else float(frame[col].iloc[loc_idx])
        ins_p, risk_p = _v(comp100, f'ins_{a}'), _v(comp100, f'risk_{a}')
        ins_c, risk_c = _v(comp, f'ins_{a}'), _v(comp, f'risk_{a}')
        return {'CapEx': round(_v(comp, f'capex_{a}'), 4),
                'Insurance (base)': round(min(ins_p, ins_c), 4),
                'Insurance (unreliability)': round(max(ins_c - ins_p, 0.0), 4),
                'Dev Risk (base)': round(min(risk_p, risk_c), 4),
                'Dev Risk (unreliability)': round(max(risk_c - risk_p, 0.0), 4)}

    if comp100 is not None:
        st.markdown("##### Cost Composition by Species (avg ¢/W)")
        st.caption("Stack: CapEx (blue), insurance (orange) and developer risk (purple) at perfect "
                   "stow, plus the extra insurance (light orange) and developer risk (light purple) "
                   "taken on by stow un-reliability.")
        stack_rows = []
        for port, a, comp, r in species:
            stack_rows.append({'Species': f"{port} {a}°", **_species_stack(comp, a)})
        stk = pd.DataFrame(stack_rows).set_index('Species')
        st.bar_chart(stk, color=STACK_COLORS, height=340)
        st.dataframe(stk.reset_index(), hide_index=True, use_container_width=True)

        # ── Site-level stacked bars: Central Texas, Kern County, Chicago ──
        st.markdown("##### Cost Composition at Key Sites (¢/W)")
        SITES = [("Central Texas", 31.5, -99.2),
                 ("Kern County, CA", 35.34, -118.73),
                 ("Chicago, IL", 41.88, -87.63)]
        site_cols = st.columns(len(SITES))
        for sc, (site_name, s_lat, s_lon) in zip(site_cols, SITES):
            d2 = (compA['lat'] - s_lat) ** 2 + (compA['lon'] - s_lon) ** 2
            li = int(d2.idxmin())
            with sc:
                st.markdown(f"**{site_name}**")
                st.caption(f"Grid point ({compA['lat'].iloc[li]:.1f}, {compA['lon'].iloc[li]:.1f})")
                s_rows = [{'Species': f"{port} {a}°", **_species_stack(comp, a, loc_idx=li)}
                          for port, a, comp, r in species]
                s_stk = pd.DataFrame(s_rows).set_index('Species')
                st.bar_chart(s_stk, color=STACK_COLORS, height=300)

    # ── Vulnerability curves ──
    if base_df is not None:
        st.markdown("#### 🧬 Vulnerability Curves (total cost vs site hail severity)")
        st.caption("Each tracker species in the two portfolios plotted independently: "
                   "y = total cost (insurance + developer risk + CapEx, ¢/W) at that species' "
                   "portfolio reliability; x = locations ordered by site PML at 0° tilt "
                   "(flat-stow probable maximum loss, %). Averaged within each PML level."
                   + (" Severity index uses the 3.2 mm PML for blended glass." if "Blended" in glass_lbl else ""))
        vdf = pd.DataFrame({'sev': (base_df['pml_0'] * 100).round(0).values})
        for port, a, comp, r in species:
            vdf[f"{port} {a}° ({r:.1f}%)"] = comp[f'total_{a}'].values
        vcurve = vdf.groupby('sev').mean().sort_index()
        vcurve.index.name = "Site PML at 0° (%)"
        st.line_chart(vcurve, height=340)

    # ── Best-angle shift ──
    st.markdown("#### 🔀 Optimal-Angle Shifts")
    moved = int((compA['best_angle'] != compB['best_angle']).sum())
    st.markdown(f"The winning tracker differs between portfolios at **{moved}** of {len(compA)} "
                f"locations ({moved / len(compA) * 100:.1f}%). "
                f"(A picks from [{angA_str}], B from [{angB_str}].)")
    ct = pd.crosstab(compA['best_angle'].map(lambda x: f'A best: {x}°'),
                     compB['best_angle'].map(lambda x: f'B best: {x}°'))
    st.dataframe(ct, use_container_width=True)

    # ── Market share + dollar value (hidden by default for live demos) ──
    with st.expander("📈 Market Share & Dollar Value of Reliability (click to expand)",
                     expanded=False):
        years = sorted(market_sizes.keys())
        sel_years = st.multiselect("Years", years, default=years, key="cmp_years")
        if not sel_years:
            st.warning("Select at least one year.")
            return
        total_gw = sum(market_sizes[yr] for yr in sel_years)
        total_mw = total_gw * 1000

        dem = None
        if shape_df is not None:
            shape_sub = shape_df[shape_df['Year'].isin(sel_years)]
            if len(shape_sub) > 0:
                dbl = shape_sub.groupby(['hail_lat', 'hail_lon']).agg(
                    mw=('DC Capacity (MW)', 'sum')).reset_index()
                dem = dbl.merge(
                    compA[['lat', 'lon']].assign(_idx=np.arange(len(compA))),
                    left_on=['hail_lat', 'hail_lon'], right_on=['lat', 'lon'], how='inner')
                if len(dem) > 0 and dem['mw'].sum() > 0:
                    dem['mw'] = dem['mw'] * (total_mw / dem['mw'].sum())
                else:
                    dem = None
        if dem is None:
            dem = compA[['lat', 'lon']].copy()
            dem['_idx'] = np.arange(len(compA))
            dem['mw'] = total_mw / len(compA)
            st.caption("No demand-shape overlap for the selected years — using uniform demand.")

        idx = dem['_idx'].values
        bA = compA['best_angle'].values[idx]
        bB = compB['best_angle'].values[idx]
        union_angles = sorted(set(anglesA) | set(anglesB))
        share_rows = []
        for a in union_angles:
            gwA = dem['mw'].values[bA == a].sum() / 1000
            gwB = dem['mw'].values[bB == a].sum() / 1000
            share_rows.append({'Angle': f'{a}°',
                               'A GWdc': round(gwA, 2) if a in anglesA else None,
                               'B GWdc': round(gwB, 2) if a in anglesB else None,
                               'Δ GWdc (A−B)': round(gwA - gwB, 2)})
        sdf = pd.DataFrame(share_rows)
        mc1, mc2 = st.columns([1, 2])
        with mc1:
            st.dataframe(sdf, hide_index=True, use_container_width=True)
        with mc2:
            st.bar_chart(sdf.set_index('Angle')[['A GWdc', 'B GWdc']].fillna(0.0), height=300)

        adv_loc = adv.values[idx]
        dollars = float((dem['mw'].values * 1e4 * adv_loc).sum())      # ¢/W × MW → $
        wavg = float((dem['mw'].values * adv_loc).sum() / dem['mw'].values.sum())
        yr_lbl = (f"{min(sel_years)}–{max(sel_years)}" if len(sel_years) > 1 else str(sel_years[0]))
        st.markdown(f'<div class="winner-banner">💵 Demand-weighted portfolio advantage of A: '
                    f'{wavg:+.4f} ¢/W  ≈  ${dollars / 1e6:+,.1f}M across {total_gw:.1f} GWdc '
                    f'({yr_lbl})</div>', unsafe_allow_html=True)
        st.caption("Dollar value = Σ location demand × (B portfolio best cost − A portfolio best "
                   "cost): the total-cost gap across the selected market if every MW buys the "
                   "cheapest tracker offered by each portfolio.")

        # ── Export ──
        exp = [{'Section': 'Inputs', 'Key': 'Portfolio A Angles', 'Value': angA_str},
               {'Section': 'Inputs', 'Key': 'Portfolio A Reliability (%)', 'Value': relA},
               {'Section': 'Inputs', 'Key': 'Portfolio A Failed At (% of design)', 'Value': stuckA},
               {'Section': 'Inputs', 'Key': 'Portfolio B Angles', 'Value': angB_str},
               {'Section': 'Inputs', 'Key': 'Portfolio B Reliability (%)', 'Value': relB},
               {'Section': 'Inputs', 'Key': 'Portfolio B Failed At (% of design)', 'Value': stuckB},
               {'Section': 'Inputs', 'Key': 'Glass', 'Value': glass_lbl},
               {'Section': 'Inputs', 'Key': 'Damage Decay Sharpness (power)', 'Value': damage_shape_pow},
               {'Section': 'Inputs', 'Key': 'Damage Floor (% of flat)', 'Value': damage_floor_pct},
               {'Section': 'Inputs', 'Key': 'Years', 'Value': ', '.join(map(str, sel_years))},
               {'Section': 'Inputs', 'Key': 'Total Market (GWdc)', 'Value': round(total_gw, 2)},
               {'Section': 'Summary', 'Key': 'Avg Portfolio A Advantage (¢/W)', 'Value': round(float(adv.mean()), 5)},
               {'Section': 'Summary', 'Key': 'Median Portfolio A Advantage (¢/W)', 'Value': round(float(adv.median()), 5)},
               {'Section': 'Summary', 'Key': 'Demand-weighted A Advantage (¢/W)', 'Value': round(wavg, 5)},
               {'Section': 'Summary', 'Key': 'Dollar Value of A Advantage ($M)', 'Value': round(dollars / 1e6, 2)},
               {'Section': 'Summary', 'Key': 'Locations Where Winning Tracker Differs', 'Value': moved}]
        for r in angle_rows:
            exp.append({'Section': 'Head-to-Head', 'Key': f"{r['Matchup']} A Avg (¢/W)", 'Value': r['A Avg Total (¢/W)']})
            exp.append({'Section': 'Head-to-Head', 'Key': f"{r['Matchup']} B Avg (¢/W)", 'Value': r['B Avg Total (¢/W)']})
            exp.append({'Section': 'Head-to-Head', 'Key': f"{r['Matchup']} A Advantage (¢/W)", 'Value': r['A Advantage (¢/W)']})
        for _, r in sdf.iterrows():
            exp.append({'Section': 'Market Share', 'Key': f"{r['Angle']} A GWdc", 'Value': r['A GWdc']})
            exp.append({'Section': 'Market Share', 'Key': f"{r['Angle']} B GWdc", 'Value': r['B GWdc']})
        st.download_button("📥 Export Comparison CSV",
                           pd.DataFrame(exp).to_csv(index=False).encode(),
                           file_name="reliability_comparison.csv", mime="text/csv", key="cmp_export")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
st.markdown('<p class="main-title">🌨️ Hail Risk TCO — Stow Reliability Edition</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Interactive sensitivity analysis across stow angles (52°, 60°, 70°, 77°) with partial-stow reliability modeling</p>',
            unsafe_allow_html=True)

new_annuity = annuity_factor(interest_rate / 100.0)
st.caption(f"Discount rate: {interest_rate:.2f}% → {new_annuity:.2f}× annuity "
           f"(base: 6.00% → {BASE_ANNUITY:.2f}×)  |  "
           f"Premium: {annual_premium:.2f}%  |  Coverage: {coverage_ratio}%  |  "
           + _rel_caption
           + (f"  |  ⚠️ Curve correction: r→max(r^{damage_shape_pow:.2f}, {damage_floor_pct:.1f}%)"
              if (damage_floor_pct > 0 or abs(damage_shape_pow - 1.0) > 1e-9) else "") + "  |  "
           + f"Demand shape: **{demand_shape}**  |  Magnitude: **{magnitude_source}**  |  "
           f"σ = **{sigma:.2f}**")

with st.expander("📐 Damage Curves (P50, into-wind — ATI/VDE site studies, absolute magnitudes)"):
    st.markdown(
        "Site-average P50 into-wind PML/AAL vs tilt angle (**Ft. Stockton TX**, **Snyder TX**, "
        "**Stuttgart AR**), in **absolute magnitudes** — 2.0 mm glass takes more damage than "
        "3.2 mm at every angle. These curves shape the interpolation **between** each location's "
        "anchor values at 0°/45°/50°/52°/60°/70°/75°/77° (anchors honored exactly at baseline). "
        "P50 is used because P90/MC curves saturate at the 50% loss cap and would falsely flatten "
        "the shallow-angle region. Peak damage sits ~10–20° into the wind. "
        "When the **Damage Curve Correction** is active, the corrected curves are overlaid.")
    _, _shape_table = load_shape_functions()
    _sc = _shape_table.set_index('theta')
    _corr_on = (damage_floor_pct > 0) or (abs(damage_shape_pow - 1.0) > 1e-9)
    _pml_plot = _sc[['pml_20', 'pml_32']].rename(columns={'pml_20': '2.0 mm', 'pml_32': '3.2 mm'})
    _aal_plot = _sc[['aal_20', 'aal_32']].rename(columns={'aal_20': '2.0 mm', 'aal_32': '3.2 mm'})
    if _corr_on:
        for _plot, _c20, _c32 in [(_pml_plot, 'pml_20', 'pml_32'), (_aal_plot, 'aal_20', 'aal_32')]:
            for col, srccol in [('2.0 mm corrected', _c20), ('3.2 mm corrected', _c32)]:
                _plot[col] = apply_curve_correction(
                    _sc[srccol].values, np.full(len(_sc), _sc[srccol].iloc[0]),
                    damage_floor_pct, damage_shape_pow)
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**PML vs tilt angle (%)**")
        st.line_chart(_pml_plot, height=280)
    with sc2:
        st.markdown("**AAL vs tilt angle (%)**")
        st.line_chart(_aal_plot, height=280)

shape_df = _shape_df  # demand shape data (Orennia or WoodMac)

if rel_mode == "Compare Two Portfolios":
    if glass_choice == "Compare Both":
        st.info("Portfolio comparison runs one glass configuration at a time — using **3.2 mm**. "
                "Pick 2.0 mm, 3.2 mm, or Blended in the sidebar to change it.")
        eff_glass = "3.2 mm"
    else:
        eff_glass = glass_choice

    if eff_glass == "Blended":
        df20, df32 = load_data('20'), load_data('32')
        base_df, base_suffix = df32, '32'
        compA = compute_blended(df20, df32, glass_pct_20, replacement_cost, coverage_ratio,
                                annual_premium, interest_rate, risk_pct, capex_dict,
                                ins_on, risk_on, capex_on, active_angles=angles_a,
                                reliability=rel_a, stuck_pct=stuck_a)
        compB = compute_blended(df20, df32, glass_pct_20, replacement_cost, coverage_ratio,
                                annual_premium, interest_rate, risk_pct, capex_dict,
                                ins_on, risk_on, capex_on, active_angles=angles_b,
                                reliability=rel_b, stuck_pct=stuck_b)
        comp100 = compute_blended(df20, df32, glass_pct_20, replacement_cost, coverage_ratio,
                                  annual_premium, interest_rate, risk_pct, capex_dict,
                                  ins_on, risk_on, capex_on, active_angles=ANGLES,
                                  reliability=100.0, stuck_pct=0.0)
        glass_lbl = f"Blended ({glass_pct_20}% 2.0mm / {100 - glass_pct_20}% 3.2mm)"
    else:
        base_suffix = '32' if eff_glass == "3.2 mm" else '20'
        base_df = load_data(base_suffix)
        compA = compute_costs(base_df, base_suffix, replacement_cost, coverage_ratio,
                              annual_premium, interest_rate, risk_pct, capex_dict,
                              ins_on, risk_on, capex_on, active_angles=angles_a,
                              reliability=rel_a, stuck_pct=stuck_a)
        compB = compute_costs(base_df, base_suffix, replacement_cost, coverage_ratio,
                              annual_premium, interest_rate, risk_pct, capex_dict,
                              ins_on, risk_on, capex_on, active_angles=angles_b,
                              reliability=rel_b, stuck_pct=stuck_b)
        comp100 = compute_costs(base_df, base_suffix, replacement_cost, coverage_ratio,
                                annual_premium, interest_rate, risk_pct, capex_dict,
                                ins_on, risk_on, capex_on, active_angles=ANGLES,
                                reliability=100.0, stuck_pct=0.0)
        glass_lbl = eff_glass

    tabA, tabB, tabC = st.tabs([f"🅰 Portfolio A — {_ang_a_str} @ {rel_a:.1f}%",
                                f"🅱 Portfolio B — {_ang_b_str} @ {rel_b:.1f}%",
                                "⚖️ A vs B Comparison"])
    with tabA:
        render_single(base_df, f"Portfolio A — {glass_lbl}, [{_ang_a_str}] @ {rel_a:.1f}% stow reliability",
                      precomputed=compA, suffix=base_suffix, angles=angles_a)
        render_lookup(base_df, compA, "PortA", suffix_lk=base_suffix,
                      rel_lk=rel_a, stuck_lk=stuck_a, angles_lk=angles_a)
    with tabB:
        render_single(base_df, f"Portfolio B — {glass_lbl}, [{_ang_b_str}] @ {rel_b:.1f}% stow reliability",
                      precomputed=compB, suffix=base_suffix, angles=angles_b)
        render_lookup(base_df, compB, "PortB", suffix_lk=base_suffix,
                      rel_lk=rel_b, stuck_lk=stuck_b, angles_lk=angles_b)
    with tabC:
        render_portfolio_comparison(compA, compB, rel_a, rel_b, stuck_a, stuck_b,
                                    shape_df, glass_lbl, anglesA=angles_a, anglesB=angles_b,
                                    comp100=comp100, base_df=base_df)

elif glass_choice == "Compare Both":
    df20, df32 = load_data('20'), load_data('32')
    tab1, tab2 = st.tabs(["2.0 mm Glass", "3.2 mm Glass"])
    with tab1:
        comp20 = render_single(df20, "2.0 mm Glass", suffix='20')
        render_lookup(df20, comp20, "2.0mm", suffix_lk='20')
        render_market_share(comp20, shape_df, "2.0mm")
        render_value_gap(comp20, "2.0mm")
        luce20 = render_luce_market_share(comp20, shape_df, "2.0mm")
        render_luce_demand_map(luce20, "2.0mm")
    with tab2:
        comp32 = render_single(df32, "3.2 mm Glass", suffix='32')
        render_lookup(df32, comp32, "3.2mm", suffix_lk='32')
        render_market_share(comp32, shape_df, "3.2mm")
        render_value_gap(comp32, "3.2mm")
        luce32 = render_luce_market_share(comp32, shape_df, "3.2mm")
        render_luce_demand_map(luce32, "3.2mm")

    # Glass comparison
    st.markdown("---")
    st.markdown("### 🔀 Where Does Glass Thickness Change the Optimal Angle?")
    merged = comp20[['lat', 'lon', 'best_angle', 'best_cost']].merge(
        comp32[['lat', 'lon', 'best_angle', 'best_cost']], on=['lat', 'lon'], suffixes=('_20', '_32'))
    changed = merged[merged['best_angle_20'] != merged['best_angle_32']]
    st.markdown(f"At **{len(changed)}** of {len(merged)} locations ({len(changed)/len(merged)*100:.1f}%), "
                f"the optimal angle differs between glass types.")
    if len(changed) > 0:
        cdf = changed[['lat', 'lon', 'best_angle_20', 'best_cost_20', 'best_angle_32', 'best_cost_32']].copy()
        cdf.columns = ['Lat', 'Lon', '2.0mm Best', '2.0mm Cost', '3.2mm Best', '3.2mm Cost']
        cdf['2.0mm Cost'] = cdf['2.0mm Cost'].round(4)
        cdf['3.2mm Cost'] = cdf['3.2mm Cost'].round(4)
        st.dataframe(cdf.sort_values('Lat'), hide_index=True, use_container_width=True, height=300)

elif glass_choice == "Blended":
    df20, df32 = load_data('20'), load_data('32')
    blended = compute_blended(df20, df32, glass_pct_20, replacement_cost, coverage_ratio,
                              annual_premium, interest_rate, risk_pct, capex_dict,
                              ins_on, risk_on, capex_on, active_angles=ACTIVE_ANGLES,
                              reliability=stow_reliability, stuck_pct=stuck_pct)
    label = f"Blended ({glass_pct_20}% 2.0mm / {100-glass_pct_20}% 3.2mm)"
    render_single(df32, label, precomputed=blended, suffix='32')
    render_lookup(df32, blended, "Blended", suffix_lk='32')
    render_market_share(blended, shape_df, "Blended")
    render_value_gap(blended, "Blended")
    luce_b = render_luce_market_share(blended, shape_df, "Blended")
    render_luce_demand_map(luce_b, "Blended")

else:
    suffix = '32' if glass_choice == "3.2 mm" else '20'
    df = load_data(suffix)
    computed = render_single(df, glass_choice, suffix=suffix)
    render_lookup(df, computed, glass_choice, suffix_lk=suffix)
    render_market_share(computed, shape_df, glass_choice)
    render_value_gap(computed, glass_choice)
    luce_r = render_luce_market_share(computed, shape_df, glass_choice)
    render_luce_demand_map(luce_r, glass_choice)

# ─── Parameter Summary ───
st.markdown("---")
st.markdown("### 📊 Current Parameter Summary")
p1, p2, p3 = st.columns(3)
with p1:
    glass_label = glass_choice
    if glass_choice == "Blended":
        glass_label = f"Blended ({glass_pct_20}/{100-glass_pct_20})"
    st.markdown(f"| Parameter | Value |\n|---|---|\n| Glass Type | **{glass_label}** |"
                f"\n| Replacement Cost | **${replacement_cost:.2f}/W** |"
                f"\n| Discount Rate | **{interest_rate:.2f}%** |\n| Annuity Factor | **{new_annuity:.2f}×** |"
                f"\n| Active Angles | **{', '.join(str(a)+'°' for a in ACTIVE_ANGLES)}** |")
with p2:
    st.markdown(f"| Parameter | Value |\n|---|---|\n| Annual Premium | **{annual_premium:.2f}%** |"
                f"\n| Coverage Ratio | **{coverage_ratio}%** |\n| Dev Risk Factor | **{risk_pct}%** |"
                + (f"\n| Stow Reliability | **{stow_reliability:.1f}%** |"
                   f"\n| Failed Trackers At | **{stuck_pct}% of design** |"
                   if rel_mode == "Single Portfolio" else
                   f"\n| Portfolio A | **[{_ang_a_str}] {rel_a:.1f}% rel, failed at {stuck_a}%** |"
                   f"\n| Portfolio B | **[{_ang_b_str}] {rel_b:.1f}% rel, failed at {stuck_b}%** |")
                + 
                f"\n| Curve Correction | **power {damage_shape_pow:.2f}, floor {damage_floor_pct:.1f}%** |"
                f"\n| Active Layers | **{layers_str}** |"
                f"\n| Demand Shape | **{demand_shape}** |"
                f"\n| Magnitude Source | **{magnitude_source}** |"
                f"\n| Sigma (Luce) | **{sigma:.2f}** |")
with p3:
    mkt_str = "| Year | GWdc |\n|---|---|"
    for yr in sorted(market_sizes.keys()):
        mkt_str += f"\n| {yr} | **{market_sizes[yr]:.1f}** |"
    st.markdown(mkt_str)
