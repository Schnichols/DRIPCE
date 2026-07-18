# Hail Reliability Tool — Stow Reliability Edition (v5)

Fork of the Hail Risk TCO tool ([Hailmappingv4](https://github.com/Schnichols/Hailmappingv4)) adding
**stow-reliability modeling**: what does hail TCO look like when only X% of trackers reach full stow?

## New in this edition

**Portfolio comparison mode (v2):** a Mode toggle in the Stow Reliability section switches
between *Single Portfolio* and *Compare Two Portfolios*. Compare mode renders three tabs:

- **🅰 Portfolio A** / **🅱 Portfolio B** — the full standard analysis (metric cards, 3D cost map,
  optimal-angle map, win counts, site lookup) at each portfolio's own reliability & stuck angle.
  Each portfolio has its **own tracker-angle picker** (v3): defaults A = [52°, 77°] @ 99.7%,
  B = [60°, 77°] @ 90.0%; more angles can be added to either side.
- **⚖️ A vs B Comparison** — **rank-matched head-to-head** cards and table (highest angles
  compete against each other, lowest against lowest), a diverging **portfolio advantage** map
  (each location buys its cheapest tracker from each portfolio's own set; green = A cheaper,
  red = B cheaper), **vulnerability curves** (total cost of every tracker species vs site PML
  at 0° tilt), **stacked cost-composition bars** per species — CapEx (blue), insurance (orange),
  developer risk (purple), with lighter hues for the extra insurance/risk taken on by stow
  un-reliability — optimal-angle shift crosstab, market-share comparison by year, the
  demand-weighted dollar value of the portfolio gap, and a comparison CSV export.

**Angle Visibility Filter (v3):** the last section of the sidebar controls which tilt angles are
offered at the top of the panel and in the portfolio pickers. **70° is hidden by default** — it
stays fully functional in the code and can be re-enabled with one checkbox, no reprogramming.

Reliability sliders use 0.1% steps in both modes (so 99.7% is settable exactly).

**Two sidebar sliders (single-portfolio mode):**
- **Trackers Reaching Full Stow (%)** — 0–100. At 100% the tool reproduces the original
  Hailmappingv4 results *exactly* (verified to 1e-14).
- **Failed Trackers At (% of design angle)** — where the failed fraction ends up.
  0% = stuck flat. 15% puts a 52° tracker at 7.8°, a 60° tracker at 9.0°, etc.

Effective cost per design angle `a`:

```
cost(a) = R · [ins(a) + risk(a)]  +  (1−R) · [ins(θs) + risk(θs)]  +  capex(a)
θs = stuck% × a
ins(θ)  = PML(θ) · RC · coverage · premium · annuity      (insurance premium)
risk(θ) = AAL(θ) · RC · annuity · risk%                   (below-deductible developer risk)
```

## Continuous PML/AAL curves

Per-location anchors exist at **0°, 45°, 50°, 52°, 60°, 70°, 77°** (77° carries the 75° VDE data,
matching the original tool's convention). Between anchors, values follow a **canonical P50
into-wind damage shape** derived from the ATI/VDE site studies (Ft. Stockton TX, Snyder TX,
Stuttgart AR — `250804_ATI_77deg_ResultsSummary.xlsx`), normalized per glass type:

```
V(θ) = v_i + (v_j − v_i) · (f(θ) − f(θ_i)) / (f(θ_j) − f(θ_i))
```

- Anchors are always honored **exactly**; the shape only bridges the gaps.
- P50 curves were used deliberately — P90/MC saturate at the 50% loss cap and would
  falsely flatten the shallow-angle region.
- The into-wind damage **bump (peak ~10–20°)** is reproduced inside the 0–45° gap,
  scaled to each location's own 0°→45° drop.
- Design-angle anchors (52/60/70/77) are the *true unrounded* PML/AAL values recovered from
  the cost columns (`PML = ins / (RC·1.25·1.25%·annuity)`, `AAL = risk / (RC·annuity)`),
  which is what guarantees exact parity at 100% reliability.

## Data files

| File | Contents |
|---|---|
| `hail_data_20_ext.csv` / `hail_data_32_ext.csv` | 1,112 locations × PML/AAL anchors at 7 angles + original ins/risk columns |
| `shape_curves.csv` | Canonical P50 into-wind PML/AAL shapes (per glass type) |
| `orennia_market_demand_05.18.26_v2.csv`, `woodmac_demand_05.18.26.csv` | Demand data (unchanged) |

## Known data-resolution limits

- Shallow anchors (0/45/50) come from the source CSV's **integer-percent** columns. For PML
  (10–40% at risky sites) this is fine. For **AAL**, integer rounding is coarse — sites where
  AAL at 0/45/50 all round to the same value get a flat AAL curve across 0–50° instead of the
  bump. Fix: re-export AAL 0/45/50 from the JMP table with decimals and rebuild the ext files.
- An isotonic cleanup lifts rounded shallow anchors that undercut the true 52° value
  (rounding artifacts only; never lowers a given value).

## Deploy

Streamlit Cloud → New app → this repo → Main file path: `hail_reliability_v5.py`

## v4 changes

- **75° product added** — carries the same VDE 75° damage data as the 77° label (the honest
  treatment: "77" was always priced with 75° data), so an A 77° vs B 75° matchup isolates
  reliability + CapEx. Default portfolios: **A = [52°, 77°] @ 99.7%**, **B = [60°, 75°] @ 90%**,
  both with failed trackers defaulting to **20% of design angle** (~12–15°, near the damage peak).
- **New defaults**: glass = Blended 50/50; CapEx 52°/60° = 0, 70° = 1.7, 75° = 2.5, 77° = 2.5 ¢/W;
  developer risk on at **50%** weight.
- **Comparison sheet**: demand-by-location map (intensity only — no product implication);
  site-level stacked cost bars for Central Texas, Kern County CA, and Chicago; vulnerability
  curves moved below the stacked bars; **market share & dollar value collapsed into an expander**
  (hidden by default for live demos).
- **Tilt-angle color legends** on every angle-colored map.
- Normalized damage-shape-curve expander removed (curves still power the interpolation).

## v5 changes

- **Damage curves shown in absolute magnitudes**: `shape_curves.csv` now stores the site-average
  P50 into-wind PML/AAL in % (normalized shape × average flat magnitude), so 2.0 mm sits above
  3.2 mm everywhere and the curves are directly comparable. Interpolation math is scale-invariant,
  so model behavior is bit-identical. The curve expander is restored with these absolute curves
  (and overlays the corrected curves when the correction is active).
- **Damage Curve Correction** (new sidebar section): commercial adjustment of the
  damage-vs-tilt discount, applied to the damage ratio r = V(θ)/V(flat) as
  **r′ = max(r^power, floor)**:
  - *Decay Sharpness (power)*: < 1 → damage declines less sharply with tilt (smaller risk
    discount for steeper stow — the commercial-feedback direction); 1.0 = VDE baseline.
  - *Minimum Damage Floor (% of flat)*: damage at any tilt retains at least this share of the
    site's flat (0°) damage; zero-risk sites always stay zero.
  - At defaults (power 1.0, floor 0) the tool reproduces the given VDE data exactly. When
    active, the correction applies consistently to design angles, stuck angles, insurance,
    developer risk, lookup curves, and vulnerability curves, and is recorded in captions,
    the parameter summary, and every CSV export.

## v6 changes (`hail_reliability_v6.py`)

- **Texas reference site relocated**: "Central Texas" (31.5, −99.2) → **"Texas Extreme"
  (32.6, −100.3)** — the TX grid point where A [60°/77°] has the largest advantage over
  B [60°/75°] (~1.99 ¢/W vs 1.45 at the old site; near Snyder, one of the VDE study sites).
- **Value-of-unreliability table** below the key-site stacked bars (added insurance + added
  owner risk per site × species).
- **Common Y-axis toggle** on the key-site cost-composition charts (Altair-rendered).
- **Darker/thicker state lines** on the A-vs-B comparison maps.
- **Portfolio A default** → [52°, 60°, 77°].

## v7 changes (`hail_reliability_v7.py`)

- **Insurer Tail Reliability (P90)** — new sidebar section, off by default. Event-level stow
  fraction is modeled as a Gaussian around each portfolio's mean reliability; when enabled the
  **insurance layer only** is priced at a lower percentile (P75/P90/P95/P99 selectable,
  default P90 = mean − 1.2816σ), while **developer/owner risk stays at mean reliability**
  (developers hold the average risk; insurers underwrite the tail).
  - Per-portfolio σ sliders. Defaults: **A σ = 2.0** (tail-fit to Array field data: 230 stow
    events, 2 below 95% → P(X<95) = 0.87% → σ ≈ 1.98; P90 ≈ 97.2% at a 99.7% mean) and
    **B σ = 4.0** (no public NXT distribution data — claims censored below the ~5% deductible;
    wider uncertainty; P90 ≈ 84.9% at a 90% mean).
  - Toggle-off reproduces v6 behavior exactly (verified bit-identical).
  - Recorded in the comparison caption, parameter summary, and comparison CSV export.
