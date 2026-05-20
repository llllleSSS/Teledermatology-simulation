"""
Teledermatology Simulation App

Configurable simulation for capacity planning in teledermatology clinics.
Layout: horizontal flow-chart input, minimal sidebar for simulation controls.

Author: Lexie Sun
Date: April 2025
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation import run_simulation, run_sensitivity, detect_steady_state

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Teledermatology Capacity Planning",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: keep sidebar narrow, style flow-chart boxes, minimum width
st.markdown("""
<style>
    section[data-testid="stSidebar"] { width: 300px !important; min-width: 300px !important; }
    .main .block-container { max-width: 1600px; padding-top: 1.5rem; }
    
    /* Flow chart header bars */
    .flow-header-green {
        background-color: #2E8B57;
        color: white;
        padding: 8px 10px;
        text-align: center;
        font-weight: 600;
        font-size: 14px;
        border-radius: 6px 6px 0 0;
        margin: -16px -16px 10px -16px;
    }
    .flow-header-blue {
        background-color: #1E90FF;
        color: white;
        padding: 8px 10px;
        text-align: center;
        font-weight: 600;
        font-size: 14px;
        border-radius: 6px 6px 0 0;
        margin: -16px -16px 10px -16px;
    }
    .flow-header-amber {
        background-color: #D4A017;
        color: white;
        padding: 8px 10px;
        text-align: center;
        font-weight: 600;
        font-size: 14px;
        border-radius: 6px 6px 0 0;
        margin: -16px -16px 10px -16px;
    }
    .icon-row {
        color: #5F5E5A;
        font-size: 13px;
        font-style: italic;
        margin-bottom: 4px;
    }
    .icon-symbol {
        font-size: 18px;
        margin-right: 6px;
        color: #2E8B57;
    }
    .icon-symbol-blue { color: #1E90FF; }
    .flow-arrow {
        text-align: center;
        font-size: 28px;
        color: #5F5E5A;
        padding-top: 70px;
        font-weight: bold;
    }
    .discharge-label {
        padding-top: 70px;
        font-weight: 600;
        color: #0F6E56;
        font-size: 14px;
    }
    .discharge-caption {
        font-size: 11px;
        font-style: italic;
        color: #0F6E56;
        margin-top: -5px;
    }
    /* Tighten number_input vertical spacing in flow chart */
    div[data-testid="stNumberInput"] { margin-bottom: 4px; }
    
    /* Section label inside a flow-chart box (e.g. 'Buffer', 'Waiting line') */
    .section-label {
        font-weight: 600;
        color: #1E293B;
        font-size: 13px;
        margin-top: 4px;
        margin-bottom: 2px;
    }
    .section-divider {
        border: 0;
        border-top: 1px solid #E2E8F0;
        margin: 12px 0 8px 0;
    }
    .max-queue-display {
        font-size: 12px;
        color: #5F5E5A;
        font-style: italic;
        margin: -2px 0 6px 2px;
    }
    
    /* Inline warning strips under flow-chart boxes */
    .inline-warning-error {
        background-color: #FEE2E2;
        border-left: 4px solid #DC2626;
        color: #991B1B;
        padding: 6px 10px;
        margin: 4px 0 0 0;
        border-radius: 4px;
        font-size: 12px;
        line-height: 1.3;
    }
    .inline-warning-warn {
        background-color: #FEF3C7;
        border-left: 4px solid #D97706;
        color: #92400E;
        padding: 6px 10px;
        margin: 4px 0 0 0;
        border-radius: 4px;
        font-size: 12px;
        line-height: 1.3;
    }
    
    /* Min-required-hours indicators inside Service boxes */
    .min-hrs-ok {
        color: #065F46;
        background-color: #D1FAE5;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        margin-top: 4px;
    }
    .min-hrs-bad {
        color: #991B1B;
        background-color: #FEE2E2;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        margin-top: 4px;
    }
    
    /* Recommendation cards */
    .rec-card {
        border-left: 5px solid #6B7280;
        background-color: #F9FAFB;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 6px;
    }
    .rec-card-critical {
        border-left-color: #DC2626;
        background-color: #FEF2F2;
    }
    .rec-card-warn {
        border-left-color: #D97706;
        background-color: #FFFBEB;
    }
    .rec-card-info {
        border-left-color: #2563EB;
        background-color: #EFF6FF;
    }
    .rec-card-good {
        border-left-color: #059669;
        background-color: #ECFDF5;
    }
    .rec-card-title {
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 4px;
        color: #1F2937;
    }
    .rec-card-body {
        font-size: 13px;
        color: #374151;
        line-height: 1.45;
    }
    .rec-card-body ul {
        margin: 6px 0 0 16px;
        padding: 0;
    }
    .rec-card-body li {
        margin: 2px 0;
    }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({'font.size': 9})

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_INITIAL_Q_E = 50
DEFAULT_INITIAL_Q_F = 100
DEFAULT_TARGET_Q_E = 10
DEFAULT_TARGET_Q_F = 20

DEFAULT_LAMBDA_E = 28
DEFAULT_LAMBDA_D = 34
DEFAULT_GAMMA = 0.4

# Service times in MINUTES per case (UI-facing; rate is derived as 60/min)
DEFAULT_ECONSULT_MIN_PER_CASE = 5.0   # 5 min/case → 12 cases/hr
DEFAULT_FTF_MIN_PER_CASE = 15.0       # 15 min/case → 4 cases/hr

DEFAULT_HRS_ECONSULT = 2.0
DEFAULT_HRS_FTF = 6.0

DEFAULT_HRS_E_SD = 0.0          # eConsult: Gaussian SD on hours
DEFAULT_P_ABSENCE_F = 0.0       # FTF: Bernoulli absence probability (0 = no absences)

DEFAULT_BUFFER_DAYS_E = 10
DEFAULT_BUFFER_DAYS_F = 10

# Plot colors
COLOR_ECONSULT = '#2E8B57'
COLOR_FTF = '#1E90FF'
COLOR_TOTAL = '#8B008B'

# =============================================================================
# SESSION STATE
# =============================================================================

if 'sim_results' not in st.session_state:
    st.session_state.sim_results = None
if 'sim_params' not in st.session_state:
    st.session_state.sim_params = None
if 'cancel_simulation' not in st.session_state:
    st.session_state.cancel_simulation = False
if 'cap_sens_results' not in st.session_state:
    st.session_state.cap_sens_results = None
if 'ftf_clusters' not in st.session_state:
    # Default: 1 cluster representing the full practice
    st.session_state.ftf_clusters = [
        {'hours': 6.0, 'sd': 0.0, 'p_absent': 0.0}
    ]
if 'sec_per_rep_day' not in st.session_state:
    # Calibration constant for time estimates: seconds per (replication × day).
    # Initial guess based on benchmarking; will be replaced by measured value after first run.
    st.session_state.sec_per_rep_day = 0.00003

# =============================================================================
# ABOUT DIALOG
# =============================================================================

@st.dialog("About the App")
def show_about():
    st.markdown("""
    ### Teledermatology Capacity Planning Tool
    
    This tool simulates a teledermatology clinic with two service pathways:
    - **eConsult**: Faster asynchronous consultations
    - **Face-to-Face (FTF)**: Traditional in-person appointments
    
    ---
    
    **How to Use:**
    
    Fill in the parameters along the patient-flow diagram:
    1. **Referrals** - how many new patients arrive per day (on average)
    2. **Queue** - your current backlog and target queue level
    3. **Service** - how fast each case is handled and how many hours per day are allocated
    4. **Conversion rate** - fraction of eConsult patients needing FTF follow-up
    5. **Buffer** - max wait time before new arrivals are blocked
    
    ---
    
    **Backlog vs Buffer:**
    - **Backlog**: Current queue size (can exceed buffer - this is your real starting point)
    - **Buffer**: Max queue size for NEW arrivals (= buffer_days × capacity)
    - When queue exceeds buffer, new arrivals are blocked until queue drops back down
    
    ---
    
    **Fixed Assumptions:**
    - FIFO queue discipline
    - No priority between patient types
    - Poisson arrivals (daily counts vary randomly around the mean)
    
    ---
    
    *Developed for URMC Dermatology, University of Rochester*
    """)


@st.dialog("Glossary of Terms")
def show_glossary():
    st.markdown("""
    Quick reference for technical terms used throughout the app.
    
    ---
    
    **Standard deviation (SD)** — A measure of variability around an average.
    Larger SD = more day-to-day variation. SD = 0 means the value is the same every day (deterministic).
    Example: if eConsult hours have mean = 2 and SD = 0.5, most days will fall between 1 and 3 hours.
    
    ---
    
    **P(absent) — absence probability** — The chance that a cluster of providers
    is absent on a given day (sick, on vacation, at a conference, etc.). On absent days,
    that cluster contributes 0 hours to FTF capacity. P = 0 means no absences;
    P = 0.05 means roughly 5% of days have that cluster absent.
    
    ---
    
    **Efficiency ratio** — How many eConsult cases can be completed per hour, compared
    to FTF cases. Calculated as: (eConsult cases per hour) / (FTF cases per hour).
    A ratio of 3.0 means eConsults are completed 3× faster than FTF visits per hour of work.
    
    ---
    
    **Max queue size (buffer)** — The maximum number of patients allowed in a queue
    before new arrivals are blocked. Calculated as: buffer days × mean daily capacity.
    Once the queue reaches this size, new patients cannot join until the queue shrinks.
    Existing initial backlog can exceed the buffer — they are served eventually,
    but no new patients are admitted while the queue is over the limit.
    
    ---
    
    **Utilization (ρ, rho)** — Fraction of capacity that demand consumes.
    ρ = arrival rate / service capacity.
    - ρ < 1 → queue stable (does not grow indefinitely)
    - ρ ≈ 1 → queue near capacity limit (small fluctuations cause large spikes)
    - ρ ≥ 1 → queue grows without bound (capacity is insufficient)
    
    ---
    
    **Conversion rate (γ, gamma)** — The fraction of eConsults that subsequently
    require an FTF follow-up visit. A conversion rate of 0.40 means 40% of completed
    eConsults end up scheduling an in-person visit.
    
    ---
    
    **Effective FTF demand** — The total daily demand for FTF visits, including
    both direct FTF referrals AND patients converted from eConsults.
    Calculated as: λ_d + γ × λ_e (direct arrivals + conversion fraction × eConsult arrivals).
    
    ---
    
    **Replication** — A single complete simulation run with its own random seed.
    Multiple replications are run and averaged to estimate the typical behavior of the system,
    since simulations include random elements (arrivals, conversions, absences).
    
    ---
    
    **Long-term analysis (post-warmup period)** — The portion of the simulation
    used for computing average performance. By default, the first half of the simulation
    is discarded because the queues are still settling from their initial backlog.
    The second half gives a more representative view of how the system behaves
    once it has stabilized.
    
    ---
    
    **Cluster (FTF capacity)** — A group of providers who share the same working pattern
    (same mean hours per day, same variability, same absence rate).
    A practice with several types of staff (e.g., attendings vs. fellows)
    can be modeled with multiple clusters.
    """)

# =============================================================================
# SIDEBAR - Minimal: About + Simulation controls
# =============================================================================

st.sidebar.title("🏥 Teledermatology")
st.sidebar.markdown("---")

if st.sidebar.button("ℹ️ About the App", use_container_width=True):
    show_about()

if st.sidebar.button("📖 Glossary of Terms", use_container_width=True):
    show_glossary()

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Simulation Settings")

sim_horizon = st.sidebar.number_input(
    "Simulation horizon (days)",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    help="Total number of days to simulate. The first half is used to let the system stabilize after the initial backlog; long-term performance metrics are computed from the second half."
)

num_replications = st.sidebar.number_input(
    "Number of replications",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
    help="Number of simulation runs to average. More replications give smoother estimates but take longer."
)

# Rough time estimate, using calibration from previous runs (if any)
_sec_per_rep_day = st.session_state.get('sec_per_rep_day', 0.00003)
_est_seconds = max(1, int(sim_horizon * num_replications * _sec_per_rep_day * 1.2))
if _est_seconds < 60:
    _est_str = f"~{_est_seconds} sec"
else:
    _est_str = f"~{_est_seconds // 60} min {_est_seconds % 60} sec"
st.sidebar.caption(
    f"📅 {sim_horizon} days × {num_replications} reps  \n"
    f"⏱ Estimated run time: **{_est_str}**"
)

st.sidebar.markdown("---")
run_sim_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# =============================================================================
# MAIN AREA - Flow Chart Input
# =============================================================================

st.title("📊 Teledermatology Simulation")
st.markdown("Enter your practice parameters for each stage of patient flow below.")

st.markdown("## Patient Flow & Parameters")

# -----------------------------------------------------------------------------
# ROW 1: eConsult path
# -----------------------------------------------------------------------------

row1_cols = st.columns([2.2, 0.2, 3.0, 0.2, 2.5, 0.2, 1.2])

# --- eConsult Referrals (col 0) ---
with row1_cols[0]:
    with st.container(border=True):
        st.markdown('<div class="flow-header-green">eConsult Referrals</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="icon-row"><span class="icon-symbol">◉</span>Arrivals</div>',
                    unsafe_allow_html=True)
        lambda_e = st.number_input(
            "Per day (average)",
            min_value=0,
            max_value=200,
            value=DEFAULT_LAMBDA_E,
            step=1,
            key="lambda_e",
            help=(
                "Average number of new eConsult referrals arriving per day. "
                "The actual daily count varies randomly — the simulation models arrivals "
                "as a Poisson process, meaning each day's count is drawn from a Poisson "
                "distribution with this mean. "
                "Example: with an average of 28, some days may see 22 and others 35. "
                "The standard deviation is about √mean (so about 5 when the mean is 28)."
            )
        )
        st.markdown("<br>", unsafe_allow_html=True)

# Arrow (col 1)
with row1_cols[1]:
    st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)

# --- eConsult Service (col 4) — render first so we know c_e before filling Queue col ---
with row1_cols[4]:
    with st.container(border=True):
        st.markdown('<div class="flow-header-green">eConsult Service</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="icon-row"><span class="icon-symbol">⚕</span>Practice capacity</div>',
                    unsafe_allow_html=True)
        # Service time in MINUTES per case (UI-facing); rate derived as 60/min
        min_per_case_e = st.number_input(
            "Minutes per eConsult case",
            min_value=1.0,
            max_value=120.0,
            value=DEFAULT_ECONSULT_MIN_PER_CASE,
            step=1.0,
            key="min_per_case_e",
            help=(
                "Average minutes one dermatologist spends per eConsult case. "
                "Example: 5 means each eConsult takes 5 minutes on average. "
                "Service rate (cases/hour) = 60 ÷ minutes per case."
            )
        )
        econsult_rate = 60.0 / min_per_case_e if min_per_case_e > 0 else 0.0
        st.caption(f"≈ {econsult_rate:.1f} cases/hour")
        
        # Side-by-side Mean / SD inputs for PRACTICE-LEVEL hours
        hrs_e_col1, hrs_e_col2 = st.columns([1, 1], gap="small")
        with hrs_e_col1:
            hrs_econsult = st.number_input(
                "Mean hrs/day (practice)",
                min_value=0.0,
                max_value=200.0,
                value=DEFAULT_HRS_ECONSULT,
                step=0.5,
                key="hrs_econsult",
                help=(
                    "Average total hours per day the practice spends on eConsult, "
                    "across all dermatologists working in parallel."
                )
            )
        with hrs_e_col2:
            hrs_e_sd = st.number_input(
                "SD hrs/day (practice)",
                min_value=0.0,
                max_value=24.0,
                value=DEFAULT_HRS_E_SD,
                step=0.25,
                key="hrs_e_sd",
                help=(
                    "Day-to-day variability in practice eConsult hours "
                    "(vacations, conferences, normal scheduling fluctuations). "
                    "SD = 0 means same hours every day. "
                    "Example: mean = 2, SD = 0.5 → most days are between 1 and 3 hrs."
                )
            )
        st.caption("SD = 0 means same hours every day (deterministic)")
        
        # --- Min required hours (practice-level) — uses floor for integer capacity ---
        if econsult_rate > 0 and lambda_e > 0:
            min_hrs_e = lambda_e / econsult_rate
            if hrs_econsult >= min_hrs_e:
                st.markdown(
                    f'<div class="min-hrs-ok">✓ Min required: '
                    f'<b>{min_hrs_e:.2f}</b> hrs/day</div>',
                    unsafe_allow_html=True
                )
            else:
                gap_e = min_hrs_e - hrs_econsult
                st.markdown(
                    f'<div class="min-hrs-bad">✗ Min required: '
                    f'<b>{min_hrs_e:.2f}</b> hrs/day '
                    f'— add <b>{gap_e:.2f}</b> hrs/day to keep up</div>',
                    unsafe_allow_html=True
                )

# Now we know c_e — compute it (FLOOR, since we can't see fractional patients)
c_e = int(np.floor(hrs_econsult * econsult_rate))

# --- eConsult Queue (col 2) — two sub-columns: Buffer | Waiting line ---
with row1_cols[2]:
    with st.container(border=True):
        st.markdown('<div class="flow-header-green">eConsult Queue</div>',
                    unsafe_allow_html=True)
        
        # Two side-by-side sub-columns inside the box
        q_buf_col, q_wait_col = st.columns([1, 1], gap="small")
        
        # Left: Buffer section
        with q_buf_col:
            st.markdown('<div class="section-label">Buffer</div>', unsafe_allow_html=True)
            buffer_days_e = st.number_input(
                "Buffer (days)",
                min_value=1,
                max_value=60,
                value=DEFAULT_BUFFER_DAYS_E,
                step=1,
                key="buffer_days_e",
                help=(
                    "Maximum wait time tolerated for new eConsult arrivals, in days. "
                    "Max queue size = buffer days × eConsult capacity. When the queue exceeds "
                    "this size, new arrivals are blocked (turned away) until the queue drops back below. "
                    "The initial backlog is NOT subject to this limit."
                )
            )
            max_q_e = buffer_days_e * c_e if c_e > 0 else 0
            st.markdown(
                f'<div class="max-queue-display">Max queue size: <b>{max_q_e}</b> patients '
                f'({buffer_days_e} days × {c_e} patients/day)</div>',
                unsafe_allow_html=True
            )
            st.caption(
                f"Daily capacity: {c_e} = floor({hrs_econsult:.2f} hrs × {econsult_rate:.2f} cases/hr), "
                f"rounded down since we can't serve a fractional patient."
            )
        
        # Right: Waiting line section
        with q_wait_col:
            st.markdown(
                '<div class="icon-row"><span class="icon-symbol">≡</span>Waiting line</div>',
                unsafe_allow_html=True
            )
            initial_q_e = st.number_input(
                "Current backlog",
                min_value=0,
                max_value=5000,
                value=DEFAULT_INITIAL_Q_E,
                step=5,
                key="initial_q_e",
                help=(
                    "Average number of eConsult cases currently waiting in the queue. "
                    "This is the starting state of the simulation. "
                    "Can exceed the buffer size — if so, new arrivals will be blocked until "
                    "the queue drops below the buffer."
                )
            )
            target_q_e = st.number_input(
                "Target level",
                min_value=0,
                max_value=max(initial_q_e, 1),
                value=min(DEFAULT_TARGET_Q_E, initial_q_e) if initial_q_e > 0 else 0,
                step=5,
                key="target_q_e",
                help=(
                    "The queue level you want to reach. The simulation tracks how many "
                    "days it takes for the eConsult queue to reach this number (on average "
                    "across replications) and how often that target is actually achieved."
                )
            )

# Arrow between Queue and Service (col 3)
with row1_cols[3]:
    st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)

# Arrow + Discharge (cols 5, 6)
with row1_cols[5]:
    st.markdown('<div class="flow-arrow" style="color: #0F6E56;">→</div>',
                unsafe_allow_html=True)

with row1_cols[6]:
    st.markdown(
        '<div class="discharge-label">Discharge</div>'
        '<div class="discharge-caption">(not converted)</div>',
        unsafe_allow_html=True
    )

# --- Inline warning strip under eConsult Service ---
warn_row1 = st.columns([2.2, 0.2, 3.0, 0.2, 2.5, 0.2, 1.2])
with warn_row1[4]:
    if c_e < lambda_e:
        st.markdown(
            f'<div class="inline-warning-error">'
            f'⚠ Capacity ({c_e}/day) &lt; Demand ({lambda_e}/day). '
            f'Backlog will grow indefinitely.'
            f'</div>',
            unsafe_allow_html=True
        )
    elif c_e == lambda_e:
        st.markdown(
            f'<div class="inline-warning-warn">'
            f'⚠ Capacity ({c_e}/day) equals demand ({lambda_e}/day). '
            f'Backlog will not shrink.'
            f'</div>',
            unsafe_allow_html=True
        )

# -----------------------------------------------------------------------------
# MIDDLE: Conversion Rate (spans between rows)
# -----------------------------------------------------------------------------

# Vertical connector arrow from eConsult Service down to Conversion Rate.
# Rendered with negative top margin so it overlaps upward into the warning-row space
# (just beneath eConsult Service). Absolute-positioned SVG keeps Streamlit layout intact.
st.markdown(
    """
    <div style="position: relative; width: 100%; height: 28px;
                margin-top: -8px; margin-bottom: -8px; pointer-events: none;">
      <svg width="100%" height="28" viewBox="0 0 1000 28" preserveAspectRatio="none"
           style="position: absolute; top: 0; left: 0; pointer-events: none;">
        <defs>
          <marker id="conv-down-head" viewBox="0 0 10 10" refX="8" refY="5"
                  markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#D4A017"/>
          </marker>
        </defs>
        <!-- Vertical line in the eConsult Service column (centered roughly at x=735) -->
        <line x1="735" y1="0" x2="735" y2="22"
              stroke="#D4A017" stroke-width="2.5"
              marker-end="url(#conv-down-head)"
              vector-effect="non-scaling-stroke"/>
      </svg>
    </div>
    """,
    unsafe_allow_html=True
)

conv_cols = st.columns([2.2, 0.2, 3.0, 0.2, 2.5, 0.2, 1.2])

with conv_cols[4]:
    with st.container(border=True):
        st.markdown('<div class="flow-header-amber">Conversion Rate</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="icon-row" style="color: #5F5E5A;">'
            '<span style="color: #D4A017; font-size: 18px; margin-right: 6px;">⇣</span>'
            'eConsult → FTF follow-up</div>',
            unsafe_allow_html=True
        )
        gamma = st.number_input(
            "Fraction needing FTF",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_GAMMA,
            step=0.05,
            key="gamma",
            help=(
                "On average, the fraction of eConsult patients who need FTF follow-up. "
                "**Enter as a decimal between 0 and 1** (e.g., 0.40 means 40%, not 40). "
                "For each eConsult patient served, the simulation flips a weighted coin "
                "with this probability to decide whether they convert to FTF."
            )
        )
        st.caption(f"≈ {gamma:.0%} of eConsults convert to FTF (enter as decimal 0–1)")

# L-shaped arrow from left side of Conversion Rate box down into FTF Queue.
# The SVG is pulled up with negative margin so its drawing area overlaps the
# Conversion Rate box vertically. The horizontal line aligns with the middle
# of the box, visually originating from the box's left edge.
#
# The text label is rendered as a SEPARATE HTML element (absolutely positioned
# inside the same wrapper div) instead of an SVG <text>. This avoids the font
# distortion that happens when the SVG is stretched non-uniformly via
# preserveAspectRatio="none". HTML text always renders at native font quality.
converted_flow = gamma * lambda_e  # patients/day flowing eConsult -> FTF
arrow_label = f"Converted to FTF: <b>{converted_flow:.2f}</b>/day ({gamma:.2f} × {lambda_e})"
st.markdown(
    f"""
    <div style="position: relative; width: 100%; height: 190px;
                margin-top: -130px; margin-bottom: -10px; pointer-events: none;">
      <svg width="100%" height="190" viewBox="0 0 1000 190" preserveAspectRatio="none"
           style="position: absolute; top: 0; left: 0; pointer-events: none;">
        <defs>
          <marker id="conv-arrowhead" viewBox="0 0 10 10" refX="8" refY="5"
                  markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#D4A017"/>
          </marker>
        </defs>
        <!-- L-shape:
             (1) horizontal: from left edge of Conversion Rate box (~x=580)
                 at its vertical middle (~y=60) leftward to FTF Queue column (x=420)
             (2) vertical: down to top of FTF Queue (y=180) -->
        <polyline points="580,60 420,60 420,180"
                  fill="none"
                  stroke="#D4A017" stroke-width="2.5"
                  stroke-linecap="round" stroke-linejoin="round"
                  marker-end="url(#conv-arrowhead)"
                  vector-effect="non-scaling-stroke"/>
      </svg>
      <!-- HTML text label positioned above the horizontal segment.
           top: ~22% of 190px = ~42px (above y=60 in SVG coords).
           left: 42% (corresponds to midpoint of horizontal segment, x=500/1000). -->
      <div style="position: absolute; top: 14px; left: 50%;
                  transform: translateX(-50%);
                  color: #D4A017; font-size: 13px; font-weight: 600;
                  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                               Roboto, sans-serif;
                  white-space: nowrap;
                  background-color: rgba(255, 255, 255, 0.95);
                  padding: 2px 8px; border-radius: 4px;">
        {arrow_label}
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# ROW 2: FTF path
# -----------------------------------------------------------------------------

row2_cols = st.columns([2.2, 0.2, 3.0, 0.2, 2.5, 0.2, 1.2])

# --- Direct FTF Referrals (col 0) ---
with row2_cols[0]:
    with st.container(border=True):
        st.markdown('<div class="flow-header-blue">Direct FTF Referrals</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="icon-row"><span class="icon-symbol icon-symbol-blue">◉</span>Arrivals</div>',
            unsafe_allow_html=True
        )
        lambda_d = st.number_input(
            "Per day (average)",
            min_value=0,
            max_value=200,
            value=DEFAULT_LAMBDA_D,
            step=1,
            key="lambda_d",
            help=(
                "Average number of direct FTF referrals per day (patients who skip "
                "eConsult and go straight to FTF scheduling). "
                "Like eConsult arrivals, the actual daily count varies randomly and "
                "is modeled as a Poisson process. "
                "Example: with an average of 34, daily counts typically range between "
                "28 and 40 (standard deviation ≈ √34 ≈ 5.8)."
            )
        )
        st.markdown("<br>", unsafe_allow_html=True)

# Arrow (col 1)
with row2_cols[1]:
    st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)

# --- FTF Service (col 4) — render first so we know c_f ---
with row2_cols[4]:
    with st.container(border=True):
        st.markdown('<div class="flow-header-blue">FTF Service</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="icon-row"><span class="icon-symbol icon-symbol-blue">⚕</span>Practice capacity</div>',
            unsafe_allow_html=True
        )
        # Service time in MINUTES per case; rate derived as 60/min
        min_per_case_f = st.number_input(
            "Minutes per FTF case",
            min_value=1.0,
            max_value=120.0,
            value=DEFAULT_FTF_MIN_PER_CASE,
            step=1.0,
            key="min_per_case_f",
            help=(
                "Average minutes one dermatologist spends per FTF visit. "
                "Example: 15 means each FTF visit takes 15 minutes on average. "
                "Service rate (cases/hour) = 60 ÷ minutes per case."
            )
        )
        ftf_rate = 60.0 / min_per_case_f if min_per_case_f > 0 else 0.0
        st.caption(f"≈ {ftf_rate:.1f} cases/hour")
        
        # ---- Cluster-based capacity input ----
        st.markdown("**Capacity by cluster:**")
        st.caption(
            "Each row represents a group of identical physicians "
            "(same hours, variability, and absence rate). "
            "Use **+ Add cluster** for more groups."
        )
        
        clusters = st.session_state.ftf_clusters
        
        # Render header row
        hdr_c1, hdr_c2, hdr_c3, hdr_c4 = st.columns([2, 2, 2, 1])
        with hdr_c1:
            st.caption("Hours/day")
        with hdr_c2:
            st.caption("SD (hrs)")
        with hdr_c3:
            st.caption(
                "P(absent) [0–1]",
                help=(
                    "Probability that this cluster of providers is absent on a given day "
                    "(sick, on vacation, at a conference, etc.). "
                    "**Enter as a decimal between 0 and 1** — e.g., 0.05 means 5%. "
                    "On absent days, the cluster contributes 0 hours; on present days, "
                    "it contributes its full Hours/day (with SD-driven variation if set)."
                )
            )
        with hdr_c4:
            st.caption(" ")
        
        # Track indices to remove (deferred so we don't mutate during iter)
        indices_to_remove = []
        
        for idx, cluster in enumerate(clusters):
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            with c1:
                new_hours = st.number_input(
                    "",
                    min_value=0.0,
                    max_value=200.0,
                    value=float(cluster.get('hours', 6.0)),
                    step=0.5,
                    key=f"ftf_cluster_hours_{idx}",
                    label_visibility="collapsed",
                    help=(
                        "Mean total hours per day contributed by this cluster of providers "
                        "(on days when they are not absent)."
                    )
                )
            with c2:
                new_sd = st.number_input(
                    "",
                    min_value=0.0,
                    max_value=24.0,
                    value=float(cluster.get('sd', 0.0)),
                    step=0.1,
                    key=f"ftf_cluster_sd_{idx}",
                    label_visibility="collapsed",
                    help=(
                        "Standard deviation of daily working hours for this cluster. "
                        "Captures day-to-day variation around the mean (e.g., scheduling fluctuations). "
                        "Set to 0 for a fixed daily amount."
                    )
                )
            with c3:
                new_p_absent = st.number_input(
                    "",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(cluster.get('p_absent', 0.0)),
                    step=0.01,
                    format="%.2f",
                    key=f"ftf_cluster_p_{idx}",
                    label_visibility="collapsed",
                    help=(
                        "Probability that this cluster of providers is absent on a given day "
                        "(sick, on vacation, at a conference, etc.). "
                        "**Enter as a decimal between 0 and 1** (e.g., 0.05 means 5%, not 5). "
                        "On absent days, this cluster contributes 0 hours. "
                        "Set to 0 for a cluster that is always available."
                    )
                )
            with c4:
                # Don't allow removing the last cluster
                if len(clusters) > 1:
                    if st.button("✕", key=f"ftf_cluster_rm_{idx}",
                                 help="Remove this cluster"):
                        indices_to_remove.append(idx)
                else:
                    st.caption(" ")  # placeholder for alignment
            
            # Persist edits (only if not being removed)
            if idx not in indices_to_remove:
                clusters[idx] = {
                    'hours': new_hours,
                    'sd': new_sd,
                    'p_absent': new_p_absent,
                }
        
        # Apply removals (in reverse so indices stay valid)
        if indices_to_remove:
            for i in sorted(indices_to_remove, reverse=True):
                clusters.pop(i)
            st.session_state.ftf_clusters = clusters
            st.rerun()
        
        # Add-cluster button
        if st.button("➕ Add cluster", key="ftf_add_cluster", help="Add another physician cluster"):
            clusters.append({'hours': 4.0, 'sd': 0.0, 'p_absent': 0.0})
            st.session_state.ftf_clusters = clusters
            st.rerun()
        
        # Compute practice-wide totals (mean hours sum)
        hrs_ftf = sum(c.get('hours', 0.0) for c in clusters)
        # Note: SD doesn't simply sum across clusters; we don't display a practice-wide SD here.
        # Effective mean accounts for absence: sum(hours × (1 - p_absent))
        effective_hrs_ftf = sum(c.get('hours', 0.0) * (1.0 - c.get('p_absent', 0.0))
                                 for c in clusters)
        
        st.caption(
            f"**Practice total** (sum across {len(clusters)} cluster"
            f"{'s' if len(clusters) > 1 else ''}): "
            f"<b>{hrs_ftf:.2f}</b> hrs/day mean",
            unsafe_allow_html=True
        )
        if effective_hrs_ftf < hrs_ftf:
            st.caption(
                f"Effective mean (accounting for absences): <b>{effective_hrs_ftf:.2f}</b> hrs/day",
                unsafe_allow_html=True
            )
        
        # --- Min required hours (practice-level) ---
        if ftf_rate > 0:
            effective_demand = gamma * lambda_e + lambda_d
            if effective_demand > 0:
                min_hrs_f = effective_demand / ftf_rate
                if effective_hrs_ftf >= min_hrs_f:
                    st.markdown(
                        f'<div class="min-hrs-ok">✓ Min required (effective): '
                        f'<b>{min_hrs_f:.2f}</b> hrs/day</div>',
                        unsafe_allow_html=True
                    )
                else:
                    gap_f = min_hrs_f - effective_hrs_ftf
                    st.markdown(
                        f'<div class="min-hrs-bad">✗ Min required (effective): '
                        f'<b>{min_hrs_f:.2f}</b> hrs/day '
                        f'— add <b>{gap_f:.2f}</b> hrs/day mean to keep up</div>',
                        unsafe_allow_html=True
                    )

# Now we know c_f (FLOOR for integer capacity, based on EFFECTIVE hours
# i.e., mean accounting for expected absences — used for buffer sizing and theoretical rho)
c_f = int(np.floor(effective_hrs_ftf * ftf_rate))
effective_ftf_demand = gamma * lambda_e + lambda_d

# --- FTF Queue (col 2) — two sub-columns: Buffer | Waiting line ---
with row2_cols[2]:
    with st.container(border=True):
        st.markdown('<div class="flow-header-blue">FTF Queue</div>',
                    unsafe_allow_html=True)
        
        # Two side-by-side sub-columns inside the box
        fq_buf_col, fq_wait_col = st.columns([1, 1], gap="small")
        
        # Left: Buffer section
        with fq_buf_col:
            st.markdown('<div class="section-label">Buffer</div>', unsafe_allow_html=True)
            buffer_days_f = st.number_input(
                "Buffer (days)",
                min_value=1,
                max_value=60,
                value=DEFAULT_BUFFER_DAYS_F,
                step=1,
                key="buffer_days_f",
                help=(
                    "Maximum wait time tolerated for new FTF arrivals, in days. "
                    "Max queue size = buffer days × FTF capacity. When the queue exceeds "
                    "this size, new arrivals (both direct and converted from eConsult) are blocked "
                    "until the queue drops back below. The initial backlog is NOT subject to this limit."
                )
            )
            max_q_f = buffer_days_f * c_f if c_f > 0 else 0
            st.markdown(
                f'<div class="max-queue-display">Max queue size: <b>{max_q_f}</b> patients '
                f'({buffer_days_f} days × {c_f} patients/day)</div>',
                unsafe_allow_html=True
            )
            # FTF capacity comes from cluster sums; show the derivation
            if len(clusters) == 1:
                _cluster = clusters[0]
                _eff_h = _cluster['hours'] * (1.0 - _cluster.get('p_absent', 0.0))
                if _cluster.get('p_absent', 0.0) > 0:
                    st.caption(
                        f"Daily capacity: {c_f} = floor({_eff_h:.2f} effective hrs × {ftf_rate:.2f} cases/hr); "
                        f"effective hrs = {_cluster['hours']:.2f} × (1 - {_cluster.get('p_absent', 0.0):.2f} absence) "
                        f"= {_eff_h:.2f}. Rounded down."
                    )
                else:
                    st.caption(
                        f"Daily capacity: {c_f} = floor({_cluster['hours']:.2f} hrs × {ftf_rate:.2f} cases/hr), "
                        f"rounded down since we can't serve a fractional patient."
                    )
            else:
                # Multi-cluster: show sum across clusters
                _eff_total = sum(c['hours'] * (1.0 - c.get('p_absent', 0.0)) for c in clusters)
                _has_absent = any(c.get('p_absent', 0.0) > 0 for c in clusters)
                if _has_absent:
                    st.caption(
                        f"Daily capacity: {c_f} = floor({_eff_total:.2f} effective hrs × {ftf_rate:.2f} cases/hr); "
                        f"effective hrs sums across {len(clusters)} clusters, "
                        f"accounting for each cluster's absence rate. Rounded down."
                    )
                else:
                    st.caption(
                        f"Daily capacity: {c_f} = floor({_eff_total:.2f} total hrs × {ftf_rate:.2f} cases/hr); "
                        f"total hrs = sum across {len(clusters)} clusters. Rounded down."
                    )
        
        # Right: Waiting line section
        with fq_wait_col:
            st.markdown(
                '<div class="icon-row"><span class="icon-symbol icon-symbol-blue">≡</span>Waiting line</div>',
                unsafe_allow_html=True
            )
            initial_q_f = st.number_input(
                "Current backlog",
                min_value=0,
                max_value=5000,
                value=DEFAULT_INITIAL_Q_F,
                step=5,
                key="initial_q_f",
                help=(
                    "Average number of FTF appointments currently waiting in the queue. "
                    "This is the starting state of the simulation and includes both "
                    "direct referrals and prior conversions from eConsult. "
                    "Can exceed the buffer size — if so, new arrivals will be blocked until "
                    "the queue drops below the buffer."
                )
            )
            target_q_f = st.number_input(
                "Target level",
                min_value=0,
                max_value=max(initial_q_f, 1),
                value=min(DEFAULT_TARGET_Q_F, initial_q_f) if initial_q_f > 0 else 0,
                step=5,
                key="target_q_f",
                help=(
                    "The queue level you want to reach. The simulation tracks how many "
                    "days it takes for the FTF queue to reach this number (on average "
                    "across replications) and how often that target is actually achieved."
                )
            )

# Arrow between Queue and Service (col 3)
with row2_cols[3]:
    st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)

# Arrow + Discharge (cols 5, 6)
with row2_cols[5]:
    st.markdown('<div class="flow-arrow" style="color: #0F6E56;">→</div>',
                unsafe_allow_html=True)

with row2_cols[6]:
    st.markdown('<div class="discharge-label">Discharge</div>',
                unsafe_allow_html=True)

# --- Inline warning strip under FTF Service ---
warn_row2 = st.columns([2.2, 0.2, 3.0, 0.2, 2.5, 0.2, 1.2])
with warn_row2[4]:
    if c_f < effective_ftf_demand:
        st.markdown(
            f'<div class="inline-warning-error">'
            f'⚠ Capacity ({c_f}/day) &lt; Effective demand ({effective_ftf_demand:.1f}/day). '
            f'Backlog will grow indefinitely.'
            f'</div>',
            unsafe_allow_html=True
        )
    elif abs(c_f - effective_ftf_demand) < 0.01:
        st.markdown(
            f'<div class="inline-warning-warn">'
            f'⚠ Capacity ({c_f}/day) equals effective demand ({effective_ftf_demand:.1f}/day). '
            f'Backlog will not shrink.'
            f'</div>',
            unsafe_allow_html=True
        )

# -----------------------------------------------------------------------------
# SHARED SETTINGS ROW (Buffer + derived metrics)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# DERIVED SUMMARY (compact — no buffer, no duplicated warnings since inline above)
# -----------------------------------------------------------------------------

total_hrs = hrs_econsult + hrs_ftf
efficiency_ratio = econsult_rate / ftf_rate if ftf_rate > 0 else 1.0

st.markdown("### Summary")
summary_cols = st.columns(3)

with summary_cols[0]:
    st.markdown("**Daily capacity**")
    st.caption(
        f"eConsult: **{c_e}** patients/day",
        help=f"floor({hrs_econsult:.2f} mean hrs × {econsult_rate:.2f} cases/hr) = {c_e}. Rounded down since we can't serve a fractional patient."
    )
    _ftf_eff_total = sum(c['hours'] * (1.0 - c.get('p_absent', 0.0)) for c in clusters) if clusters else hrs_ftf
    st.caption(
        f"FTF: **{c_f}** patients/day",
        help=(
            f"floor({_ftf_eff_total:.2f} effective hrs × {ftf_rate:.2f} cases/hr) = {c_f}. "
            f"Effective hrs sums across {len(clusters) if clusters else 1} cluster"
            f"{'s' if len(clusters or [0]) > 1 else ''}, accounting for each cluster's absence rate. Rounded down."
        )
    )
    st.caption(f"Total: **{total_hrs}** hrs/day")

with summary_cols[1]:
    st.markdown("**Demand**")
    st.caption(f"eConsult: **{lambda_e}** patients/day")
    st.caption(
        f"Effective FTF: **{effective_ftf_demand:.1f}** patients/day",
        help=(
            "Effective FTF demand = direct FTF arrivals + (conversion rate × eConsult arrivals). "
            "This includes patients converted from eConsult to FTF."
        )
    )
    st.caption(
        f"Efficiency ratio: **{efficiency_ratio:.2f}×**",
        help=(
            "How many eConsult cases can be done per hour relative to FTF cases. "
            "Calculated as (eConsult cases/hour) ÷ (FTF cases/hour). "
            "A ratio of 3× means eConsults complete 3 times faster per hour of work."
        )
    )

with summary_cols[2]:
    st.markdown("**Max queue size**")
    st.caption(
        f"eConsult: **{max_q_e}** patients ({buffer_days_e} days × {c_e})",
        help=(
            "Maximum number of patients allowed in the queue before new arrivals are blocked. "
            "Calculated as buffer days × mean daily capacity. "
            "Existing initial backlog can exceed this — those patients are still served, "
            "but no new patients can join while the queue is above the max."
        )
    )
    st.caption(
        f"FTF: **{max_q_f}** patients ({buffer_days_f} days × {c_f})",
        help=(
            "Maximum number of patients allowed in the queue before new arrivals are blocked."
        )
    )

# =============================================================================
# RUN SIMULATION
# =============================================================================

if run_sim_button:
    if c_e == 0 and c_f == 0:
        st.error("Please allocate some hours to at least one service type.")
        st.stop()
    
    progress_bar = st.progress(0, text="Initializing simulation...")
    st.caption("ℹ️ To cancel a long-running simulation, refresh the page.")
    
    sim_params = {
        'buffer_days_e': buffer_days_e,
        'buffer_days_f': buffer_days_f,
        'lambda_e': lambda_e,
        'lambda_d': lambda_d,
        'c_e': c_e,
        'c_f': c_f,
        'gamma': gamma,
        'sim_horizon': sim_horizon,
        'num_replications': num_replications,
        'warmup_fraction': 0.5,
        'initial_q_e': initial_q_e,
        'initial_q_f': initial_q_f,
        'target_q_e': target_q_e,
        'target_q_f': target_q_f,
        # Service rates (derived from minutes per case)
        'min_per_case_e': min_per_case_e,
        'min_per_case_f': min_per_case_f,
        'econsult_rate': econsult_rate,
        'ftf_rate': ftf_rate,
        # Practice-level hours
        'hrs_econsult': hrs_econsult,
        'hrs_ftf': hrs_ftf,
        # Variability: eConsult Gaussian (single number)
        'hrs_e_sd': hrs_e_sd,
        # FTF: cluster-based variability — list of dicts
        'ftf_clusters': [dict(c) for c in st.session_state.ftf_clusters],
        'efficiency_ratio': efficiency_ratio,
        'effective_ftf_demand': effective_ftf_demand,
    }
    
    def update_progress(current, total, phase):
        if phase == "simulation":
            frac = current / total if total > 0 else 0
            progress_bar.progress(frac, text=f"Running replication {current + 1} of {total}...")
        elif phase == "analysis":
            progress_bar.progress(1.0, text="Analyzing results...")
    
    import time as _time
    _t_start = _time.time()
    results = run_simulation(sim_params, progress_callback=update_progress)
    _t_elapsed = _time.time() - _t_start
    
    # Update calibration: sec per (rep × day), based on this measured run.
    # Use a smoothing factor — blend new measurement with old (50/50)
    # so single anomalous runs don't whipsaw the estimate.
    _measured_sec_per_rep_day = _t_elapsed / max(1, num_replications * sim_horizon)
    _old_calibration = st.session_state.get('sec_per_rep_day', 0.00003)
    st.session_state.sec_per_rep_day = 0.5 * _old_calibration + 0.5 * _measured_sec_per_rep_day
    
    st.session_state.sim_results = results
    st.session_state.sim_params = sim_params
    
    progress_bar.empty()
    st.success(f"Simulation complete! ({_t_elapsed:.1f} sec)")

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

if st.session_state.sim_results is not None:
    results = st.session_state.sim_results
    params = st.session_state.sim_params
    
    warmup_day = results['warmup_day']
    sim_horizon_result = params['sim_horizon']
    
    def fmt_sd(mean, sd, decimals=1, suffix=''):
        """Format a value with ± SD. SD shown only if defined and >= 2 reps."""
        m_str = f"{mean:.{decimals}f}"
        if sd is None:
            return f"{m_str}{suffix}"
        return f"{m_str} ± {sd:.{decimals}f}{suffix}"
    
    # -------------------------------------------------------------------------
    # BACKLOG REDUCTION ANALYSIS
    # -------------------------------------------------------------------------
    
    st.header("📉 Backlog Reduction Analysis")
    
    col1, col2 = st.columns(2)
    
    # Pre-compute whether the MEAN queue trajectory crosses each target
    # (this is the same check used to decide whether to draw the vertical line on the plot)
    _q_e_mean_arr = results['daily_data']['q_e']
    _q_f_mean_arr = results['daily_data']['q_f']
    _e_mean_crosses = np.any(_q_e_mean_arr <= params['target_q_e'])
    _f_mean_crosses = np.any(_q_f_mean_arr <= params['target_q_f'])
    
    with col1:
        if results['avg_target_day_e'] is not None and _e_mean_crosses:
            st.metric(
                "Days to reach eConsult target",
                fmt_sd(results['avg_target_day_e'], results['sd_target_day_e'], decimals=0, suffix=' days'),
                delta=f"{results['pct_reached_target_e']:.0f}% of runs reached target",
                delta_color="normal"
            )
        elif results['avg_target_day_e'] is not None and not _e_mean_crosses:
            # Some runs reached the target but the mean trajectory does not cross —
            # the average day-reached would appear in empty space on the plot.
            st.metric(
                "Days to reach eConsult target",
                "Not reliably reached",
                delta=f"Only {results['pct_reached_target_e']:.0f}% of runs reached target",
                delta_color="off",
                help=(
                    "The average across runs that reached the target exists, but the "
                    "mean queue trajectory across all runs never crosses the target line. "
                    "Most runs do not reach the target, so the average day is not representative."
                )
            )
        else:
            st.metric(
                "Days to reach eConsult target",
                "Not reached",
                delta="0% of runs reached target",
                delta_color="off"
            )
    
    with col2:
        if results['avg_target_day_f'] is not None and _f_mean_crosses:
            st.metric(
                "Days to reach FTF target",
                fmt_sd(results['avg_target_day_f'], results['sd_target_day_f'], decimals=0, suffix=' days'),
                delta=f"{results['pct_reached_target_f']:.0f}% of runs reached target",
                delta_color="normal"
            )
        elif results['avg_target_day_f'] is not None and not _f_mean_crosses:
            st.metric(
                "Days to reach FTF target",
                "Not reliably reached",
                delta=f"Only {results['pct_reached_target_f']:.0f}% of runs reached target",
                delta_color="off",
                help=(
                    "The average across runs that reached the target exists, but the "
                    "mean queue trajectory across all runs never crosses the target line. "
                    "Most runs do not reach the target, so the average day is not representative."
                )
            )
        else:
            st.metric(
                "Days to reach FTF target",
                "Not reached",
                delta="0% of runs reached target",
                delta_color="off"
            )
    
    # -------------------------------------------------------------------------
    # QUEUE TRAJECTORY PLOTS
    # -------------------------------------------------------------------------
    
    st.subheader("Queue Trajectories")
    
    days = results['daily_data']['day']
    
    # Plot: Separate eConsult and FTF queues (total queue plot removed
    # per UI feedback — total can be misleading when the two queues are
    # very different in scale)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # eConsult queue (GREEN)
    ax1 = axes[0]
    q_e_mean = results['daily_data']['q_e']
    q_e_std = results['daily_data']['q_e_std']
    ax1.fill_between(days, q_e_mean - q_e_std, q_e_mean + q_e_std,
                     color=COLOR_ECONSULT, alpha=0.15, label='± 1 SD')
    ax1.plot(days, q_e_mean, color=COLOR_ECONSULT, linewidth=1.5, label='eConsult Queue (mean)')
    ax1.axhline(y=params['initial_q_e'], color='red', linestyle='--', alpha=0.5, label=f'Initial ({params["initial_q_e"]})')
    ax1.axhline(y=params['target_q_e'], color='green', linestyle='--', alpha=0.7, label=f'Target ({params["target_q_e"]})')
    ax1.axvline(x=warmup_day, color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Long-term analysis starts')
    ax1.axvspan(0, warmup_day, alpha=0.1, color='orange')
    
    if results['avg_target_day_e'] is not None:
        # Only draw the "Target reached" annotation if the MEAN queue trajectory
        # actually crosses the target line. Otherwise the line would appear in
        # empty space (because avg_target_day is averaged only over the reps
        # that reached the target, while the plotted curve is the mean over ALL reps).
        q_e_mean_arr = results['daily_data']['q_e']
        if np.any(q_e_mean_arr <= params['target_q_e']):
            ax1.axvline(x=results['avg_target_day_e'], color='green', linestyle=':', alpha=0.7)
            ax1.annotate(f"Target reached\nDay {results['avg_target_day_e']:.0f}",
                         xy=(results['avg_target_day_e'], params['target_q_e']),
                         xytext=(results['avg_target_day_e'] + sim_horizon_result*0.05, params['target_q_e'] + params['initial_q_e']*0.1),
                         fontsize=8, color='green',
                         arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))
    
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Queue Length')
    ax1.set_title('eConsult Queue Over Time')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(0, sim_horizon_result)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    
    # FTF queue (BLUE)
    ax2 = axes[1]
    q_f_mean = results['daily_data']['q_f']
    q_f_std = results['daily_data']['q_f_std']
    ax2.fill_between(days, q_f_mean - q_f_std, q_f_mean + q_f_std,
                     color=COLOR_FTF, alpha=0.15, label='± 1 SD')
    ax2.plot(days, q_f_mean, color=COLOR_FTF, linewidth=1.5, label='FTF Queue (mean)')
    ax2.axhline(y=params['initial_q_f'], color='red', linestyle='--', alpha=0.5, label=f'Initial ({params["initial_q_f"]})')
    ax2.axhline(y=params['target_q_f'], color='green', linestyle='--', alpha=0.7, label=f'Target ({params["target_q_f"]})')
    ax2.axvline(x=warmup_day, color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Long-term analysis starts')
    ax2.axvspan(0, warmup_day, alpha=0.1, color='orange')
    
    if results['avg_target_day_f'] is not None:
        # Only draw the "Target reached" annotation if the MEAN queue trajectory
        # actually crosses the target line. See eConsult comment above.
        q_f_mean_arr = results['daily_data']['q_f']
        if np.any(q_f_mean_arr <= params['target_q_f']):
            ax2.axvline(x=results['avg_target_day_f'], color='green', linestyle=':', alpha=0.7)
            ax2.annotate(f"Target reached\nDay {results['avg_target_day_f']:.0f}",
                         xy=(results['avg_target_day_f'], params['target_q_f']),
                         xytext=(results['avg_target_day_f'] + sim_horizon_result*0.05, params['target_q_f'] + params['initial_q_f']*0.1),
                         fontsize=8, color='green',
                         arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))
    
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Queue Length')
    ax2.set_title('FTF Queue Over Time')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(0, sim_horizon_result)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # -------------------------------------------------------------------------
    # LONG-TERM ANALYSIS RESULTS
    # -------------------------------------------------------------------------
    
    st.header("📊 Long-Term Analysis Results")
    st.caption(f"Metrics computed from day {warmup_day} to day {sim_horizon_result} (long-term analysis range)")
    
    # Wait Times
    st.subheader("⏱️ Average Wait Times")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("eConsult (resolved)", fmt_sd(results['avg_wait_e'], results['sd_wait_e'], 1, ' days'))
    with col2:
        st.metric("FTF (all patients)", fmt_sd(results['avg_wait_f'], results['sd_wait_f'], 1, ' days'))
    with col3:
        st.metric("Direct FTF", fmt_sd(results['avg_wait_direct'], results['sd_wait_direct'], 1, ' days'))
    with col4:
        st.metric("Converted (total)", fmt_sd(results['avg_wait_converted'], results['sd_wait_converted'], 1, ' days'))
    with col5:
        st.metric("Weighted Average", fmt_sd(results['weighted_avg_wait'], results['sd_weighted_wait'], 1, ' days'))
    
    # Service Distribution
    st.subheader("👥 Service Distribution")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
        labels = ['Resolved\n(eConsult only)', 'Converted\n(eConsult→FTF)', 'Direct FTF']
        sizes = [results['n_resolved'], results['n_converted'], results['n_direct']]
        colors = [COLOR_ECONSULT, '#FFA500', COLOR_FTF]
        
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax_pie.pie(
                sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                startangle=90, textprops={'fontsize': 9}
            )
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
        ax_pie.set_title('Service Distribution', fontsize=12)
        st.pyplot(fig_pie)
        plt.close()
    
    with col2:
        st.markdown("**Patient Counts (Long-Term Analysis Period)**")
        
        total_patients = results['n_resolved'] + results['n_converted'] + results['n_direct']
        
        patient_data = {
            'Category': ['Resolved via eConsult', 'Converted to FTF', 'Direct FTF', '**Total**'],
            'Count': [
                f"{results['n_resolved']:,}",
                f"{results['n_converted']:,}",
                f"{results['n_direct']:,}",
                f"**{total_patients:,}**"
            ],
            'Percentage': [
                f"{results['n_resolved']/total_patients*100:.1f}%" if total_patients > 0 else "0%",
                f"{results['n_converted']/total_patients*100:.1f}%" if total_patients > 0 else "0%",
                f"{results['n_direct']/total_patients*100:.1f}%" if total_patients > 0 else "0%",
                "100%"
            ]
        }
        st.dataframe(pd.DataFrame(patient_data), hide_index=True, use_container_width=True)
    
    # Utilization
    st.subheader("⚡ Utilization")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_util, ax_util = plt.subplots(figsize=(8, 4))
        
        x = np.arange(3)
        width = 0.35
        
        total_theoretical = (params['c_e'] * results['rho_e_theoretical'] + params['c_f'] * results['rho_f_theoretical']) / (params['c_e'] + params['c_f']) if (params['c_e'] + params['c_f']) > 0 else 0
        total_empirical = (params['c_e'] * results['rho_e_empirical'] + params['c_f'] * results['rho_f_empirical']) / (params['c_e'] + params['c_f']) if (params['c_e'] + params['c_f']) > 0 else 0
        
        theoretical = [total_theoretical, results['rho_e_theoretical'], results['rho_f_theoretical']]
        empirical = [total_empirical, results['rho_e_empirical'], results['rho_f_empirical']]
        
        bar_colors = [COLOR_TOTAL, COLOR_ECONSULT, COLOR_FTF]
        
        bars1 = ax_util.bar(x - width/2, theoretical, width, label='Theoretical',
                            color=bar_colors, alpha=0.5, edgecolor='black', linewidth=1)
        bars2 = ax_util.bar(x + width/2, empirical, width, label='Empirical',
                            color=bar_colors, alpha=0.9, edgecolor='black', linewidth=1)
        
        def add_labels(bars, values):
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax_util.annotate(f'{val:.2f}',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3),
                                 textcoords="offset points",
                                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        add_labels(bars1, theoretical)
        add_labels(bars2, empirical)
        
        ax_util.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Capacity limit (100%)')
        ax_util.set_ylabel('Utilization', fontsize=11)
        ax_util.set_xticks(x)
        ax_util.set_xticklabels(['Total\n(Weighted)', 'eConsult', 'FTF'], fontsize=10)
        ax_util.legend(fontsize=9, loc='upper right')
        ax_util.set_ylim(0, max(1.3, max(theoretical + empirical) * 1.15))
        ax_util.set_title('Utilization by Service Type', fontsize=12)
        ax_util.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig_util)
        plt.close()
    
    with col2:
        st.markdown("**Utilization Details**")
        
        util_data = {
            'Service': ['eConsult', 'FTF', 'Total (Weighted)'],
            'Theoretical': [
                f"{results['rho_e_theoretical']:.3f}",
                f"{results['rho_f_theoretical']:.3f}",
                f"{total_theoretical:.3f}"
            ],
            'Empirical (mean ± SD)': [
                fmt_sd(results['rho_e_empirical'], results['sd_rho_e_empirical'], 3),
                fmt_sd(results['rho_f_empirical'], results['sd_rho_f_empirical'], 3),
                f"{total_empirical:.3f}"  # weighted blend; no per-rep value tracked
            ],
            'Status': [
                "✅ OK" if results['rho_e_theoretical'] < 1 else "⚠️ Overloaded",
                "✅ OK" if results['rho_f_theoretical'] < 1 else "⚠️ Overloaded",
                "✅ OK" if total_theoretical < 1 else "⚠️ Overloaded"
            ]
        }
        st.dataframe(pd.DataFrame(util_data), hide_index=True, use_container_width=True)
        
        st.caption("""
        **Theoretical utilization** = arrival rate / capacity  
        **Empirical utilization** = actual served / available capacity (± SD across replications)
        """)
    
    # Blocking Rates
    st.subheader("🚫 Blocking Rates")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("eConsult Blocking Rate", fmt_sd(results['block_rate_e'], results['sd_block_rate_e'], 2, '%'))
    with col2:
        st.metric("FTF Blocking Rate", fmt_sd(results['block_rate_f'], results['sd_block_rate_f'], 2, '%'))
    
    fig_block, ax_block = plt.subplots(figsize=(12, 3))
    
    window = 50
    blocked_e = results['daily_data']['blocked_e']
    blocked_f = results['daily_data']['blocked_f']
    arrivals_e = results['daily_data']['arrivals_e']
    arrivals_f = results['daily_data']['arrivals_f']
    
    rolling_block_e = np.convolve(blocked_e, np.ones(window), 'valid') / np.maximum(np.convolve(arrivals_e, np.ones(window), 'valid'), 1) * 100
    rolling_block_f = np.convolve(blocked_f, np.ones(window), 'valid') / np.maximum(np.convolve(arrivals_f, np.ones(window), 'valid'), 1) * 100
    
    ax_block.plot(range(window-1, len(blocked_e)), rolling_block_e, color=COLOR_ECONSULT, label='eConsult', alpha=0.9, linewidth=1.5)
    ax_block.plot(range(window-1, len(blocked_f)), rolling_block_f, color=COLOR_FTF, label='FTF', alpha=0.9, linewidth=1.5)
    ax_block.axvline(x=warmup_day, color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Long-term analysis starts')
    ax_block.set_xlabel('Day')
    ax_block.set_ylabel('Blocking Rate (%)')
    ax_block.set_title(f'Rolling {window}-Day Blocking Rate (% of arrivals turned away)')
    ax_block.legend(fontsize=9)
    ax_block.set_xlim(0, sim_horizon_result)
    ax_block.set_ylim(bottom=0)
    ax_block.grid(True, alpha=0.3)
    
    st.pyplot(fig_block)
    plt.close()
    
    # =========================================================================
    # CAPACITY SENSITIVITY ANALYSIS
    # =========================================================================
    
    st.markdown("---")
    st.subheader("🔍 Capacity Sensitivity Analysis")
    st.caption(
        "Explore how performance responds to capacity scaling. "
        "Pick what to vary, set a range/step in %, and run the sweep. "
        "All performance metrics are computed once; you can switch which to plot afterwards "
        "without re-simulating."
    )
    
    SENS_METRIC_OPTIONS = {
        "Average FTF wait time": ("avg_wait_f", "days", 1),
        "Average eConsult wait time": ("avg_wait_e", "days", 1),
        "Weighted average wait time": ("weighted_avg_wait", "days", 1),
        "Avg FTF queue length": ("avg_queue_f", "patients", 1),
        "Avg eConsult queue length": ("avg_queue_e", "patients", 1),
        "FTF blocking rate": ("block_rate_f", "%", 2),
        "eConsult blocking rate": ("block_rate_e", "%", 2),
        "Days to reach FTF target": ("avg_target_day_f", "days", 0),
        "Days to reach eConsult target": ("avg_target_day_e", "days", 0),
        "FTF utilization (theoretical)": ("rho_f_theoretical", "", 3),
        "eConsult utilization (theoretical)": ("rho_e_theoretical", "", 3),
    }
    
    # ===== Input: radio for what-to-vary + shared range/step =====
    st.markdown("**1. Pick what to vary, range, and step**")
    
    # Improved radio labels — explain each option in-line
    sens_target_label = st.radio(
        "Vary capacity of",
        options=[
            "eConsult only — vary eConsult, FTF stays fixed",
            "FTF only — vary FTF, eConsult stays fixed",
            "Both proportional — vary both at the same rate, keeping ratio fixed",
        ],
        index=1,  # default to FTF since that's usually the bottleneck
        key="sens_target_label",
        help=(
            "**eConsult only**: Tests pure eConsult capacity impact. FTF held constant.\n\n"
            "**FTF only**: Tests pure FTF capacity impact. eConsult held constant.\n\n"
            "**Both proportional**: Scales both at the same %. The ratio between "
            "eConsult and FTF hours stays fixed throughout the sweep — useful for "
            "modeling overall practice expansion/contraction."
        )
    )
    if "eConsult only" in sens_target_label:
        sens_target = "econsult"
    elif "FTF only" in sens_target_label:
        sens_target = "ftf"
    else:
        sens_target = "both"
    
    sens_range_cols = st.columns([1, 1, 1, 1])
    with sens_range_cols[0]:
        sens_pct_min = st.number_input(
            "Range min (%)",
            min_value=-90.0,
            max_value=0.0,
            value=-50.0,
            step=10.0,
            key="sens_pct_min",
            help="Lower bound of % change. -50 = 50% less capacity than current."
        )
    with sens_range_cols[1]:
        sens_pct_max = st.number_input(
            "Range max (%)",
            min_value=0.0,
            max_value=300.0,
            value=100.0,
            step=10.0,
            key="sens_pct_max",
            help="Upper bound of % change. +100 = 2× current capacity."
        )
    with sens_range_cols[2]:
        sens_pct_step = st.number_input(
            "Step (%)",
            min_value=5.0,
            max_value=100.0,
            value=10.0,
            step=5.0,
            key="sens_pct_step",
            help="Granularity. Smaller = more points = slower."
        )
    with sens_range_cols[3]:
        sens_reps = st.number_input(
            "Replications",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="sens_reps",
            help="Reps per point. Lower = faster but noisier. Default 20."
        )
    
    # ===== Preview caption: what will the sweep actually do? =====
    # Compute endpoints based on current values
    _mult_min = 1.0 + sens_pct_min / 100.0
    _mult_max = 1.0 + sens_pct_max / 100.0
    _hrs_e_now = hrs_econsult
    _hrs_f_now = hrs_ftf
    
    if sens_target == "econsult":
        _preview = (
            f"📋 **Preview:** Sweep will test eConsult: "
            f"**{_hrs_e_now * _mult_min:.2f}** → **{_hrs_e_now * _mult_max:.2f}** hrs/day "
            f"(current: {_hrs_e_now:.2f}). "
            f"FTF stays at **{_hrs_f_now:.2f}** hrs/day total."
        )
    elif sens_target == "ftf":
        _preview = (
            f"📋 **Preview:** Sweep will test FTF total: "
            f"**{_hrs_f_now * _mult_min:.2f}** → **{_hrs_f_now * _mult_max:.2f}** hrs/day "
            f"(current: {_hrs_f_now:.2f}). "
            f"eConsult stays at **{_hrs_e_now:.2f}** hrs/day. "
            f"Each cluster's mean hours scaled proportionally; SD and absence rates unchanged."
        )
    else:  # both proportional
        # Compute a clean ratio representation
        if _hrs_e_now > 0 and _hrs_f_now > 0:
            from math import gcd
            # Use integer ratio if reasonable, otherwise show as decimal
            _e_int = int(round(_hrs_e_now * 10))
            _f_int = int(round(_hrs_f_now * 10))
            _g = gcd(_e_int, _f_int) if _e_int > 0 and _f_int > 0 else 1
            ratio_str = f"{_e_int // _g}:{_f_int // _g}"
        else:
            ratio_str = f"{_hrs_e_now:.1f}:{_hrs_f_now:.1f}"
        _preview = (
            f"📋 **Preview:** Sweep will test eConsult: "
            f"**{_hrs_e_now * _mult_min:.2f}** → **{_hrs_e_now * _mult_max:.2f}** hrs/day "
            f"(current: {_hrs_e_now:.2f}) AND FTF total: "
            f"**{_hrs_f_now * _mult_min:.2f}** → **{_hrs_f_now * _mult_max:.2f}** hrs/day "
            f"(current: {_hrs_f_now:.2f}). "
            f"Ratio **{ratio_str}** preserved throughout. "
            f"FTF clusters' mean hours scaled proportionally; SD and absence rates unchanged."
        )
    
    st.info(_preview)
    
    # Compute number of points and time estimate (calibrated)
    n_pct_points = int(np.floor((sens_pct_max - sens_pct_min) / sens_pct_step)) + 1
    n_pct_points = max(1, n_pct_points)
    _sec_per_rep_day = st.session_state.get('sec_per_rep_day', 0.00003)
    est_sens_seconds = max(2, int(n_pct_points * sens_reps * sim_horizon * _sec_per_rep_day * 1.2))
    if est_sens_seconds < 60:
        est_sens_str = f"~{est_sens_seconds} sec"
    else:
        est_sens_str = f"~{est_sens_seconds // 60} min {est_sens_seconds % 60} sec"
    
    _calib_note = (" (calibrated from your last simulation)"
                    if st.session_state.get('sim_results') is not None
                    else " (rough estimate)")
    st.caption(
        f"⏱ {n_pct_points} points × {sens_reps} reps × {sim_horizon} days — "
        f"estimated **{est_sens_str}**{_calib_note}"
    )
    
    run_sens_button = st.button(
        "🔍 Run sensitivity analysis",
        type="primary",
        use_container_width=True,
        key="run_sens_button"
    )
    
    if run_sens_button:
        sens_progress = st.progress(0, text="Initializing sensitivity...")
        pct_changes = np.arange(sens_pct_min, sens_pct_max + sens_pct_step / 2, sens_pct_step)
        
        base_params = dict(st.session_state.sim_params)
        base_hrs_e = base_params['hrs_econsult']
        base_hrs_f = base_params['hrs_ftf']
        base_clusters = base_params.get('ftf_clusters', None)
        
        rows = []
        import time as _time
        _t_sens_start = _time.time()
        
        for i, pct in enumerate(pct_changes):
            sens_progress.progress(
                (i + 1) / len(pct_changes),
                text=f"Evaluating {pct:+.0f}% ({i+1} of {len(pct_changes)})..."
            )
            multiplier = 1.0 + pct / 100.0
            params = dict(base_params)
            params['num_replications'] = sens_reps
            
            # Apply scaling depending on what's being varied
            if sens_target == "econsult" or sens_target == "both":
                new_hrs_e = base_hrs_e * multiplier
                params['hrs_econsult'] = new_hrs_e
                params['c_e'] = int(np.floor(new_hrs_e * params['econsult_rate']))
            if sens_target == "ftf" or sens_target == "both":
                new_hrs_f = base_hrs_f * multiplier
                params['hrs_ftf'] = new_hrs_f
                # Scale each cluster's hours proportionally
                if base_clusters:
                    scaled_clusters = [
                        {'hours': c['hours'] * multiplier,
                         'sd': c.get('sd', 0.0),
                         'p_absent': c.get('p_absent', 0.0)}
                        for c in base_clusters
                    ]
                    params['ftf_clusters'] = scaled_clusters
                # c_f based on effective hours
                if base_clusters:
                    effective_new = sum(c['hours'] * multiplier * (1.0 - c.get('p_absent', 0.0))
                                          for c in base_clusters)
                else:
                    effective_new = new_hrs_f
                params['c_f'] = int(np.floor(effective_new * params['ftf_rate']))
            
            sim_result = run_simulation(params)
            
            row = {
                'pct_change': round(pct, 2),
                'multiplier': round(multiplier, 4),
                'hrs_econsult': params['hrs_econsult'],
                'hrs_ftf': params['hrs_ftf'],
                'c_e': params['c_e'],
                'c_f': params['c_f'],
            }
            # Add all metric values
            for label, (key, _, _) in SENS_METRIC_OPTIONS.items():
                row[key] = sim_result.get(key)
            rows.append(row)
        
        _t_sens_elapsed = _time.time() - _t_sens_start
        if len(pct_changes) > 0 and sim_horizon > 0:
            _measured = _t_sens_elapsed / (len(pct_changes) * sens_reps * sim_horizon)
            _old = st.session_state.get('sec_per_rep_day', 0.00003)
            st.session_state.sec_per_rep_day = 0.5 * _old + 0.5 * _measured
        
        st.session_state.cap_sens_results = {
            'rows': rows,
            'target': sens_target,
            'target_label': sens_target_label,
            'pct_min': sens_pct_min,
            'pct_max': sens_pct_max,
            'pct_step': sens_pct_step,
            'reps': sens_reps,
            'metric_options': SENS_METRIC_OPTIONS,
            'elapsed_seconds': _t_sens_elapsed,
        }
        sens_progress.empty()
        st.success(f"Sensitivity analysis complete! ({_t_sens_elapsed:.1f} sec)")
    
    # ===== Output: multi-metric selection + plots + table =====
    if st.session_state.get('cap_sens_results') is not None:
        cap = st.session_state.cap_sens_results
        rows = cap['rows']
        
        if not rows:
            st.warning("No results to display.")
        else:
            st.markdown("---")
            st.markdown("**2. Pick metrics to plot** (you can select multiple)")
            
            metric_opts = cap['metric_options']
            selected_metric_labels = st.multiselect(
                "Metrics to plot",
                options=list(metric_opts.keys()),
                default=["Average FTF wait time"],
                key="sens_selected_metrics",
                label_visibility="collapsed",
            )
            
            if selected_metric_labels:
                # One subplot per metric, stacked vertically
                n_metrics = len(selected_metric_labels)
                fig_height = max(3, 2.5 * n_metrics)
                fig_sens, axes = plt.subplots(n_metrics, 1, figsize=(10, fig_height),
                                                sharex=True)
                if n_metrics == 1:
                    axes = [axes]
                
                # Build x-axis values from ACTUAL CAPACITY at each sweep point
                # (not % change). For "both proportional", show "eConsult / FTF".
                target = cap['target']
                if target == 'econsult':
                    x_values = [r['hrs_econsult'] for r in rows]
                    x_axis_label = "Average eConsult working hours per day (practice total)"
                    current_x = next((r['hrs_econsult'] for r in rows if r['pct_change'] == 0), None)
                    tick_labels = None  # default numeric ticks
                elif target == 'ftf':
                    x_values = [r['hrs_ftf'] for r in rows]
                    x_axis_label = "Average FTF working hours per day (practice total)"
                    current_x = next((r['hrs_ftf'] for r in rows if r['pct_change'] == 0), None)
                    tick_labels = None
                else:  # both proportional
                    # x_values: use FTF hours as the numeric position (consistent ordering)
                    # but show "eConsult / FTF" combined labels at each point
                    x_values = [r['hrs_ftf'] for r in rows]
                    x_axis_label = "Average eConsult / FTF working hours per day (practice total)"
                    current_x = next((r['hrs_ftf'] for r in rows if r['pct_change'] == 0), None)
                    tick_labels = [f"{r['hrs_econsult']:.1f} / {r['hrs_ftf']:.1f}" for r in rows]
                
                for ax, label in zip(axes, selected_metric_labels):
                    key, units, decimals = metric_opts[label]
                    values = [r.get(key) for r in rows]
                    valid_pairs = [(x, v) for x, v in zip(x_values, values) if v is not None]
                    if valid_pairs:
                        xs, ys = zip(*valid_pairs)
                        ax.plot(xs, ys, marker='o', linewidth=2, markersize=5, color='#1f77b4')
                    if current_x is not None:
                        ax.axvline(current_x, color='gray', linestyle='--', alpha=0.5, label='Current')
                    ax.set_ylabel(f"{label}\n({units})" if units else label)
                    ax.grid(alpha=0.3)
                    ax.legend(loc='best', fontsize=9)
                    if units in ('days', 'patients', '%') or 'rho' in key:
                        ax.set_ylim(bottom=0)
                
                # X-axis: use combined "eConsult / FTF" labels for proportional case
                if tick_labels is not None:
                    axes[-1].set_xticks(x_values)
                    axes[-1].set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=8)
                
                axes[-1].set_xlabel(x_axis_label)
                # No suptitle (removed per UI feedback)
                plt.tight_layout()
                st.pyplot(fig_sens)
                plt.close()
                
                # Caption explaining non-monotonic behavior potential
                st.caption(
                    "ℹ️ Queue length can behave non-monotonically when capacity is very low. "
                    "The buffer (max queue size) is proportional to capacity, so at very low "
                    "capacity the queue saturates against the buffer rather than growing freely. "
                    "To verify, add **FTF blocking rate** to the selected metrics — if it is high "
                    "(say above 50%) in that region, the queue is artificially capped by patient rejection."
                )
            else:
                st.info("Select one or more metrics above to display plots.")
            
            # ===== Full data table =====
            st.markdown("---")
            st.markdown("**3. Full data table**")
            df_sens = pd.DataFrame(rows)
            st.dataframe(df_sens, hide_index=True, use_container_width=True)
            
            # CSV download
            st.download_button(
                "📥 Download full data as CSV",
                data=df_sens.to_csv(index=False).encode('utf-8'),
                file_name=f"capacity_sensitivity_{cap['target']}.csv",
                mime="text/csv",
            )
            
            st.caption(
                "ℹ️ Switching which metrics are plotted does NOT re-run the simulation — "
                "all metrics were computed during the sweep. Only changing the input (target/range/step/reps) "
                "and clicking the button above triggers a new run."
            )

else:
    st.info("👆 Configure your parameters in the flow chart above and click **Run Simulation** in the sidebar.")

# =============================================================================
# FOOTER
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("*Developed for URMC Dermatology*")
st.sidebar.markdown("*University of Rochester*")
