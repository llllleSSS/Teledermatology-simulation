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

from simulation import run_simulation, detect_steady_state

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

DEFAULT_ECONSULT_RATE = 6.0
DEFAULT_FTF_RATE = 2.0

DEFAULT_HRS_ECONSULT = 2.0
DEFAULT_HRS_FTF = 6.0

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

# =============================================================================
# SIDEBAR - Minimal: About + Simulation controls
# =============================================================================

st.sidebar.title("🏥 Teledermatology")
st.sidebar.markdown("---")

if st.sidebar.button("ℹ️ About the App", use_container_width=True):
    show_about()

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Simulation Settings")

sim_horizon = st.sidebar.number_input(
    "Simulation horizon (days)",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    help="Total number of days to simulate. First half is warmup; steady-state metrics are computed from the second half."
)

num_replications = st.sidebar.number_input(
    "Number of replications",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
    help="Number of simulation runs to average. More replications give smoother estimates but take longer."
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
        econsult_rate = st.number_input(
            "Cases per hour (practice total)",
            min_value=0.5,
            max_value=50.0,
            value=DEFAULT_ECONSULT_RATE,
            step=0.5,
            key="econsult_rate",
            help=(
                "Total eConsult cases the practice completes per hour of active work, on average — "
                "aggregated across all dermatologists working in parallel. "
                "Example: if 2 dermatologists each handle 3 cases/hour at the same time, enter 6. "
                "Daily capacity = this rate × total hours per day (below)."
            )
        )
        hrs_econsult = st.number_input(
            "Total hours per day (practice-wide)",
            min_value=0.0,
            max_value=200.0,
            value=DEFAULT_HRS_ECONSULT,
            step=0.5,
            key="hrs_econsult",
            help=(
                "Total hours per day the practice actively works on eConsult, on average. "
                "For parallel work, count clock hours, not summed person-hours — "
                "2 dermatologists working 3 hours in parallel = 3 hours. "
                "Can exceed 24 for unusual cross-location or round-the-clock setups."
            )
        )

# Now we know c_e — compute it
c_e = int(hrs_econsult * econsult_rate)

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
                f'({buffer_days_e} × {c_e})</div>',
                unsafe_allow_html=True
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
                "For each eConsult patient served, the simulation flips a weighted coin "
                "with this probability to decide whether they convert to FTF. "
                "For example, 0.4 means 40% of eConsult patients (on average) will be "
                "referred to the FTF queue after their eConsult."
            )
        )

# L-shaped arrow from left side of Conversion Rate box down into FTF Queue.
# The SVG is pulled up with negative margin so its drawing area overlaps the
# Conversion Rate box vertically. The horizontal line aligns with the middle
# of the box, visually originating from the box's left edge.
st.markdown(
    """
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
        ftf_rate = st.number_input(
            "Cases per hour (practice total)",
            min_value=0.5,
            max_value=50.0,
            value=DEFAULT_FTF_RATE,
            step=0.5,
            key="ftf_rate",
            help=(
                "Total FTF visits the practice completes per hour of active work, on average — "
                "aggregated across all dermatologists working in parallel. "
                "Example: if 2 dermatologists each handle 2 visits/hour at the same time, enter 4. "
                "Daily capacity = this rate × total hours per day (below)."
            )
        )
        hrs_ftf = st.number_input(
            "Total hours per day (practice-wide)",
            min_value=0.0,
            max_value=200.0,
            value=DEFAULT_HRS_FTF,
            step=0.5,
            key="hrs_ftf",
            help=(
                "Total hours per day the practice actively works on FTF visits, on average. "
                "For parallel work, count clock hours, not summed person-hours — "
                "2 dermatologists working 4 hours in parallel = 4 hours. "
                "Can exceed 24 for unusual cross-location or round-the-clock setups."
            )
        )

# Now we know c_f
c_f = int(hrs_ftf * ftf_rate)
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
                f'({buffer_days_f} × {c_f})</div>',
                unsafe_allow_html=True
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
    st.caption(f"eConsult: **{c_e}** patients/day")
    st.caption(f"FTF: **{c_f}** patients/day")
    st.caption(f"Total: **{total_hrs}** hrs/day")

with summary_cols[1]:
    st.markdown("**Demand**")
    st.caption(f"eConsult: **{lambda_e}** patients/day")
    st.caption(f"Effective FTF: **{effective_ftf_demand:.1f}** patients/day")
    st.caption(f"Efficiency ratio: **{efficiency_ratio:.2f}×**")

with summary_cols[2]:
    st.markdown("**Max queue size**")
    st.caption(f"eConsult: **{max_q_e}** patients ({buffer_days_e} days × {c_e})")
    st.caption(f"FTF: **{max_q_f}** patients ({buffer_days_f} days × {c_f})")

# =============================================================================
# RUN SIMULATION
# =============================================================================

if run_sim_button:
    if c_e == 0 and c_f == 0:
        st.error("Please allocate some hours to at least one service type.")
        st.stop()
    
    progress_bar = st.progress(0, text="Initializing simulation...")
    
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
        'econsult_rate': econsult_rate,
        'ftf_rate': ftf_rate,
        'hrs_econsult': hrs_econsult,
        'hrs_ftf': hrs_ftf,
        'efficiency_ratio': efficiency_ratio,
        'effective_ftf_demand': effective_ftf_demand,
    }
    
    def update_progress(current, total, phase):
        if phase == "simulation":
            frac = current / total if total > 0 else 0
            progress_bar.progress(frac, text=f"Running replication {current + 1} of {total}...")
        elif phase == "analysis":
            progress_bar.progress(1.0, text="Analyzing results...")
    
    results = run_simulation(sim_params, progress_callback=update_progress)
    
    st.session_state.sim_results = results
    st.session_state.sim_params = sim_params
    
    progress_bar.empty()
    st.success("Simulation complete!")

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

if st.session_state.sim_results is not None:
    results = st.session_state.sim_results
    params = st.session_state.sim_params
    
    warmup_day = results['warmup_day']
    sim_horizon_result = params['sim_horizon']
    
    # -------------------------------------------------------------------------
    # BACKLOG REDUCTION ANALYSIS
    # -------------------------------------------------------------------------
    
    st.header("📉 Backlog Reduction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if results['avg_target_day_e'] is not None:
            st.metric(
                "Days to reach eConsult target",
                f"{results['avg_target_day_e']:.0f} days",
                delta=f"{results['pct_reached_target_e']:.0f}% of runs reached target",
                delta_color="normal"
            )
        else:
            st.metric(
                "Days to reach eConsult target",
                "Not reached",
                delta="0% of runs reached target",
                delta_color="off"
            )
    
    with col2:
        if results['avg_target_day_f'] is not None:
            st.metric(
                "Days to reach FTF target",
                f"{results['avg_target_day_f']:.0f} days",
                delta=f"{results['pct_reached_target_f']:.0f}% of runs reached target",
                delta_color="normal"
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
    
    # Plot 1: Total Queue
    fig_total, ax_total = plt.subplots(figsize=(12, 3.5))
    
    total_queue = results['daily_data']['q_e'] + results['daily_data']['q_f']
    initial_total = params['initial_q_e'] + params['initial_q_f']
    target_total = params['target_q_e'] + params['target_q_f']
    
    ax_total.plot(days, total_queue, color=COLOR_TOTAL, linewidth=1.5, label='Total Queue')
    ax_total.axhline(y=initial_total, color='red', linestyle='--', alpha=0.5, label=f'Initial ({initial_total})')
    ax_total.axhline(y=target_total, color='green', linestyle='--', alpha=0.7, label=f'Target ({target_total})')
    ax_total.axvline(x=warmup_day, color='orange', linestyle=':', linewidth=2, alpha=0.8,
                     label=f'Warmup ends (Day {warmup_day})')
    
    ax_total.axvspan(0, warmup_day, alpha=0.1, color='orange')
    
    ax_total.set_xlabel('Day')
    ax_total.set_ylabel('Total Queue Length')
    ax_total.set_title('Total Queue (eConsult + FTF) Over Time')
    ax_total.legend(loc='upper right', fontsize=8)
    ax_total.set_xlim(0, sim_horizon_result)
    ax_total.grid(True, alpha=0.3)
    
    st.pyplot(fig_total)
    plt.close()
    
    # Plot 2: Separate queues
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # eConsult queue (GREEN)
    ax1 = axes[0]
    ax1.plot(days, results['daily_data']['q_e'], color=COLOR_ECONSULT, linewidth=1.5, label='eConsult Queue')
    ax1.axhline(y=params['initial_q_e'], color='red', linestyle='--', alpha=0.5, label=f'Initial ({params["initial_q_e"]})')
    ax1.axhline(y=params['target_q_e'], color='green', linestyle='--', alpha=0.7, label=f'Target ({params["target_q_e"]})')
    ax1.axvline(x=warmup_day, color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Warmup ends')
    ax1.axvspan(0, warmup_day, alpha=0.1, color='orange')
    
    if results['avg_target_day_e'] is not None:
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
    ax1.grid(True, alpha=0.3)
    
    # FTF queue (BLUE)
    ax2 = axes[1]
    ax2.plot(days, results['daily_data']['q_f'], color=COLOR_FTF, linewidth=1.5, label='FTF Queue')
    ax2.axhline(y=params['initial_q_f'], color='red', linestyle='--', alpha=0.5, label=f'Initial ({params["initial_q_f"]})')
    ax2.axhline(y=params['target_q_f'], color='green', linestyle='--', alpha=0.7, label=f'Target ({params["target_q_f"]})')
    ax2.axvline(x=warmup_day, color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Warmup ends')
    ax2.axvspan(0, warmup_day, alpha=0.1, color='orange')
    
    if results['avg_target_day_f'] is not None:
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
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # -------------------------------------------------------------------------
    # STEADY-STATE RESULTS
    # -------------------------------------------------------------------------
    
    st.header("📊 Steady-State Results")
    st.caption(f"Metrics computed from day {warmup_day} to day {sim_horizon_result} (after warmup period)")
    
    # Wait Times
    st.subheader("⏱️ Average Wait Times")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("eConsult (resolved)", f"{results['avg_wait_e']:.1f} days")
    with col2:
        st.metric("FTF (all patients)", f"{results['avg_wait_f']:.1f} days")
    with col3:
        st.metric("Direct FTF", f"{results['avg_wait_direct']:.1f} days")
    with col4:
        st.metric("Converted (total)", f"{results['avg_wait_converted']:.1f} days")
    with col5:
        st.metric("Weighted Average", f"{results['weighted_avg_wait']:.1f} days")
    
    # Patient Flow
    st.subheader("👥 Patient Flow")
    
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
        ax_pie.set_title('Patient Outcomes', fontsize=12)
        st.pyplot(fig_pie)
        plt.close()
    
    with col2:
        st.markdown("**Patient Counts (Steady-State Period)**")
        
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
            'Empirical': [
                f"{results['rho_e_empirical']:.3f}",
                f"{results['rho_f_empirical']:.3f}",
                f"{total_empirical:.3f}"
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
        **Empirical utilization** = actual served / available capacity
        """)
    
    # Blocking Rates
    st.subheader("🚫 Blocking Rates")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("eConsult Blocking Rate", f"{results['block_rate_e']:.2f}%")
    with col2:
        st.metric("FTF Blocking Rate", f"{results['block_rate_f']:.2f}%")
    
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
    ax_block.axvline(x=warmup_day, color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Warmup ends')
    ax_block.set_xlabel('Day')
    ax_block.set_ylabel('Blocking Rate (%)')
    ax_block.set_title(f'Rolling {window}-Day Blocking Rate')
    ax_block.legend(fontsize=9)
    ax_block.set_xlim(0, sim_horizon_result)
    ax_block.grid(True, alpha=0.3)
    
    st.pyplot(fig_block)
    plt.close()

else:
    st.info("👆 Configure your parameters in the flow chart above and click **Run Simulation** in the sidebar.")

# =============================================================================
# FOOTER
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("*Developed for URMC Dermatology*")
st.sidebar.markdown("*University of Rochester*")
