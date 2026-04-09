"""
Teledermatology Simulation App

This Streamlit app simulates patient flow in a teledermatology clinic
to help optimize capacity allocation between eConsult and FTF appointments.

Author: Lexie Sun
Date: March 2025
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Import custom modules
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

# Custom CSS: widen sidebar, constrain popover width
st.markdown("""
<style>
    section[data-testid="stSidebar"] { width: 350px !important; }
    [data-testid="stPopover"] > div { max-width: 280px; }
</style>
""", unsafe_allow_html=True)

# Compact matplotlib defaults
plt.rcParams.update({'font.size': 9})

# =============================================================================
# CONSTANTS
# =============================================================================

TOTAL_WORK_UNITS = 44
ECONSULT_WORK_FACTOR = 2.8
GAMMA = 0.4    # Conversion rate (fixed)

# Default referral rates
DEFAULT_LAMBDA_E = 28
DEFAULT_LAMBDA_D = 34

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'sim_results' not in st.session_state:
    st.session_state.sim_results = None
if 'sim_params' not in st.session_state:
    st.session_state.sim_params = None

# =============================================================================
# ABOUT DIALOG (modal overlay)
# =============================================================================

@st.dialog("About the App")
def show_about():
    st.markdown("""
    This tool simulates patient flow in a **teledermatology clinic** to help 
    optimize capacity allocation between **eConsult** and **Face-to-Face (FTF)** 
    appointments.
    """)
    
    st.markdown("---")
    st.markdown("**Model Setup & Assumptions:**")
    st.markdown("""
    - Based on referral patterns at **URMC Dermatology**
    - A **single dermatologist** serves the clinic each day
    - Two appointment types: **eConsult** (electronic) and **FTF** (in-person)
    - eConsult appointments are **2.8× faster** than FTF appointments  
      (1 FTF uses the same time as 2.8 eConsults)
    - Some eConsult patients require FTF follow-up (**conversion**)
    - Patients who cannot be served within the buffer period are **blocked** (turned away)
    - Service follows **First-In-First-Out (FIFO)** order
    """)
    
    st.markdown("---")
    st.markdown("**Fixed Parameters:**")
    st.markdown(f"""
    - **γ = {GAMMA}** — {int(GAMMA*100)}% of eConsult patients require FTF follow-up
    - **Total work units = {TOTAL_WORK_UNITS}** — Daily physician capacity
    - **Work unit ratio = {ECONSULT_WORK_FACTOR}** — One FTF uses {ECONSULT_WORK_FACTOR}× more time than one eConsult
    """)
    
    st.markdown("**User-Adjustable Parameters:**")
    st.markdown("""
    - **λ_e** — Average daily eConsult referrals
    - **λ_d** — Average daily direct FTF referrals
    - **Buffer size** — Maximum days a patient can wait
    - **c_f** — Daily FTF appointment slots (c_e auto-calculated)
    """)
    
    st.markdown("---")
    st.markdown("*Developed for URMC Dermatology, University of Rochester*")

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🏥 Teledermatology")
st.sidebar.markdown("---")

# ----- About the App (modal dialog) -----
if st.sidebar.button("ℹ️ About the App", use_container_width=True):
    show_about()

st.sidebar.markdown("---")

# ----- Problem Settings -----
st.sidebar.subheader("Problem Settings")

buffer_days = st.sidebar.slider(
    "Buffer Size (days)",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="Maximum number of days patients can wait in queue before being blocked (turned away)."
)

lambda_e = st.sidebar.slider(
    "λ_e (eConsult referrals/day)",
    min_value=0,
    max_value=125,
    value=DEFAULT_LAMBDA_E,
    step=1,
    help="Average number of eConsult referrals arriving per day."
)

lambda_d = st.sidebar.slider(
    "λ_d (Direct FTF referrals/day)",
    min_value=0,
    max_value=50,
    value=DEFAULT_LAMBDA_D,
    step=1,
    help="Average number of direct FTF referrals arriving per day (patients who skip eConsult)."
)

st.sidebar.markdown("---")

# ----- Capacity Allocation -----
st.sidebar.subheader("Capacity Allocation")

c_f = st.sidebar.slider(
    "FTF Capacity (c_f)",
    min_value=0,
    max_value=44,
    value=34,
    step=1,
    help="Number of FTF appointment slots per day. The clinic has 44 total work units. FTF uses 1 unit each; eConsult uses 1/2.8 units each. Remaining capacity goes to eConsult."
)

# Auto-calculate c_e
c_e = int((TOTAL_WORK_UNITS - c_f) * ECONSULT_WORK_FACTOR)

st.sidebar.markdown(f"**eConsult Capacity (c_e):** {c_e} slots/day")

with st.sidebar.popover("How is c_e calculated?"):
    st.markdown(f"**c_e = (44 - c_f) × 2.8**")
    st.markdown(f"= ({TOTAL_WORK_UNITS} - {c_f}) × {ECONSULT_WORK_FACTOR} = **{c_e}**")
    st.markdown("FTF appointments use 1 work unit each. eConsult appointments use 1/2.8 work units each (they're 2.8× faster).")

# Max queue lengths
max_q_e = buffer_days * c_e if c_e > 0 else 0
max_q_f = buffer_days * c_f if c_f > 0 else 0
st.sidebar.markdown(f"**Max Queue Lengths** (Buffer × Capacity):")
st.sidebar.markdown(f"- eConsult: {buffer_days} × {c_e} = **{max_q_e}**")
st.sidebar.markdown(f"- FTF: {buffer_days} × {c_f} = **{max_q_f}**")

# ----- Utilization Warnings -----
# Calculate theoretical utilization
rho_e = lambda_e / c_e if c_e > 0 else float('inf')
rho_f = (GAMMA * lambda_e + lambda_d) / c_f if c_f > 0 else float('inf')

# Display warnings if ρ ≥ 1
if rho_e >= 1 or rho_f >= 1:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚠️ Capacity Warnings:**")
    
    if rho_e >= 1:
        st.sidebar.error(
            f"**eConsult overloaded!**  \n"
            f"ρ_e = λ_e / c_e = {lambda_e} / {c_e} = **{rho_e:.2f}**  \n"
            f"High blocking expected. Increase c_e or reduce λ_e."
        )
    
    if rho_f >= 1:
        st.sidebar.error(
            f"**FTF overloaded!**  \n"
            f"ρ_f = (γ×λ_e + λ_d) / c_f = ({GAMMA}×{lambda_e} + {lambda_d}) / {c_f} = **{rho_f:.2f}**  \n"
            f"High blocking expected. Increase c_f or reduce λ_d."
        )

st.sidebar.markdown("---")

# ----- Simulation Settings -----
st.sidebar.subheader("Simulation Settings")

sim_horizon = st.sidebar.number_input(
    "Simulation Horizon (days)",
    min_value=1000,
    max_value=50000,
    value=8000,
    step=1000,
    help="Total number of days to simulate per replication."
)

num_replications = st.sidebar.number_input(
    "Number of Replications",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
    help="Number of independent simulation runs to average over."
)

st.sidebar.markdown("---")

# Run Simulation Button
run_sim_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("*Developed for URMC Dermatology*")
st.sidebar.markdown("*University of Rochester*")

# =============================================================================
# MAIN AREA
# =============================================================================

st.title("📊 Teledermatology Simulation")
st.markdown("Simulate the queueing system and analyze steady-state performance.")

if run_sim_button:
    # Progress bar in main area
    progress_bar = st.progress(0, text="Initializing simulation...")

    # Store parameters
    sim_params = {
        'buffer_days': buffer_days,
        'lambda_e': lambda_e,
        'lambda_d': lambda_d,
        'c_e': c_e,
        'c_f': c_f,
        'gamma': GAMMA,
        'sim_horizon': sim_horizon,
        'num_replications': num_replications,
        'warmup_fraction': 0.5  # First 50% is warmup
    }

    def update_progress(current, total, phase):
        if phase == "simulation":
            frac = current / total if total > 0 else 0
            progress_bar.progress(frac, text=f"Running replication {current + 1} of {total}...")
        elif phase == "analysis":
            progress_bar.progress(1.0, text="Analyzing results...")

    # Run simulation
    start_time = time.time()
    results = run_simulation(sim_params, progress_callback=update_progress)
    elapsed = time.time() - start_time

    progress_bar.empty()
    st.success(f"Simulation completed in {elapsed:.1f} seconds!")

    # Store results
    st.session_state.sim_results = results
    st.session_state.sim_params = sim_params

# Display results if available
if st.session_state.sim_results is not None:
    results = st.session_state.sim_results
    params = st.session_state.sim_params

    # ----- Part 1: Warmup Analysis -----
    st.header("🔥 Warmup Analysis")

    warmup_day = int(params['sim_horizon'] * params['warmup_fraction'])

    # Create figure with 2 subplots (smaller size)
    fig_warmup, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    days = results['daily_data']['day']
    q_e_data = results['daily_data']['q_e']
    q_f_data = results['daily_data']['q_f']
    total_queue = results['daily_data']['total_queue']

    # --- Top plot: Total Queue Length ---
    axes[0].plot(days, total_queue, linewidth=0.8, color='purple', label='$q_e + q_f$')
    axes[0].axvline(x=warmup_day, color='red', linestyle='--', linewidth=2, label=f'Steady State (Day {warmup_day})')
    axes[0].axvspan(0, warmup_day, alpha=0.15, color='gray', label='Warmup Period')
    axes[0].set_ylabel('Total Queue (patients)')
    axes[0].set_title('Total Queue Length Over Time')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # --- Bottom plot: Separate Queues ---
    axes[1].plot(days, q_e_data, linewidth=0.8, color='green', label='$q_e$ (eConsult)')
    axes[1].plot(days, q_f_data, linewidth=0.8, color='blue', label='$q_f$ (FTF)')
    axes[1].axvline(x=warmup_day, color='red', linestyle='--', linewidth=2)
    axes[1].axvspan(0, warmup_day, alpha=0.15, color='gray')
    # Add buffer limits
    axes[1].axhline(y=results['buffer_size_e'], color='green', linestyle=':', alpha=0.5, label=f'Buffer e={results["buffer_size_e"]}')
    axes[1].axhline(y=results['buffer_size_f'], color='blue', linestyle=':', alpha=0.5, label=f'Buffer f={results["buffer_size_f"]}')
    axes[1].set_xlabel('Simulation Day')
    axes[1].set_ylabel('Queue Length (patients)')
    axes[1].set_title('Queue Length by Type')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_warmup)
    plt.close()

    st.info(f"**Warmup Period:** Days 0 - {warmup_day} | **Steady State:** Days {warmup_day} - {params['sim_horizon']}")

    # Steady-state queue lengths
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.metric("Avg q_e (eConsult)", f"{results['avg_queue_e']:.1f} patients",
                   help="Average eConsult queue length during the steady-state period.")
    with col_q2:
        st.metric("Avg q_f (FTF)", f"{results['avg_queue_f']:.1f} patients",
                   help="Average FTF queue length during the steady-state period.")

    st.markdown("---")

    # ----- Part 2: Simulation Results -----
    st.header("📊 Steady-State Results")

    # --- Section 1: Wait Time Metrics ---
    st.subheader("⏱️ Wait Times")

    col_w1, col_w2 = st.columns(2)

    with col_w1:
        st.metric("Avg Wait (eConsult Resolved)", f"{results['avg_wait_e']:.2f} days",
                   help="Days patients wait in eConsult queue before being served and resolved (no FTF needed).")
        st.metric("Avg Wait (FTF)", f"{results['avg_wait_f']:.2f} days",
                   help="Days patients wait in FTF queue (includes both direct FTF and converted patients).")

    with col_w2:
        st.metric("Avg Wait (Converted Total)", f"{results['avg_wait_converted']:.2f} days",
                   help="Total days for converted patients: eConsult wait + FTF wait.")
        st.metric("Weighted Avg Wait", f"{results['weighted_avg_wait']:.2f} days",
                   help="Average wait across all patient types, weighted by their proportion.")

    # Wait Time Bar Chart (smaller)
    fig_wait, ax_wait = plt.subplots(figsize=(6, 2.5))
    wait_labels = ['eConsult\nResolved', 'FTF\n(All)', 'Converted\n(Total)']
    wait_values = [results['avg_wait_e'], results['avg_wait_f'], results['avg_wait_converted']]
    colors = ['green', 'blue', 'orange']
    bars = ax_wait.bar(wait_labels, wait_values, color=colors, alpha=0.7, edgecolor='black')
    ax_wait.axhline(y=results['weighted_avg_wait'], color='red', linestyle='--', linewidth=2, label=f"Weighted Avg: {results['weighted_avg_wait']:.2f}")
    ax_wait.set_ylabel('Wait Time (days)')
    ax_wait.set_title('Average Wait Time by Patient Type')
    ax_wait.legend()
    ax_wait.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars, wait_values):
        ax_wait.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_wait)
    plt.close()

    st.markdown("---")

    # --- Section 2: Patient Flow ---
    st.subheader("👥 Patient Flow")

    n_total = results['n_resolved'] + results['n_converted'] + results['n_direct']
    if n_total > 0:
        pct_resolved = results['n_resolved'] / n_total * 100
        pct_converted = results['n_converted'] / n_total * 100
        pct_direct = results['n_direct'] / n_total * 100
    else:
        pct_resolved = pct_converted = pct_direct = 0

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.metric("Resolved via eConsult", f"{pct_resolved:.1f}%",
                   help="Patients who only needed eConsult (no FTF follow-up).")
        st.metric("Converted (eConsult -> FTF)", f"{pct_converted:.1f}%",
                   help="Patients who needed a FTF appointment after their eConsult.")
        st.metric("Direct FTF", f"{pct_direct:.1f}%",
                   help="Patients who went directly to FTF without an eConsult.")

    with col_p2:
        # Pie Chart (smaller)
        fig_pie, ax_pie = plt.subplots(figsize=(3, 3))
        sizes = [results['n_resolved'], results['n_converted'], results['n_direct']]
        labels = ['Resolved\n(eConsult)', 'Converted\n(eConsult->FTF)', 'Direct FTF']
        colors = ['lightgreen', 'orange', 'lightblue']
        explode = (0.02, 0.02, 0.02)

        if sum(sizes) > 0:
            ax_pie.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                      shadow=False, startangle=90)
            ax_pie.set_title('Patient Flow Distribution')
        else:
            ax_pie.text(0.5, 0.5, 'No data', ha='center', va='center')

        st.pyplot(fig_pie)
        plt.close()

    st.markdown("---")

    # --- Section 3: Blocking Rate Over Time ---
    st.subheader("🚫 Blocking Rate Over Time")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.metric("Steady-State Blocking (eConsult)", f"{results['block_rate_e']:.2f}%",
                   help="Percentage of eConsult referrals blocked because the queue was full.")
    with col_b2:
        st.metric("Steady-State Blocking (FTF)", f"{results['block_rate_f']:.2f}%",
                   help="Percentage of FTF referrals blocked because the queue was full.")

    # Blocking Rate Time Series Plot (smaller)
    fig_block, ax_block = plt.subplots(figsize=(8, 2.5))

    days = results['daily_data']['day']
    arrivals_e = results['daily_data']['arrivals_e']
    arrivals_f = results['daily_data']['arrivals_f']
    blocked_e = results['daily_data']['blocked_e']
    blocked_f = results['daily_data']['blocked_f']

    # Compute blocking rate (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        block_rate_e_daily = np.where(arrivals_e > 0, blocked_e / arrivals_e * 100, 0)
        block_rate_f_daily = np.where(arrivals_f > 0, blocked_f / arrivals_f * 100, 0)

    # Use rolling average for smoother plot
    window = 50
    if len(block_rate_e_daily) > window:
        block_rate_e_smooth = np.convolve(block_rate_e_daily, np.ones(window)/window, mode='valid')
        block_rate_f_smooth = np.convolve(block_rate_f_daily, np.ones(window)/window, mode='valid')
        days_smooth = days[window-1:]
    else:
        block_rate_e_smooth = block_rate_e_daily
        block_rate_f_smooth = block_rate_f_daily
        days_smooth = days

    ax_block.plot(days_smooth, block_rate_e_smooth, linewidth=1, color='green', label='eConsult Blocking %', alpha=0.8)
    ax_block.plot(days_smooth, block_rate_f_smooth, linewidth=1, color='blue', label='FTF Blocking %', alpha=0.8)
    ax_block.axvline(x=warmup_day, color='red', linestyle='--', linewidth=2, label=f'Steady State (Day {warmup_day})')
    ax_block.axvspan(0, warmup_day, alpha=0.15, color='gray')
    ax_block.set_xlabel('Simulation Day')
    ax_block.set_ylabel('Blocking Rate (%)')
    ax_block.set_title(f'Blocking Rate Over Time (Rolling {window}-day Average)')
    ax_block.legend(loc='upper right')
    ax_block.grid(True, alpha=0.3)
    ax_block.set_ylim(bottom=0)

    plt.tight_layout()
    st.pyplot(fig_block)
    plt.close()

    st.markdown("---")

    # --- Section 4: Utilization ---
    st.subheader("⚡ Utilization & Throughput")

    col_u1, col_u2 = st.columns(2)

    with col_u1:
        st.markdown("**Theoretical Utilization** (referral rate / capacity)")
        st.metric("ρ_e (eConsult)", f"{results['rho_e_theoretical']:.1%}",
                   help=f"ρ_e = λ_e / c_e = {params['lambda_e']} / {params['c_e']} = {results['rho_e_theoretical']:.2f}. If ρ ≥ 1, queue is overloaded.")
        st.metric("ρ_f (FTF)", f"{results['rho_f_theoretical']:.1%}",
                   help=f"ρ_f = (γ × λ_e + λ_d) / c_f = ({GAMMA} × {params['lambda_e']} + {params['lambda_d']}) / {params['c_f']} = {results['rho_f_theoretical']:.2f}")
        
        st.markdown("**Empirical Utilization** (from simulation)")
        st.metric("ρ_e (eConsult)", f"{results['rho_e_empirical']:.1%}",
                   help="Actual busy time / total time. Capped at 100% due to blocking.")
        st.metric("ρ_f (FTF)", f"{results['rho_f_empirical']:.1%}",
                   help="Actual busy time / total time. Capped at 100% due to blocking.")

    with col_u2:
        st.metric("Throughput", f"{results['throughput']:.1f} patients/day",
                   help="Total patients served per day (eConsult + FTF), averaged across replications.")

        # Utilization Bar Chart - show both theoretical and empirical (smaller)
        fig_util, ax_util = plt.subplots(figsize=(4, 2.5))
        x = np.arange(2)
        width = 0.35
        
        theoretical = [results['rho_e_theoretical'], results['rho_f_theoretical']]
        empirical = [results['rho_e_empirical'], results['rho_f_empirical']]
        
        bars1 = ax_util.bar(x - width/2, theoretical, width, label='Theoretical', color='lightcoral', edgecolor='black')
        bars2 = ax_util.bar(x + width/2, empirical, width, label='Empirical', color='lightgreen', edgecolor='black')
        
        ax_util.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='ρ = 1 (critical)')
        ax_util.set_ylabel('Utilization (ρ)')
        ax_util.set_title('Theoretical vs Empirical Utilization')
        ax_util.set_xticks(x)
        ax_util.set_xticklabels(['ρ_e (eConsult)', 'ρ_f (FTF)'])
        ax_util.legend(loc='upper right', fontsize=8)
        ax_util.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            ax_util.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax_util.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig_util)
        plt.close()

    st.markdown("---")

else:
    # No results yet - show placeholder
    st.info("Set parameters in the sidebar and click **Run Simulation** to see results.")

    st.markdown("### What you'll see:")
    st.markdown("""
    1. **Warmup Plot**: Queue length over time with steady-state detection
    2. **Performance Metrics**: Wait times, queue lengths, blocking rates
    3. **Utilization**: Theoretical vs empirical utilization comparison
    """)
