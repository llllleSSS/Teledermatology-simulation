"""
Simulation Module for Teledermatology Queueing System

Slimmed-down version of the full simulation for Streamlit app.
Based on the original simulation code by Lexie Sun.

Key features:
- Single scenario simulation (not batch)
- Parameters from user input
- Returns data for plotting and metrics display
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time

# =============================================================================
# CONSTANTS
# =============================================================================

TOTAL_WORK_UNITS = 44
ECONSULT_WORK_FACTOR = 2.8  # 1 FTF slot = 2.8 eConsult slots in work units

# =============================================================================
# SCENARIO CONFIGURATION
# =============================================================================

@dataclass
class SimConfig:
    """Configuration for a single simulation scenario."""
    lambda_e: float          # eConsult arrival rate
    lambda_d: float          # Direct FTF arrival rate
    c_e: int                 # eConsult capacity
    c_f: int                 # FTF capacity
    gamma: float             # Conversion rate (default 0.4)
    buffer_days: int         # Buffer size in days
    sim_horizon: int         # Total simulation days
    num_replications: int    # Number of replications
    warmup_fraction: float   # Fraction of simulation for warmup (default 0.5)
    
    @property
    def buffer_size_e(self) -> int:
        """Max eConsult queue size."""
        return self.buffer_days * self.c_e if self.c_e > 0 else 0
    
    @property
    def buffer_size_f(self) -> int:
        """Max FTF queue size."""
        return self.buffer_days * self.c_f
    
    @property
    def warmup_days(self) -> int:
        """Number of warmup days."""
        return int(self.sim_horizon * self.warmup_fraction)
    
    @property
    def analysis_days(self) -> int:
        """Number of analysis (steady-state) days."""
        return self.sim_horizon - self.warmup_days
    
    @property
    def rho_e(self) -> float:
        """eConsult utilization."""
        return self.lambda_e / self.c_e if self.c_e > 0 else 0
    
    @property
    def rho_f(self) -> float:
        """FTF utilization."""
        ftf_load = self.gamma * self.lambda_e + self.lambda_d
        return ftf_load / self.c_f if self.c_f > 0 else float('inf')


# =============================================================================
# SINGLE REPLICATION SIMULATION
# =============================================================================

def run_single_replication(seed: int, config: SimConfig) -> Dict:
    """
    Run a single simulation replication.
    
    Parameters:
    -----------
    seed : int
        Random seed for this replication
    config : SimConfig
        Simulation configuration
    
    Returns:
    --------
    dict with daily data and patient-level wait times
    """
    np.random.seed(seed)
    
    # Initialize queues (store arrival day for wait time calculation)
    econsult_queue = deque()  # Each element: (arrival_day,)
    ftf_queue = deque()       # Each element: (arrival_day, source, original_arrival_day)
    
    # Daily tracking arrays
    daily_q_e = []  # End-of-day eConsult queue length
    daily_q_f = []  # End-of-day FTF queue length
    daily_blocked_e = []
    daily_blocked_f = []
    daily_served_e = []
    daily_served_f = []
    daily_idle_e = []
    daily_idle_f = []
    daily_arrivals_e = []           # Actual eConsult arrivals each day
    daily_arrivals_f_direct = []    # Actual direct FTF arrivals each day
    daily_arrivals_f_converted = [] # Conversions each day (arrivals to FTF from eConsult)
    
    # Patient-level wait time tracking
    wait_times_resolved = []    # eConsult patients resolved (not converted)
    wait_times_converted = []   # eConsult patients who converted (total system time)
    wait_times_direct_ftf = []  # Direct FTF patients
    
    # Daily wait times for detailed analysis
    daily_econsult_wait_times = []
    daily_ftf_wait_times = []
    
    for day in range(config.sim_horizon):
        
        # ----- STEP 1: GENERATE ARRIVALS -----
        n_arrivals_e = np.random.poisson(config.lambda_e) if config.lambda_e > 0 else 0
        n_arrivals_d = np.random.poisson(config.lambda_d) if config.lambda_d > 0 else 0
        
        # ----- STEP 2: ADD ARRIVALS TO QUEUES (with blocking) -----
        blocked_e_today = 0
        blocked_f_direct_today = 0
        
        # eConsult arrivals
        for _ in range(n_arrivals_e):
            if config.c_e > 0 and len(econsult_queue) < config.buffer_size_e:
                econsult_queue.append((day,))
            else:
                blocked_e_today += 1
        
        # Direct FTF arrivals
        for _ in range(n_arrivals_d):
            if len(ftf_queue) < config.buffer_size_f:
                ftf_queue.append((day, 'direct', None))
            else:
                blocked_f_direct_today += 1
        
        # ----- STEP 3: SERVE E-CONSULT QUEUE -----
        # Converted patients are held in a pending list until AFTER FTF service
        served_e_today = 0
        blocked_f_converted_today = 0
        conversions_today = 0
        econsult_wait_times_today = []
        pending_conversions = []  # Hold converted patients until after FTF service
        
        if config.c_e > 0:
            while econsult_queue and served_e_today < config.c_e:
                patient = econsult_queue.popleft()
                arrival_day = patient[0]
                wait_time = day - arrival_day
                econsult_wait_times_today.append(wait_time)
                served_e_today += 1
                
                # Conversion decision
                if np.random.random() < config.gamma:
                    # Patient needs FTF follow-up - add to pending list (NOT queue yet)
                    pending_conversions.append((day, 'converted', arrival_day))
                else:
                    # Patient resolved via eConsult
                    if day >= config.warmup_days:
                        wait_times_resolved.append(wait_time)
        
        idle_e_today = config.c_e - served_e_today if config.c_e > 0 else 0
        
        # ----- STEP 4: SERVE FTF QUEUE -----
        # Note: Converted patients from today are NOT in the queue yet
        served_f_today = 0
        ftf_wait_times_today = []
        
        while ftf_queue and served_f_today < config.c_f:
            patient = ftf_queue.popleft()
            ftf_arrival_day = patient[0]
            source = patient[1]
            original_arrival_day = patient[2]
            
            ftf_wait = day - ftf_arrival_day
            ftf_wait_times_today.append(ftf_wait)
            served_f_today += 1
            
            # Track patient-level wait times (only in analysis period)
            if day >= config.warmup_days:
                if source == 'direct':
                    wait_times_direct_ftf.append(ftf_wait)
                else:  # converted
                    total_system_time = day - original_arrival_day
                    wait_times_converted.append(total_system_time)
        
        idle_f_today = config.c_f - served_f_today
        
        # ----- STEP 5: ADD CONVERTED PATIENTS TO FTF QUEUE -----
        # Now add pending conversions to FTF queue (AFTER FTF service)
        # They can only be served starting NEXT day
        conversion_attempts_today = len(pending_conversions)  # Track ATTEMPTS
        for conv_patient in pending_conversions:
            if len(ftf_queue) < config.buffer_size_f:
                ftf_queue.append(conv_patient)
                conversions_today += 1
            else:
                blocked_f_converted_today += 1
        
        # ----- STEP 6: RECORD END-OF-DAY DATA -----
        daily_q_e.append(len(econsult_queue))
        daily_q_f.append(len(ftf_queue))
        daily_blocked_e.append(blocked_e_today)
        daily_blocked_f.append(blocked_f_direct_today + blocked_f_converted_today)
        daily_served_e.append(served_e_today)
        daily_served_f.append(served_f_today)
        daily_idle_e.append(idle_e_today)
        daily_idle_f.append(idle_f_today)
        daily_econsult_wait_times.append(econsult_wait_times_today)
        daily_ftf_wait_times.append(ftf_wait_times_today)
        # Track actual arrivals for blocking rate calculation
        daily_arrivals_e.append(n_arrivals_e)
        daily_arrivals_f_direct.append(n_arrivals_d)
        daily_arrivals_f_converted.append(conversion_attempts_today)  # Track ATTEMPTS, not just successful
    
    return {
        'daily_q_e': np.array(daily_q_e),
        'daily_q_f': np.array(daily_q_f),
        'daily_blocked_e': np.array(daily_blocked_e),
        'daily_blocked_f': np.array(daily_blocked_f),
        'daily_served_e': np.array(daily_served_e),
        'daily_served_f': np.array(daily_served_f),
        'daily_idle_e': np.array(daily_idle_e),
        'daily_idle_f': np.array(daily_idle_f),
        'daily_econsult_wait_times': daily_econsult_wait_times,
        'daily_ftf_wait_times': daily_ftf_wait_times,
        'daily_arrivals_e': np.array(daily_arrivals_e),
        'daily_arrivals_f_direct': np.array(daily_arrivals_f_direct),
        'daily_arrivals_f_converted': np.array(daily_arrivals_f_converted),
        'wait_times_resolved': wait_times_resolved,
        'wait_times_converted': wait_times_converted,
        'wait_times_direct_ftf': wait_times_direct_ftf,
    }


# =============================================================================
# MULTI-REPLICATION SIMULATION
# =============================================================================

def run_simulation(params: Dict, progress_callback=None) -> Dict:
    """
    Run simulation with multiple replications.
    
    Parameters:
    -----------
    params : dict
        - buffer_days: int (5, 10, or 15)
        - lambda_e: float (eConsult arrival rate)
        - lambda_d: float (direct FTF arrival rate)
        - c_e: int (eConsult capacity)
        - c_f: int (FTF capacity)
        - gamma: float (conversion rate, default 0.4)
        - sim_horizon: int (total simulation days)
        - num_replications: int (number of replications)
        - warmup_fraction: float (fraction of sim for warmup, default 0.5)
    progress_callback : callable, optional
        Function(current, total, phase) called during simulation.
        phase is "simulation" or "analysis".
    
    Returns:
    --------
    dict with:
        - daily_data: for plotting
        - steady-state metrics
    """
    
    # Create config
    config = SimConfig(
        lambda_e=params['lambda_e'],
        lambda_d=params['lambda_d'],
        c_e=params['c_e'],
        c_f=params['c_f'],
        gamma=params.get('gamma', 0.4),
        buffer_days=params['buffer_days'],
        sim_horizon=params['sim_horizon'],
        num_replications=params['num_replications'],
        warmup_fraction=params.get('warmup_fraction', 0.5)
    )
    
    # Storage for aggregating across replications
    all_daily_q_e = []
    all_daily_q_f = []
    
    # Daily blocking/arrivals for time series plot
    all_daily_blocked_e = []
    all_daily_blocked_f = []
    all_daily_arrivals_e = []
    all_daily_arrivals_f = []
    
    # Steady-state metrics accumulators
    all_wait_resolved = []
    all_wait_converted = []
    all_wait_direct = []
    all_ftf_wait_times = []  # All FTF wait times (direct + converted)
    
    total_blocked_e = 0
    total_blocked_f = 0
    total_arrivals_e_actual = 0    # Actual arrivals (from Poisson draws)
    total_arrivals_f_actual = 0    # Actual arrivals to FTF queue
    total_served_e = 0
    total_served_f = 0
    total_idle_e = 0
    total_idle_f = 0
    
    # Run replications
    for seed in range(1, config.num_replications + 1):
        if progress_callback:
            progress_callback(seed - 1, config.num_replications, "simulation")
        rep_result = run_single_replication(seed, config)
        
        # Store daily data
        all_daily_q_e.append(rep_result['daily_q_e'])
        all_daily_q_f.append(rep_result['daily_q_f'])
        all_daily_blocked_e.append(rep_result['daily_blocked_e'])
        all_daily_blocked_f.append(rep_result['daily_blocked_f'])
        all_daily_arrivals_e.append(rep_result['daily_arrivals_e'])
        # FTF arrivals = direct + converted
        daily_arrivals_f_total = rep_result['daily_arrivals_f_direct'] + rep_result['daily_arrivals_f_converted']
        all_daily_arrivals_f.append(daily_arrivals_f_total)
        
        # Accumulate wait times
        all_wait_resolved.extend(rep_result['wait_times_resolved'])
        all_wait_converted.extend(rep_result['wait_times_converted'])
        all_wait_direct.extend(rep_result['wait_times_direct_ftf'])
        
        # Collect FTF wait times from analysis period (ALL patients served at FTF)
        for day in range(config.warmup_days, config.sim_horizon):
            all_ftf_wait_times.extend(rep_result['daily_ftf_wait_times'][day])
        
        # Accumulate steady-state metrics (analysis period only)
        warmup = config.warmup_days
        total_blocked_e += np.sum(rep_result['daily_blocked_e'][warmup:])
        total_blocked_f += np.sum(rep_result['daily_blocked_f'][warmup:])
        total_served_e += np.sum(rep_result['daily_served_e'][warmup:])
        total_served_f += np.sum(rep_result['daily_served_f'][warmup:])
        total_idle_e += np.sum(rep_result['daily_idle_e'][warmup:])
        total_idle_f += np.sum(rep_result['daily_idle_f'][warmup:])
        
        # Count ACTUAL arrivals in analysis period (not expected)
        total_arrivals_e_actual += np.sum(rep_result['daily_arrivals_e'][warmup:])
        total_arrivals_f_actual += np.sum(rep_result['daily_arrivals_f_direct'][warmup:]) + \
                                   np.sum(rep_result['daily_arrivals_f_converted'][warmup:])
    
    if progress_callback:
        progress_callback(config.num_replications, config.num_replications, "analysis")

    # Compute average daily queues (for plotting)
    mean_daily_q_e = np.mean(all_daily_q_e, axis=0)
    mean_daily_q_f = np.mean(all_daily_q_f, axis=0)
    std_daily_q_e = np.std(all_daily_q_e, axis=0)
    std_daily_q_f = np.std(all_daily_q_f, axis=0)
    
    # Compute average daily blocking (for plotting)
    mean_daily_blocked_e = np.mean(all_daily_blocked_e, axis=0)
    mean_daily_blocked_f = np.mean(all_daily_blocked_f, axis=0)
    mean_daily_arrivals_e = np.mean(all_daily_arrivals_e, axis=0)
    mean_daily_arrivals_f = np.mean(all_daily_arrivals_f, axis=0)
    
    # Compute steady-state queue lengths (analysis period)
    warmup = config.warmup_days
    avg_queue_e = np.mean([q[warmup:].mean() for q in all_daily_q_e])
    avg_queue_f = np.mean([q[warmup:].mean() for q in all_daily_q_f])
    
    # Compute wait times
    avg_wait_e = np.mean(all_wait_resolved) if all_wait_resolved else 0
    avg_wait_f = np.mean(all_ftf_wait_times) if all_ftf_wait_times else 0  # All FTF patients
    avg_wait_converted = np.mean(all_wait_converted) if all_wait_converted else 0
    avg_wait_direct = np.mean(all_wait_direct) if all_wait_direct else 0
    
    # Weighted average wait time (per patient type)
    n_resolved = len(all_wait_resolved)
    n_converted = len(all_wait_converted)
    n_direct = len(all_wait_direct)
    n_total = n_resolved + n_converted + n_direct
    
    if n_total > 0:
        weighted_avg_wait = (
            (n_resolved / n_total) * avg_wait_e +
            (n_converted / n_total) * avg_wait_converted +
            (n_direct / n_total) * avg_wait_direct
        )
    else:
        weighted_avg_wait = 0
    
    # Blocking rates (using ACTUAL arrivals)
    block_rate_e = (total_blocked_e / total_arrivals_e_actual * 100) if total_arrivals_e_actual > 0 else 0
    block_rate_f = (total_blocked_f / total_arrivals_f_actual * 100) if total_arrivals_f_actual > 0 else 0
    
    # Throughput
    analysis_days_total = config.analysis_days * config.num_replications
    throughput = (total_served_e + total_served_f) / analysis_days_total
    
    # Utilization - Theoretical (arrival rate / capacity)
    rho_e_theoretical = config.rho_e
    rho_f_theoretical = config.rho_f
    
    # Utilization - Empirical (actual time busy / total time)
    total_capacity_e = config.c_e * analysis_days_total if config.c_e > 0 else 1
    total_capacity_f = config.c_f * analysis_days_total if config.c_f > 0 else 1
    
    rho_e_empirical = total_served_e / total_capacity_e
    rho_f_empirical = total_served_f / total_capacity_f
    
    # Prepare results
    results = {
        # Daily data for plotting
        'daily_data': {
            'day': np.arange(config.sim_horizon),
            'q_e': mean_daily_q_e,
            'q_f': mean_daily_q_f,
            'q_e_std': std_daily_q_e,
            'q_f_std': std_daily_q_f,
            'total_queue': mean_daily_q_e + mean_daily_q_f,
            'blocked_e': mean_daily_blocked_e,
            'blocked_f': mean_daily_blocked_f,
            'arrivals_e': mean_daily_arrivals_e,
            'arrivals_f': mean_daily_arrivals_f,
        },
        
        # Steady-state metrics
        'avg_wait_e': avg_wait_e,
        'avg_wait_f': avg_wait_f,
        'avg_wait_converted': avg_wait_converted,
        'avg_wait_direct': avg_wait_direct,
        'weighted_avg_wait': weighted_avg_wait,
        'avg_queue_e': avg_queue_e,
        'avg_queue_f': avg_queue_f,
        'block_rate_e': block_rate_e,
        'block_rate_f': block_rate_f,
        'rho_e_theoretical': rho_e_theoretical,
        'rho_f_theoretical': rho_f_theoretical,
        'rho_e_empirical': rho_e_empirical,
        'rho_f_empirical': rho_f_empirical,
        # Backward compatible keys (use empirical as default display)
        'rho_e': rho_e_empirical,
        'rho_f': rho_f_empirical,
        'throughput': throughput,
        
        # Config info
        'warmup_day': config.warmup_days,
        'buffer_size_e': config.buffer_size_e,
        'buffer_size_f': config.buffer_size_f,
        
        # Patient counts
        'n_resolved': n_resolved,
        'n_converted': n_converted,
        'n_direct': n_direct,
    }
    
    return results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_theoretical_metrics(params: Dict) -> Dict:
    """
    Compute theoretical steady-state metrics.
    
    Parameters:
    -----------
    params : dict with lambda_e, lambda_d, c_e, c_f, gamma
    
    Returns:
    --------
    dict with theoretical utilization and stability info
    """
    lambda_e = params['lambda_e']
    lambda_d = params['lambda_d']
    c_e = params['c_e']
    c_f = params['c_f']
    gamma = params.get('gamma', 0.4)
    
    # Utilization
    rho_e = lambda_e / c_e if c_e > 0 else float('inf')
    rho_f = (gamma * lambda_e + lambda_d) / c_f if c_f > 0 else float('inf')
    
    # Stability check
    stable_e = rho_e < 1
    stable_f = rho_f < 1
    
    # Total work check
    work_e = lambda_e / ECONSULT_WORK_FACTOR  # Work units for eConsult
    work_f = gamma * lambda_e + lambda_d       # Work units for FTF
    total_work = work_e + work_f
    feasible = total_work <= TOTAL_WORK_UNITS
    
    return {
        'rho_e': rho_e,
        'rho_f': rho_f,
        'stable_e': stable_e,
        'stable_f': stable_f,
        'total_work': total_work,
        'feasible': feasible
    }


def detect_steady_state(queue_data: np.ndarray, method: str = 'fixed', **kwargs) -> int:
    """
    Detect when the simulation reaches steady state.
    
    Parameters:
    -----------
    queue_data : np.ndarray
        Queue length over time
    method : str
        Detection method: 'fixed' (default), 'moving_avg', 'variance'
    
    Returns:
    --------
    int : Day when steady state begins
    """
    if method == 'fixed':
        fraction = kwargs.get('fraction', 0.5)
        return int(len(queue_data) * fraction)
    else:
        return len(queue_data) // 2
