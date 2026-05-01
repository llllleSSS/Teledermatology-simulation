"""
Simulation Module for Teledermatology Queueing System

Configurable version allowing user-defined service rates, capacity,
conversion rate, and initial queue backlogs.

Backlog vs Buffer:
- Backlog: initial queue size (can exceed buffer)
- Buffer: max queue size for NEW arrivals
- When queue > buffer, new arrivals are blocked until queue <= buffer

Author: Lexie Sun
Date: April 2025
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time


@dataclass
class SimConfig:
    """Configuration for a single simulation scenario."""
    lambda_e: float          # eConsult arrival rate (per day)
    lambda_d: float          # Direct FTF arrival rate (per day)
    c_e: int                 # eConsult capacity (patients/day)
    c_f: int                 # FTF capacity (patients/day)
    gamma: float             # Conversion rate (configurable)
    buffer_days_e: int       # Buffer size in days for eConsult queue
    buffer_days_f: int       # Buffer size in days for FTF queue
    sim_horizon: int         # Total simulation days
    num_replications: int    # Number of replications
    warmup_fraction: float   # Fraction of simulation for warmup (default 0.5)
    initial_q_e: int = 0     # Initial eConsult backlog (can exceed buffer)
    initial_q_f: int = 0     # Initial FTF backlog (can exceed buffer)
    target_q_e: Optional[int] = None
    target_q_f: Optional[int] = None
    
    @property
    def buffer_size_e(self) -> int:
        """Max eConsult queue size for new arrivals."""
        return self.buffer_days_e * self.c_e if self.c_e > 0 else 0
    
    @property
    def buffer_size_f(self) -> int:
        """Max FTF queue size for new arrivals."""
        return self.buffer_days_f * self.c_f if self.c_f > 0 else 0
    
    @property
    def warmup_days(self) -> int:
        return int(self.sim_horizon * self.warmup_fraction)
    
    @property
    def analysis_days(self) -> int:
        return self.sim_horizon - self.warmup_days
    
    @property
    def rho_e(self) -> float:
        return self.lambda_e / self.c_e if self.c_e > 0 else 0
    
    @property
    def rho_f(self) -> float:
        ftf_load = self.gamma * self.lambda_e + self.lambda_d
        return ftf_load / self.c_f if self.c_f > 0 else float('inf')


def run_single_replication(seed: int, config: SimConfig) -> Dict:
    """Run a single simulation replication."""
    np.random.seed(seed)
    
    # Initialize queues with backlog (can exceed buffer)
    econsult_queue = deque()
    ftf_queue = deque()
    
    for _ in range(config.initial_q_e):
        econsult_queue.append((-1,))
    
    for _ in range(config.initial_q_f):
        ftf_queue.append((-1, 'direct', None))
    
    # Daily tracking
    daily_q_e = []
    daily_q_f = []
    daily_blocked_e = []
    daily_blocked_f = []
    daily_served_e = []
    daily_served_f = []
    daily_idle_e = []
    daily_idle_f = []
    daily_arrivals_e = []
    daily_arrivals_f_direct = []
    daily_arrivals_f_converted = []
    
    wait_times_resolved = []
    wait_times_converted = []
    wait_times_direct_ftf = []
    
    daily_econsult_wait_times = []
    daily_ftf_wait_times = []
    
    target_day_e = None
    target_day_f = None
    
    for day in range(config.sim_horizon):
        
        # STEP 1: Generate arrivals
        n_arrivals_e = np.random.poisson(config.lambda_e) if config.lambda_e > 0 else 0
        n_arrivals_d = np.random.poisson(config.lambda_d) if config.lambda_d > 0 else 0
        
        # STEP 2: Add arrivals to queues (buffer-based blocking)
        blocked_e_today = 0
        blocked_f_direct_today = 0
        
        for _ in range(n_arrivals_e):
            if config.c_e > 0 and len(econsult_queue) < config.buffer_size_e:
                econsult_queue.append((day,))
            else:
                blocked_e_today += 1
        
        for _ in range(n_arrivals_d):
            if len(ftf_queue) < config.buffer_size_f:
                ftf_queue.append((day, 'direct', None))
            else:
                blocked_f_direct_today += 1
        
        # STEP 3: Serve eConsult queue
        served_e_today = 0
        econsult_wait_times_today = []
        pending_conversions = []
        
        if config.c_e > 0:
            while econsult_queue and served_e_today < config.c_e:
                patient = econsult_queue.popleft()
                arrival_day = patient[0]
                wait_time = day - arrival_day
                econsult_wait_times_today.append(wait_time)
                served_e_today += 1
                
                if np.random.random() < config.gamma:
                    pending_conversions.append((day, 'converted', arrival_day))
                else:
                    if day >= config.warmup_days:
                        wait_times_resolved.append(wait_time)
        
        idle_e_today = config.c_e - served_e_today if config.c_e > 0 else 0
        
        # STEP 4: Serve FTF queue
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
            
            if day >= config.warmup_days:
                if source == 'direct':
                    wait_times_direct_ftf.append(ftf_wait)
                else:
                    total_system_time = day - original_arrival_day
                    wait_times_converted.append(total_system_time)
        
        idle_f_today = config.c_f - served_f_today
        
        # STEP 5: Add pending conversions AFTER FTF service
        conversion_attempts_today = len(pending_conversions)
        blocked_f_converted_today = 0
        
        for conv_patient in pending_conversions:
            if len(ftf_queue) < config.buffer_size_f:
                ftf_queue.append(conv_patient)
            else:
                blocked_f_converted_today += 1
        
        # STEP 6: Record end-of-day
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
        daily_arrivals_e.append(n_arrivals_e)
        daily_arrivals_f_direct.append(n_arrivals_d)
        daily_arrivals_f_converted.append(conversion_attempts_today)
        
        # Check targets
        if target_day_e is None and config.target_q_e is not None:
            if len(econsult_queue) <= config.target_q_e:
                target_day_e = day
        
        if target_day_f is None and config.target_q_f is not None:
            if len(ftf_queue) <= config.target_q_f:
                target_day_f = day
    
    return {
        'daily_q_e': np.array(daily_q_e),
        'daily_q_f': np.array(daily_q_f),
        'daily_blocked_e': np.array(daily_blocked_e),
        'daily_blocked_f': np.array(daily_blocked_f),
        'daily_served_e': np.array(daily_served_e),
        'daily_served_f': np.array(daily_served_f),
        'daily_idle_e': np.array(daily_idle_e),
        'daily_idle_f': np.array(daily_idle_f),
        'daily_arrivals_e': np.array(daily_arrivals_e),
        'daily_arrivals_f_direct': np.array(daily_arrivals_f_direct),
        'daily_arrivals_f_converted': np.array(daily_arrivals_f_converted),
        'daily_econsult_wait_times': daily_econsult_wait_times,
        'daily_ftf_wait_times': daily_ftf_wait_times,
        'wait_times_resolved': wait_times_resolved,
        'wait_times_converted': wait_times_converted,
        'wait_times_direct_ftf': wait_times_direct_ftf,
        'target_day_e': target_day_e,
        'target_day_f': target_day_f,
    }


def run_simulation(params: Dict, progress_callback=None) -> Dict:
    """Run multiple replications and aggregate results, including per-replication SDs."""
    # Support legacy single buffer_days param as fallback for backward compatibility
    if 'buffer_days_e' in params and 'buffer_days_f' in params:
        buf_e = params['buffer_days_e']
        buf_f = params['buffer_days_f']
    else:
        buf_e = buf_f = params['buffer_days']

    config = SimConfig(
        lambda_e=params['lambda_e'],
        lambda_d=params['lambda_d'],
        c_e=params['c_e'],
        c_f=params['c_f'],
        gamma=params.get('gamma', 0.4),
        buffer_days_e=buf_e,
        buffer_days_f=buf_f,
        sim_horizon=params['sim_horizon'],
        num_replications=params['num_replications'],
        warmup_fraction=params.get('warmup_fraction', 0.5),
        initial_q_e=params.get('initial_q_e', 0),
        initial_q_f=params.get('initial_q_f', 0),
        target_q_e=params.get('target_q_e', None),
        target_q_f=params.get('target_q_f', None),
    )
    
    # Daily-level tracking (across reps)
    all_daily_q_e = []
    all_daily_q_f = []
    all_daily_blocked_e = []
    all_daily_blocked_f = []
    all_daily_arrivals_e = []
    all_daily_arrivals_f = []
    
    # Patient-level tracking (across reps, pooled)
    all_wait_resolved = []
    all_wait_converted = []
    all_wait_direct = []
    all_ftf_wait_times = []
    all_target_days_e = []
    all_target_days_f = []
    
    # Per-replication summary metrics (one number per rep -> SD across reps)
    per_rep_avg_wait_e = []          # mean wait of resolved eConsult patients in this rep
    per_rep_avg_wait_f = []          # mean wait of all FTF patients in this rep
    per_rep_avg_wait_direct = []     # mean wait of direct FTF patients in this rep
    per_rep_avg_wait_converted = []  # mean total system time of converted patients in this rep
    per_rep_weighted_wait = []       # weighted-avg wait across all categories in this rep
    per_rep_avg_queue_e = []         # mean queue length (post-warmup) in this rep
    per_rep_avg_queue_f = []         # same for FTF
    per_rep_block_rate_e = []        # blocking rate (%) in this rep
    per_rep_block_rate_f = []        # same for FTF
    per_rep_rho_e_emp = []           # empirical eConsult utilization in this rep
    per_rep_rho_f_emp = []           # empirical FTF utilization in this rep
    
    # Aggregate counters (still useful for global stats)
    total_blocked_e = 0
    total_blocked_f = 0
    total_arrivals_e_actual = 0
    total_arrivals_f_actual = 0
    total_served_e = 0
    total_served_f = 0
    
    warmup = config.warmup_days
    
    for seed in range(1, config.num_replications + 1):
        if progress_callback:
            progress_callback(seed - 1, config.num_replications, "simulation")
        rep_result = run_single_replication(seed, config)
        
        # Daily-level
        all_daily_q_e.append(rep_result['daily_q_e'])
        all_daily_q_f.append(rep_result['daily_q_f'])
        all_daily_blocked_e.append(rep_result['daily_blocked_e'])
        all_daily_blocked_f.append(rep_result['daily_blocked_f'])
        all_daily_arrivals_e.append(rep_result['daily_arrivals_e'])
        daily_arrivals_f_total = rep_result['daily_arrivals_f_direct'] + rep_result['daily_arrivals_f_converted']
        all_daily_arrivals_f.append(daily_arrivals_f_total)
        
        # Target day capture
        if rep_result['target_day_e'] is not None:
            all_target_days_e.append(rep_result['target_day_e'])
        if rep_result['target_day_f'] is not None:
            all_target_days_f.append(rep_result['target_day_f'])
        
        # Patient-level wait times (already filtered to post-warmup inside run_single_replication)
        all_wait_resolved.extend(rep_result['wait_times_resolved'])
        all_wait_converted.extend(rep_result['wait_times_converted'])
        all_wait_direct.extend(rep_result['wait_times_direct_ftf'])
        
        # Collect per-rep FTF wait times for "all patients" stat
        rep_ftf_wait_all = []
        for day in range(warmup, config.sim_horizon):
            rep_ftf_wait_all.extend(rep_result['daily_ftf_wait_times'][day])
        all_ftf_wait_times.extend(rep_ftf_wait_all)
        
        # ---- Per-rep summary metrics ----
        # Wait time means within this rep (only post-warmup data)
        rep_resolved = rep_result['wait_times_resolved']
        rep_direct = rep_result['wait_times_direct_ftf']
        rep_converted = rep_result['wait_times_converted']
        
        rep_avg_wait_e = float(np.mean(rep_resolved)) if rep_resolved else 0.0
        rep_avg_wait_direct = float(np.mean(rep_direct)) if rep_direct else 0.0
        rep_avg_wait_converted = float(np.mean(rep_converted)) if rep_converted else 0.0
        rep_avg_wait_f = float(np.mean(rep_ftf_wait_all)) if rep_ftf_wait_all else 0.0
        
        # Weighted average within rep
        n_r = len(rep_resolved)
        n_c = len(rep_converted)
        n_d = len(rep_direct)
        n_t = n_r + n_c + n_d
        if n_t > 0:
            rep_weighted = (
                (n_r / n_t) * rep_avg_wait_e +
                (n_c / n_t) * rep_avg_wait_converted +
                (n_d / n_t) * rep_avg_wait_direct
            )
        else:
            rep_weighted = 0.0
        
        per_rep_avg_wait_e.append(rep_avg_wait_e)
        per_rep_avg_wait_f.append(rep_avg_wait_f)
        per_rep_avg_wait_direct.append(rep_avg_wait_direct)
        per_rep_avg_wait_converted.append(rep_avg_wait_converted)
        per_rep_weighted_wait.append(rep_weighted)
        
        # Queue length means (post-warmup)
        per_rep_avg_queue_e.append(float(rep_result['daily_q_e'][warmup:].mean()))
        per_rep_avg_queue_f.append(float(rep_result['daily_q_f'][warmup:].mean()))
        
        # Blocking rates within this rep (%)
        rep_blocked_e = int(np.sum(rep_result['daily_blocked_e'][warmup:]))
        rep_blocked_f = int(np.sum(rep_result['daily_blocked_f'][warmup:]))
        rep_arrivals_e = int(np.sum(rep_result['daily_arrivals_e'][warmup:]))
        rep_arrivals_f_direct = int(np.sum(rep_result['daily_arrivals_f_direct'][warmup:]))
        rep_arrivals_f_converted = int(np.sum(rep_result['daily_arrivals_f_converted'][warmup:]))
        rep_arrivals_f_total = rep_arrivals_f_direct + rep_arrivals_f_converted
        
        rep_block_rate_e = (rep_blocked_e / rep_arrivals_e * 100) if rep_arrivals_e > 0 else 0.0
        rep_block_rate_f = (rep_blocked_f / rep_arrivals_f_total * 100) if rep_arrivals_f_total > 0 else 0.0
        per_rep_block_rate_e.append(rep_block_rate_e)
        per_rep_block_rate_f.append(rep_block_rate_f)
        
        # Empirical utilization within rep
        rep_served_e = int(np.sum(rep_result['daily_served_e'][warmup:]))
        rep_served_f = int(np.sum(rep_result['daily_served_f'][warmup:]))
        rep_capacity_e = config.c_e * config.analysis_days if config.c_e > 0 else 0
        rep_capacity_f = config.c_f * config.analysis_days if config.c_f > 0 else 0
        rep_rho_e = rep_served_e / rep_capacity_e if rep_capacity_e > 0 else 0.0
        rep_rho_f = rep_served_f / rep_capacity_f if rep_capacity_f > 0 else 0.0
        per_rep_rho_e_emp.append(rep_rho_e)
        per_rep_rho_f_emp.append(rep_rho_f)
        
        # Update global counters
        total_blocked_e += rep_blocked_e
        total_blocked_f += rep_blocked_f
        total_served_e += rep_served_e
        total_served_f += rep_served_f
        total_arrivals_e_actual += rep_arrivals_e
        total_arrivals_f_actual += rep_arrivals_f_total
    
    if progress_callback:
        progress_callback(config.num_replications, config.num_replications, "analysis")

    # ===== Daily means and SDs across replications =====
    mean_daily_q_e = np.mean(all_daily_q_e, axis=0)
    mean_daily_q_f = np.mean(all_daily_q_f, axis=0)
    std_daily_q_e = np.std(all_daily_q_e, axis=0)
    std_daily_q_f = np.std(all_daily_q_f, axis=0)
    
    # Total queue across reps (mean and SD)
    all_daily_total = [q_e + q_f for q_e, q_f in zip(all_daily_q_e, all_daily_q_f)]
    mean_daily_total = np.mean(all_daily_total, axis=0)
    std_daily_total = np.std(all_daily_total, axis=0)
    
    mean_daily_blocked_e = np.mean(all_daily_blocked_e, axis=0)
    mean_daily_blocked_f = np.mean(all_daily_blocked_f, axis=0)
    mean_daily_arrivals_e = np.mean(all_daily_arrivals_e, axis=0)
    mean_daily_arrivals_f = np.mean(all_daily_arrivals_f, axis=0)
    
    # ===== Summary statistics: mean and SD across reps =====
    def mean_sd(arr):
        """Return (mean, sd) — sd undefined if fewer than 2 reps."""
        if len(arr) == 0:
            return 0.0, None
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1)) if len(arr) > 1 else None
        return m, s
    
    avg_wait_e, sd_wait_e = mean_sd(per_rep_avg_wait_e)
    avg_wait_f, sd_wait_f = mean_sd(per_rep_avg_wait_f)
    avg_wait_direct, sd_wait_direct = mean_sd(per_rep_avg_wait_direct)
    avg_wait_converted, sd_wait_converted = mean_sd(per_rep_avg_wait_converted)
    weighted_avg_wait, sd_weighted_wait = mean_sd(per_rep_weighted_wait)
    avg_queue_e, sd_queue_e = mean_sd(per_rep_avg_queue_e)
    avg_queue_f, sd_queue_f = mean_sd(per_rep_avg_queue_f)
    block_rate_e, sd_block_rate_e = mean_sd(per_rep_block_rate_e)
    block_rate_f, sd_block_rate_f = mean_sd(per_rep_block_rate_f)
    rho_e_empirical, sd_rho_e_emp = mean_sd(per_rep_rho_e_emp)
    rho_f_empirical, sd_rho_f_emp = mean_sd(per_rep_rho_f_emp)
    
    # Target day stats — mean and SD across reps that reached target
    if all_target_days_e:
        avg_target_day_e = float(np.mean(all_target_days_e))
        sd_target_day_e = float(np.std(all_target_days_e, ddof=1)) if len(all_target_days_e) > 1 else None
    else:
        avg_target_day_e = None
        sd_target_day_e = None
    
    if all_target_days_f:
        avg_target_day_f = float(np.mean(all_target_days_f))
        sd_target_day_f = float(np.std(all_target_days_f, ddof=1)) if len(all_target_days_f) > 1 else None
    else:
        avg_target_day_f = None
        sd_target_day_f = None
    
    pct_reached_target_e = len(all_target_days_e) / config.num_replications * 100
    pct_reached_target_f = len(all_target_days_f) / config.num_replications * 100
    
    # Patient counts
    n_resolved = len(all_wait_resolved)
    n_converted = len(all_wait_converted)
    n_direct = len(all_wait_direct)
    
    # Throughput
    analysis_days_total = config.analysis_days * config.num_replications
    throughput = (total_served_e + total_served_f) / analysis_days_total if analysis_days_total > 0 else 0
    
    # Theoretical utilization
    rho_e_theoretical = config.rho_e
    rho_f_theoretical = config.rho_f
    
    results = {
        'daily_data': {
            'day': np.arange(config.sim_horizon),
            'q_e': mean_daily_q_e,
            'q_f': mean_daily_q_f,
            'q_e_std': std_daily_q_e,
            'q_f_std': std_daily_q_f,
            'total_queue': mean_daily_total,
            'total_queue_std': std_daily_total,
            'blocked_e': mean_daily_blocked_e,
            'blocked_f': mean_daily_blocked_f,
            'arrivals_e': mean_daily_arrivals_e,
            'arrivals_f': mean_daily_arrivals_f,
        },
        # Wait time means + SDs
        'avg_wait_e': avg_wait_e,
        'sd_wait_e': sd_wait_e,
        'avg_wait_f': avg_wait_f,
        'sd_wait_f': sd_wait_f,
        'avg_wait_converted': avg_wait_converted,
        'sd_wait_converted': sd_wait_converted,
        'avg_wait_direct': avg_wait_direct,
        'sd_wait_direct': sd_wait_direct,
        'weighted_avg_wait': weighted_avg_wait,
        'sd_weighted_wait': sd_weighted_wait,
        # Queue length means + SDs
        'avg_queue_e': avg_queue_e,
        'sd_queue_e': sd_queue_e,
        'avg_queue_f': avg_queue_f,
        'sd_queue_f': sd_queue_f,
        # Blocking rates means + SDs
        'block_rate_e': block_rate_e,
        'sd_block_rate_e': sd_block_rate_e,
        'block_rate_f': block_rate_f,
        'sd_block_rate_f': sd_block_rate_f,
        # Utilization
        'rho_e': rho_e_empirical,
        'rho_f': rho_f_empirical,
        'rho_e_theoretical': rho_e_theoretical,
        'rho_f_theoretical': rho_f_theoretical,
        'rho_e_empirical': rho_e_empirical,
        'sd_rho_e_empirical': sd_rho_e_emp,
        'rho_f_empirical': rho_f_empirical,
        'sd_rho_f_empirical': sd_rho_f_emp,
        # Throughput / config
        'throughput': throughput,
        'warmup_day': config.warmup_days,
        'buffer_size_e': config.buffer_size_e,
        'buffer_size_f': config.buffer_size_f,
        # Patient counts
        'n_resolved': n_resolved,
        'n_converted': n_converted,
        'n_direct': n_direct,
        # Target days + SD
        'avg_target_day_e': avg_target_day_e,
        'sd_target_day_e': sd_target_day_e,
        'avg_target_day_f': avg_target_day_f,
        'sd_target_day_f': sd_target_day_f,
        'pct_reached_target_e': pct_reached_target_e,
        'pct_reached_target_f': pct_reached_target_f,
        # Pass-through
        'target_q_e': config.target_q_e,
        'target_q_f': config.target_q_f,
        'initial_q_e': config.initial_q_e,
        'initial_q_f': config.initial_q_f,
        'num_replications': config.num_replications,
    }
    
    return results


def run_sensitivity(base_params: Dict,
                    vary_param: str,
                    values: List,
                    progress_callback=None) -> Dict:
    """
    Run the simulation once for each value of `vary_param`, holding all other
    parameters constant at the values in `base_params`.
    
    Parameters
    ----------
    base_params : dict
        The full parameter dict for `run_simulation` (same shape as the main app uses).
    vary_param : str
        Key in base_params to vary (e.g., 'lambda_e', 'hrs_econsult').
        Note: 'hrs_econsult', 'hrs_ftf', 'econsult_rate', 'ftf_rate' are NOT
        direct simulation params — they are derived into 'c_e' and 'c_f'.
        For those, we recompute c_e/c_f for each value.
    values : list
        List of values to try for vary_param.
    progress_callback : callable, optional
        Called with (current, total, phase) — phase = "sensitivity".
    
    Returns
    -------
    dict with:
        - 'vary_param': name of varied parameter
        - 'values': list of values tested
        - 'rows': list of dicts (one per value), each with all key metrics
        - 'base_params': copy of input base_params
    """
    rows = []
    n_values = len(values)
    
    for i, val in enumerate(values):
        if progress_callback:
            progress_callback(i, n_values, "sensitivity")
        
        # Build a fresh params dict for this value
        params = dict(base_params)
        
        # Handle derived parameters: capacity is computed from rate * hours
        if vary_param == 'hrs_econsult':
            params['hrs_econsult'] = val
            rate = params.get('econsult_rate', 6.0)
            params['c_e'] = int(val * rate)
        elif vary_param == 'hrs_ftf':
            params['hrs_ftf'] = val
            rate = params.get('ftf_rate', 2.0)
            params['c_f'] = int(val * rate)
        elif vary_param == 'econsult_rate':
            params['econsult_rate'] = val
            hrs = params.get('hrs_econsult', 2.0)
            params['c_e'] = int(hrs * val)
        elif vary_param == 'ftf_rate':
            params['ftf_rate'] = val
            hrs = params.get('hrs_ftf', 6.0)
            params['c_f'] = int(hrs * val)
        else:
            params[vary_param] = val
        
        # Run a single simulation (no inner progress callback)
        sim_result = run_simulation(params)
        
        # Extract key metrics for the sensitivity row
        row = {
            'parameter_name': vary_param,
            'parameter_value': val,
            # Target days
            'days_to_target_e': sim_result.get('avg_target_day_e'),
            'sd_days_to_target_e': sim_result.get('sd_target_day_e'),
            'pct_reached_target_e': sim_result.get('pct_reached_target_e'),
            'days_to_target_f': sim_result.get('avg_target_day_f'),
            'sd_days_to_target_f': sim_result.get('sd_target_day_f'),
            'pct_reached_target_f': sim_result.get('pct_reached_target_f'),
            # Wait times
            'avg_wait_e': sim_result.get('avg_wait_e'),
            'sd_wait_e': sim_result.get('sd_wait_e'),
            'avg_wait_f': sim_result.get('avg_wait_f'),
            'sd_wait_f': sim_result.get('sd_wait_f'),
            'avg_wait_direct': sim_result.get('avg_wait_direct'),
            'sd_wait_direct': sim_result.get('sd_wait_direct'),
            'avg_wait_converted': sim_result.get('avg_wait_converted'),
            'sd_wait_converted': sim_result.get('sd_wait_converted'),
            'weighted_avg_wait': sim_result.get('weighted_avg_wait'),
            'sd_weighted_wait': sim_result.get('sd_weighted_wait'),
            # Queue lengths (steady-state)
            'avg_queue_e': sim_result.get('avg_queue_e'),
            'sd_queue_e': sim_result.get('sd_queue_e'),
            'avg_queue_f': sim_result.get('avg_queue_f'),
            'sd_queue_f': sim_result.get('sd_queue_f'),
            # Blocking rates
            'block_rate_e': sim_result.get('block_rate_e'),
            'sd_block_rate_e': sim_result.get('sd_block_rate_e'),
            'block_rate_f': sim_result.get('block_rate_f'),
            'sd_block_rate_f': sim_result.get('sd_block_rate_f'),
            # Utilization
            'rho_e_theoretical': sim_result.get('rho_e_theoretical'),
            'rho_f_theoretical': sim_result.get('rho_f_theoretical'),
            'rho_e_empirical': sim_result.get('rho_e_empirical'),
            'sd_rho_e_empirical': sim_result.get('sd_rho_e_empirical'),
            'rho_f_empirical': sim_result.get('rho_f_empirical'),
            'sd_rho_f_empirical': sim_result.get('sd_rho_f_empirical'),
            # Capacity used (for diagnosis)
            'c_e': params['c_e'],
            'c_f': params['c_f'],
        }
        rows.append(row)
    
    if progress_callback:
        progress_callback(n_values, n_values, "sensitivity_done")
    
    return {
        'vary_param': vary_param,
        'values': values,
        'rows': rows,
        'base_params': dict(base_params),
    }


def compute_theoretical_metrics(params: Dict) -> Dict:
    """Compute theoretical steady-state metrics."""
    lambda_e = params['lambda_e']
    lambda_d = params['lambda_d']
    c_e = params['c_e']
    c_f = params['c_f']
    gamma = params.get('gamma', 0.4)
    
    rho_e = lambda_e / c_e if c_e > 0 else float('inf')
    rho_f = (gamma * lambda_e + lambda_d) / c_f if c_f > 0 else float('inf')
    
    return {
        'rho_e': rho_e,
        'rho_f': rho_f,
        'stable_e': rho_e < 1,
        'stable_f': rho_f < 1,
    }


def detect_steady_state(queue_data: np.ndarray, method: str = 'fixed', **kwargs) -> int:
    """Detect when the simulation reaches steady state."""
    if method == 'fixed':
        fraction = kwargs.get('fraction', 0.5)
        return int(len(queue_data) * fraction)
    else:
        return len(queue_data) // 2
