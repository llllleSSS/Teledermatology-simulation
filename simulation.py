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
    """Run multiple replications and aggregate results."""
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
    
    all_daily_q_e = []
    all_daily_q_f = []
    all_daily_blocked_e = []
    all_daily_blocked_f = []
    all_daily_arrivals_e = []
    all_daily_arrivals_f = []
    all_wait_resolved = []
    all_wait_converted = []
    all_wait_direct = []
    all_ftf_wait_times = []
    all_target_days_e = []
    all_target_days_f = []
    
    total_blocked_e = 0
    total_blocked_f = 0
    total_arrivals_e_actual = 0
    total_arrivals_f_actual = 0
    total_served_e = 0
    total_served_f = 0
    
    for seed in range(1, config.num_replications + 1):
        if progress_callback:
            progress_callback(seed - 1, config.num_replications, "simulation")
        rep_result = run_single_replication(seed, config)
        
        all_daily_q_e.append(rep_result['daily_q_e'])
        all_daily_q_f.append(rep_result['daily_q_f'])
        all_daily_blocked_e.append(rep_result['daily_blocked_e'])
        all_daily_blocked_f.append(rep_result['daily_blocked_f'])
        all_daily_arrivals_e.append(rep_result['daily_arrivals_e'])
        daily_arrivals_f_total = rep_result['daily_arrivals_f_direct'] + rep_result['daily_arrivals_f_converted']
        all_daily_arrivals_f.append(daily_arrivals_f_total)
        
        if rep_result['target_day_e'] is not None:
            all_target_days_e.append(rep_result['target_day_e'])
        if rep_result['target_day_f'] is not None:
            all_target_days_f.append(rep_result['target_day_f'])
        
        all_wait_resolved.extend(rep_result['wait_times_resolved'])
        all_wait_converted.extend(rep_result['wait_times_converted'])
        all_wait_direct.extend(rep_result['wait_times_direct_ftf'])
        
        for day in range(config.warmup_days, config.sim_horizon):
            all_ftf_wait_times.extend(rep_result['daily_ftf_wait_times'][day])
        
        warmup = config.warmup_days
        total_blocked_e += np.sum(rep_result['daily_blocked_e'][warmup:])
        total_blocked_f += np.sum(rep_result['daily_blocked_f'][warmup:])
        total_served_e += np.sum(rep_result['daily_served_e'][warmup:])
        total_served_f += np.sum(rep_result['daily_served_f'][warmup:])
        
        total_arrivals_e_actual += np.sum(rep_result['daily_arrivals_e'][warmup:])
        total_arrivals_f_actual += np.sum(rep_result['daily_arrivals_f_direct'][warmup:]) + \
                                   np.sum(rep_result['daily_arrivals_f_converted'][warmup:])
    
    if progress_callback:
        progress_callback(config.num_replications, config.num_replications, "analysis")

    mean_daily_q_e = np.mean(all_daily_q_e, axis=0)
    mean_daily_q_f = np.mean(all_daily_q_f, axis=0)
    std_daily_q_e = np.std(all_daily_q_e, axis=0)
    std_daily_q_f = np.std(all_daily_q_f, axis=0)
    
    mean_daily_blocked_e = np.mean(all_daily_blocked_e, axis=0)
    mean_daily_blocked_f = np.mean(all_daily_blocked_f, axis=0)
    mean_daily_arrivals_e = np.mean(all_daily_arrivals_e, axis=0)
    mean_daily_arrivals_f = np.mean(all_daily_arrivals_f, axis=0)
    
    warmup = config.warmup_days
    avg_queue_e = np.mean([q[warmup:].mean() for q in all_daily_q_e])
    avg_queue_f = np.mean([q[warmup:].mean() for q in all_daily_q_f])
    
    avg_wait_e = np.mean(all_wait_resolved) if all_wait_resolved else 0
    avg_wait_f = np.mean(all_ftf_wait_times) if all_ftf_wait_times else 0
    avg_wait_converted = np.mean(all_wait_converted) if all_wait_converted else 0
    avg_wait_direct = np.mean(all_wait_direct) if all_wait_direct else 0
    
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
    
    block_rate_e = (total_blocked_e / total_arrivals_e_actual * 100) if total_arrivals_e_actual > 0 else 0
    block_rate_f = (total_blocked_f / total_arrivals_f_actual * 100) if total_arrivals_f_actual > 0 else 0
    
    analysis_days_total = config.analysis_days * config.num_replications
    throughput = (total_served_e + total_served_f) / analysis_days_total if analysis_days_total > 0 else 0
    
    rho_e_theoretical = config.rho_e
    rho_f_theoretical = config.rho_f
    
    total_capacity_e = config.c_e * config.analysis_days * config.num_replications
    total_capacity_f = config.c_f * config.analysis_days * config.num_replications
    rho_e_empirical = total_served_e / total_capacity_e if total_capacity_e > 0 else 0
    rho_f_empirical = total_served_f / total_capacity_f if total_capacity_f > 0 else 0
    
    avg_target_day_e = np.mean(all_target_days_e) if all_target_days_e else None
    avg_target_day_f = np.mean(all_target_days_f) if all_target_days_f else None
    pct_reached_target_e = len(all_target_days_e) / config.num_replications * 100
    pct_reached_target_f = len(all_target_days_f) / config.num_replications * 100
    
    results = {
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
        'avg_wait_e': avg_wait_e,
        'avg_wait_f': avg_wait_f,
        'avg_wait_converted': avg_wait_converted,
        'avg_wait_direct': avg_wait_direct,
        'weighted_avg_wait': weighted_avg_wait,
        'avg_queue_e': avg_queue_e,
        'avg_queue_f': avg_queue_f,
        'block_rate_e': block_rate_e,
        'block_rate_f': block_rate_f,
        'rho_e': rho_e_empirical,
        'rho_f': rho_f_empirical,
        'rho_e_theoretical': rho_e_theoretical,
        'rho_f_theoretical': rho_f_theoretical,
        'rho_e_empirical': rho_e_empirical,
        'rho_f_empirical': rho_f_empirical,
        'throughput': throughput,
        'warmup_day': config.warmup_days,
        'buffer_size_e': config.buffer_size_e,
        'buffer_size_f': config.buffer_size_f,
        'n_resolved': n_resolved,
        'n_converted': n_converted,
        'n_direct': n_direct,
        'avg_target_day_e': avg_target_day_e,
        'avg_target_day_f': avg_target_day_f,
        'pct_reached_target_e': pct_reached_target_e,
        'pct_reached_target_f': pct_reached_target_f,
        'target_q_e': config.target_q_e,
        'target_q_f': config.target_q_f,
        'initial_q_e': config.initial_q_e,
        'initial_q_f': config.initial_q_f,
    }
    
    return results


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
