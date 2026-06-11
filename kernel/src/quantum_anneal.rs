//! Quantum Annealing — QUBO solver for kernel scheduling optimization.
//!
//! Implements a simulated quantum annealer that solves Quadratic Unconstrained
//! Binary Optimization (QUBO) problems. Maps competing tasks/processes to
//! qubits and finds the minimum energy configuration — the optimal schedule.
//!
//! QUBO formulation: minimize x^T * Q * x where x ∈ {0,1}^n
//!
//! Integration:
//!   - solve() for one-shot QUBO optimization
//!   - schedule_tasks() to prioritize N competing processes
//!   - /proc/quantum_anneal for solver statistics

use alloc::vec::Vec;
use alloc::format;
use alloc::string::String;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use spin::Mutex;
use libm::{expf, powf, truncf};

/// Whether the annealer is active.
static ANNEAL_ACTIVE: AtomicBool = AtomicBool::new(false);
/// Number of annealing runs performed.
static ANNEAL_RUNS: AtomicU64 = AtomicU64::new(0);

/// Maximum QUBO size (number of qubits/variables).
const MAX_QUBO_SIZE: usize = 16;

/// Annealing schedule parameters.
const INITIAL_TEMP: f32 = 10.0;
const FINAL_TEMP: f32 = 0.01;
const STEPS_PER_RUN: usize = 100;

/// Annealer state.
struct AnnealState {
    /// Number of qubits in current problem.
    n_qubits: usize,
    /// Current Q matrix (upper triangular).
    q_matrix: [[f32; MAX_QUBO_SIZE]; MAX_QUBO_SIZE],
    /// Best solution found.
    best_solution: [u8; MAX_QUBO_SIZE],
    /// Energy of best solution.
    best_energy: f32,
    /// Current temperature.
    temperature: f32,
    /// Total iterations performed.
    iterations: u64,
    /// Number of accepted transitions.
    accepted: u64,
}

static STATE: Mutex<Option<AnnealState>> = Mutex::new(None);

/// Initialize the quantum annealer.
pub fn init() {
    let state = AnnealState {
        n_qubits: 4,
        q_matrix: [[0.0; MAX_QUBO_SIZE]; MAX_QUBO_SIZE],
        best_solution: [0; MAX_QUBO_SIZE],
        best_energy: f32::MAX,
        temperature: INITIAL_TEMP,
        iterations: 0,
        accepted: 0,
    };

    let mut lock = STATE.lock();
    *lock = Some(state);
    ANNEAL_ACTIVE.store(true, Ordering::Release);
    crate::klog!(INFO, "quantum_anneal: QUBO solver initialized (max {} qubits)", MAX_QUBO_SIZE);
}

/// Compute energy of a solution vector x for Q matrix.
fn compute_energy(x: &[u8; MAX_QUBO_SIZE], n: usize, q: &[[f32; MAX_QUBO_SIZE]; MAX_QUBO_SIZE]) -> f32 {
    let mut energy = 0.0_f32;
    for i in 0..n {
        for j in 0..n {
            if x[i] == 1 && x[j] == 1 {
                energy += q[i][j];
            }
        }
    }
    energy
}

/// Solve a QUBO problem: minimize x^T * Q * x.
/// Q is the upper-triangular weight matrix.
/// Returns (solution_bitstring, energy).
pub fn solve(q: &[[f32; MAX_QUBO_SIZE]; MAX_QUBO_SIZE], n: usize) -> ([u8; MAX_QUBO_SIZE], f32) {
    if !ANNEAL_ACTIVE.load(Ordering::Acquire) || n == 0 || n > MAX_QUBO_SIZE {
        return ([0; MAX_QUBO_SIZE], f32::MAX);
    }

    let mut lock = STATE.lock();
    let state = match &mut *lock {
        Some(s) => s,
        None => return ([0; MAX_QUBO_SIZE], f32::MAX),
    };

    // Copy Q matrix
    state.q_matrix = *q;
    state.n_qubits = n;

    // Initialize with random solution
    let uptime = crate::scheduler::uptime_ms();
    let mut current: [u8; MAX_QUBO_SIZE] = [0; MAX_QUBO_SIZE];
    for i in 0..n {
        current[i] = ((uptime as usize + i * 7) % 2) as u8;
    }

    let mut current_energy = compute_energy(&current, n, &state.q_matrix);
    state.best_solution = current;
    state.best_energy = current_energy;

    // Simulated annealing loop
    let mut temp = INITIAL_TEMP;
    for step in 0..STEPS_PER_RUN {
        // Cool down
        let cooling = powf(FINAL_TEMP / INITIAL_TEMP, step as f32 / STEPS_PER_RUN as f32);
        temp = INITIAL_TEMP * cooling;

        // Try flipping each qubit
        for i in 0..n {
            let mut candidate = current;
            candidate[i] ^= 1; // Flip bit i

            let candidate_energy = compute_energy(&candidate, n, &state.q_matrix);
            let delta_e = candidate_energy - current_energy;

            // Accept if better, or with Boltzmann probability
            if delta_e < 0.0 || (temp > 0.0 && {
                let r_raw = (uptime as f32 * 0.01 + i as f32 * 0.1);
                let r = r_raw - truncf(r_raw);
                r < expf(-delta_e / temp)
            }) {
                current = candidate;
                current_energy = candidate_energy;
                state.accepted = state.accepted.saturating_add(1);

                if current_energy < state.best_energy {
                    state.best_solution = current;
                    state.best_energy = current_energy;
                }
            }

            state.iterations = state.iterations.saturating_add(1);
        }
    }

    state.temperature = temp;
    ANNEAL_RUNS.fetch_add(1, Ordering::Relaxed);

    (state.best_solution, state.best_energy)
}

/// Schedule N tasks by priority — maps to QUBO and solves.
/// Higher priority tasks tend to be scheduled (bit=1).
/// Returns a bitmap where bit i = 1 means task i is selected for execution.
pub fn schedule_tasks(priorities: &[f32]) -> ([u8; MAX_QUBO_SIZE], f32) {
    let n = priorities.len().min(MAX_QUBO_SIZE);
    if n == 0 { return ([0; MAX_QUBO_SIZE], 0.0); }

    // Build Q matrix: minimize energy = -sum(priority_i * x_i) + sum(x_i * x_j * penalty)
    // This creates competition: only the highest-priority tasks get selected.
    let mut q = [[0.0_f32; MAX_QUBO_SIZE]; MAX_QUBO_SIZE];
    for i in 0..n {
        // Diagonal: negative priority (we want to select high-priority tasks)
        q[i][i] = -priorities[i] * 2.0;
        for j in (i+1)..n {
            // Off-diagonal: competition penalty (can't schedule everything)
            q[i][j] = 1.0;
        }
    }

    solve(&q, n)
}

/// Get solver statistics.
pub fn total_runs() -> u64 {
    ANNEAL_RUNS.load(Ordering::Relaxed)
}

/// Format /proc/quantum_anneal report.
pub fn format_report() -> Vec<u8> {
    let active = ANNEAL_ACTIVE.load(Ordering::Acquire);
    if !active {
        return format!("Quantum Annealer (QUBO)\nNot initialized\n").into_bytes();
    }
    let lock = STATE.lock();
    match &*lock {
        Some(s) => {
            let solution_str: String = (0..s.n_qubits)
                .map(|i| if s.best_solution[i] == 1 { '1' } else { '0' })
                .collect();

            format!(
                "Quantum Annealer (QUBO Solver)\n\
                 ===============================\n\
                 max_qubits:    {}\n\
                 current_n:    {}\n\
                 best_energy:  {:.4}\n\
                 best_solution: {}\n\
                 temperature:  {:.4}\n\
                 iterations:   {}\n\
                 accepted:     {}\n\
                 runs:         {}\n\
                 \n\
                 Annealing schedule: {}→{} over {} steps\n\
                 Used for: attention scheduling, task prioritization\n",
                MAX_QUBO_SIZE,
                s.n_qubits,
                s.best_energy,
                solution_str,
                s.temperature,
                s.iterations,
                s.accepted,
                ANNEAL_RUNS.load(Ordering::Relaxed),
                INITIAL_TEMP, FINAL_TEMP, STEPS_PER_RUN,
            ).into_bytes()
        }
        None => format!("Quantum Annealer (QUBO)\nUninitialized\n").into_bytes(),
    }
}
