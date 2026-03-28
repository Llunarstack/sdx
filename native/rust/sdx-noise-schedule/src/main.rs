//! Discrete noise schedules for VP-DDPM-style training analysis (no GPU).
//! Outputs CSV: step,beta,alpha,alpha_bar,snr_db
//!
//! - **linear**: linear betas in [beta_start, beta_end] (DDPM-style).
//! - **cosine**: Improved-DDPM-style alpha_bar (Nichol & Dhariwal), then inferred betas.

use clap::{Parser, Subcommand};
use std::f64::consts::FRAC_PI_2;

#[derive(Parser, Debug)]
#[command(name = "sdx-noise-schedule", version, about = "Diffusion noise schedule tables (CSV)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Linear betas from beta_start to beta_end over T steps.
    Linear {
        #[arg(long, default_value_t = 1000)]
        steps: usize,
        #[arg(long, default_value_t = 1e-4)]
        beta_start: f64,
        #[arg(long, default_value_t = 2e-2)]
        beta_end: f64,
    },
    /// Cosine alpha_bar schedule (s offset matches diffusers-style default ~0.008).
    Cosine {
        #[arg(long, default_value_t = 1000)]
        steps: usize,
        #[arg(long, default_value_t = 0.008)]
        s: f64,
    },
}

fn linear_schedule(t: usize, beta_start: f64, beta_end: f64) -> Vec<(f64, f64, f64)> {
    if t == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(t);
    let mut ab = 1.0_f64;
    for i in 0..t {
        let beta = beta_start + (beta_end - beta_start) * (i as f64) / ((t - 1).max(1) as f64);
        let beta = beta.clamp(1e-8, 0.999);
        let alpha = 1.0 - beta;
        ab *= alpha;
        out.push((beta, alpha, ab));
    }
    out
}

/// alpha_bar_t for timestep t in [0, T], cosine schedule.
fn cosine_alpha_bar(step: usize, cap_t: usize, s: f64) -> f64 {
    let x = step as f64;
    let big_t = cap_t as f64;
    let f = |v: f64| ((v / big_t + s) / (1.0 + s) * FRAC_PI_2).cos();
    (f(x) / f(0.0)).powi(2)
}

fn cosine_schedule(steps: usize, s: f64) -> Vec<(f64, f64, f64)> {
    if steps == 0 {
        return Vec::new();
    }
    let alpha_bars: Vec<f64> = (0..=steps).map(|t| cosine_alpha_bar(t, steps, s)).collect();
    let mut out = Vec::with_capacity(steps);
    for t in 0..steps {
        let ab = alpha_bars[t];
        let ab_next = alpha_bars[t + 1];
        let beta = (1.0 - ab_next / ab).clamp(1e-8, 0.999);
        let alpha = 1.0 - beta;
        out.push((beta, alpha, ab));
    }
    out
}

fn snr_db(alpha_bar: f64) -> f64 {
    if alpha_bar <= 0.0 || alpha_bar >= 1.0 {
        return f64::NAN;
    }
    10.0 * (alpha_bar / (1.0 - alpha_bar)).log10()
}

fn print_csv(rows: &[(f64, f64, f64)]) {
    println!("step,beta,alpha,alpha_bar,snr_db");
    for (i, (beta, alpha, ab)) in rows.iter().enumerate() {
        let snr = snr_db(*ab);
        if snr.is_finite() {
            println!("{i},{beta:.9},{alpha:.9},{ab:.9},{snr:.6}");
        } else {
            println!("{i},{beta:.9},{alpha:.9},{ab:.9},");
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let rows = match cli.cmd {
        Cmd::Linear {
            steps,
            beta_start,
            beta_end,
        } => {
            if steps < 1 {
                eprintln!("steps must be >= 1");
                std::process::exit(2);
            }
            linear_schedule(steps, beta_start, beta_end)
        }
        Cmd::Cosine { steps, s } => {
            if steps < 1 {
                eprintln!("steps must be >= 1");
                std::process::exit(2);
            }
            cosine_schedule(steps, s)
        }
    };
    print_csv(&rows);
}
