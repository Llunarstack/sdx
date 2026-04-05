# Rust — `sdx-noise-schedule`

Print **VP-DDPM-style** discrete schedules as CSV (`step,beta,alpha,alpha_bar,snr_db`) for plotting or comparing to `diffusion/` Python code.

```bash
cargo build --release
target/release/sdx-noise-schedule linear --steps 1000 --beta-start 0.0001 --beta-end 0.02
target/release/sdx-noise-schedule cosine --steps 1000 --s 0.008
```

Educational / pre-training sanity — not wired into `train.py` (PyTorch owns the live schedule).
