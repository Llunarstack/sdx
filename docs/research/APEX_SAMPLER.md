# APEX — Adaptive Predictive EXtrapolation

Co-designed **sampler + scheduler** for SDX VP diffusion and rectified flow.

## Why not “another Euler”

Fixed steppers (Euler, DDIM, DPM++, UniPC) use a predetermined step sequence. APEX keeps a strong **static** default (reproducible / batchable) and optionally **adapts remaining timesteps** from embedded error, curvature, and phase.

Grounded in:

- DPM-Solver / DPM-Solver++ embedded order error (Lu et al.)
- Karras EDM σ spacing
- Align-Your-Steps mid-band density
- Classical adaptive ODE step control (atol / rtol)

## Components

| Piece | Path |
|-------|------|
| Core math + controller | [`diffusion/solvers/apex.py`](../diffusion/solvers/apex.py) |
| VP schedule `apex` | [`diffusion/inference_timesteps.py`](../diffusion/inference_timesteps.py) |
| Flow schedule `apex` | [`diffusion/solvers/flow_ode.py`](../diffusion/solvers/flow_ode.py) |
| Loop wiring | [`diffusion/gaussian_diffusion.py`](../diffusion/gaussian_diffusion.py) |

## Modes

### Static (`--solver apex` + `--scheduler apex`)

1. Build three-phase timestep density (composition → reconstruction → refinement).
2. Each step: one model eval → DPM++ order-1 and order-2+ from the **same** `x0` (no extra NFE).
3. Normalized error `E = ‖x_hi − x_lo‖ / (atol + rtol‖x‖)`.
4. If `E` is large, blend toward the safer low-order update (CFG-stable).
5. Phase picks recommended multistep order (structure ≤2, mid ≤3, refine ≤2).

### Adaptive (`--solver apex_adaptive`)

Same as static, plus after each step **rebuild the remaining** discrete indices with difficulty

`D ∝ 1 + αE + βκ + γV + spatial_topk(Δx0) + phase`

so the NFE budget stays fixed while spacing concentrates where the trajectory is hard.

## CLI

```bash
# Recommended APEX stills
python sample.py --ckpt ... --prompt "..." --steps 28 \
  --scheduler apex --solver apex

# Adaptive remaining-grid
python sample.py --ckpt ... --prompt "..." --steps 28 \
  --scheduler apex --solver apex_adaptive \
  --apex-atol 0.0078 --apex-rtol 0.05

# Flow
python sample.py --ckpt ... --flow-matching-sample \
  --flow-schedule apex --flow-solver apex --steps 28
```

## Honest limits

- Not a trained neural scheduler; difficulty is hand-designed.
- Spatial term is a **global** difficulty scalar, not per-pixel ODE stepping.
- True reject-and-retry with new NFEs is avoided to keep step budget predictable; soft blend + redistribute approximate classical adaptive solvers.
- Always benchmark against `dpmpp_2m` + `ays_dit` / `unipc` on your ckpt before changing defaults.

## Benchmark checklist

Compare at equal NFE (e.g. 12 / 20 / 28 steps):

1. `dpmpp_2m` + `ays_dit`
2. `unipc` + `ays_dit`
3. `apex` + `apex`
4. `apex_adaptive` + `apex`

Metrics: CLIP/prompt adherence, aesthetic/ViT pick scores, hand/face failure rate, wall time.
