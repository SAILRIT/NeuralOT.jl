# Flow matching (Lipman et al., 2023)

## Problem

Flow matching trains a time-dependent vector field
`v_θ : [0, 1] × ℝᵈ → ℝᵈ` so that the ODE

```math
\frac{\mathrm{d} x}{\mathrm{d} t} = v_\theta(t, x),\quad x(0) \sim \mu
```

produces `x(1) ∼ ν`. Rather than a saddle-point objective, flow matching
uses a simple regression loss. For independent samples `X₀ ∼ μ`,
`X₁ ∼ ν`, the straight-line interpolant is

```math
X_t = (1 - t) X_0 + t X_1, \qquad t \sim \mathrm{Unif}(0, 1),
```

and the conditional velocity is the constant `X₁ - X₀`. The training
objective is

```math
\mathcal{L}(\theta) = \mathbb{E}_{t, X_0, X_1}\!
  \left\| v_\theta(t, X_t) - (X_1 - X_0) \right\|^2.
```

This is simulation-free: no ODE solve during training. The learned flow
is a transport-style generative model, and iterating the procedure
(*rectified flow*; Liu et al., 2023) straightens the paths toward the
true OT map.

## Usage

```julia
sample_μ(n) = randn(Float32, 2, n)              # source: noise
sample_ν(n) = my_data_sampler(n)                # target: data

result = flow_match(sample_μ, sample_ν;
                    dim=2, hidden=[128, 128, 128],
                    steps=10_000, batch=256, lr=5e-4)

# Generate samples by integrating the flow
X0 = sample_μ(1_000)
X1 = monge_map(result, X0; n_flow_steps=100)
```

## Integration

The package provides a simple explicit-Euler integrator
(`NeuralOT.integrate_flow`) adequate for smooth vector fields. For stiff
flows or adaptive stepping, couple the learned `vfield` with
[DifferentialEquations.jl](https://diffeq.sciml.ai):

```julia
using DifferentialEquations
v = result.models.vfield
f!(du, u, p, t) = (du .= vec(v(vcat(Float32(t), u))))
prob = ODEProblem(f!, vec(X0[:, 1]), (0.0, 1.0))
sol = solve(prob, Tsit5())
```

## Tuning notes

- **hidden**: deeper/wider networks help when ν is multimodal. Start at
  `[128, 128]` for d ≤ 10.
- **steps**: flow matching needs more optimisation steps than `solve_w2`
  because the regression target has high variance (random pairing).
  10k–50k is typical.
- **n_flow_steps** (at inference): 50–200 Euler steps. If samples look
  blurred, use more steps or switch to a higher-order solver.
- **Rectification**: to straighten, retrain `flow_match` using
  `(X₀, monge_map(result, X₀))` pairs as the target — this is outside the
  current API but trivial to script.
