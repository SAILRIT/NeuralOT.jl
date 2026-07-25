# API reference

```@meta
CurrentModule = NeuralOT
```

## Solvers

```@docs
solve_dual
solve_w2
flow_match
rectify
NeuralOTResult
```

## Models

```@docs
ICNN
DualPotentialNet
VelocityNet
input_gradient
grad_x
```

## Maps

```@docs
monge_map
inverse_map
pushforward
barycentric_map
entropic_map
integrate_flow
```

## Costs

```@docs
SqEuclideanCost
EuclideanCost
GenericCost
cost_matrix
```

## Discrete optimal transport

```@docs
sinkhorn_potentials
sinkhorn_plan
sinkhorn_value
sinkhorn_cost
```

## Metrics and references

```@docs
sinkhorn_divergence
energy_distance
mmd
moment_error
transport_error
gaussian_ot
gaussian_brenier_map
w2_gaussian
```

## Data sets

```@docs
gaussian_sampler
two_moons
eight_gaussians
checkerboard
swiss_roll
circles
uniform_box
```
