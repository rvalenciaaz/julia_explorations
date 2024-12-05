using NeuralPDE
using OrdinaryDiffEq, Plots, Lux, Random, OptimizationOptimisers

# Define the ODE problem
linear(u, p, t) = cos(t * 2 * pi)
tspan = (0.0, 1.0)
u0 = 0.0
prob = ODEProblem(linear, u0, tspan)

# Initialize the neural network
rng = Random.default_rng()
Random.seed!(rng, 0)
chain = Chain(Dense(1, 5, σ), Dense(5, 1))
ps, st = Lux.setup(rng, chain) |> Lux.f64

# Setup optimizer and algorithm
opt = Adam(0.1)
alg = NNODE(chain, opt, init_params = ps)

# Solve the problem
sol = solve(prob, alg, verbose = true, maxiters = 2000, saveat = 0.01)

# Solve with a traditional method for ground truth
ground_truth = solve(prob, Tsit5(), saveat = 0.01)

# Plot results
plot(ground_truth, label = "ground truth")
plot!(sol.t, sol.u, label = "pred")

# Save the plot
savefig("ode_solution_plot.png")

