using NeuralPDE, Lux, Plots, OrdinaryDiffEq, Distributions, Random

function lotka_volterra(u, p, t)
    α, β, γ, δ = p
    x, y = u
    dx = (α - β * y) * x
    dy = (δ * x - γ) * y
    [dx, dy]
end

# Initial conditions and parameters
u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 4.0)
prob = ODEProblem(lotka_volterra, u0, tspan, p)

# Solve the ODE
dt = 0.01
solution = solve(prob, Tsit5(); saveat = dt)

# Create noisy dataset
time = solution.t
u = hcat(solution.u...)
x_noisy = u[1, :] .+ (u[1, :]) .* (0.3 .* randn(length(u[1, :])))
y_noisy = u[2, :] .+ (u[2, :]) .* (0.3 .* randn(length(u[2, :])))

dataset = [x_noisy, y_noisy, time]

# Neural network architecture
chain = Chain(Dense(1, 6, tanh), Dense(6, 6, tanh), Dense(6, 2))

# Parameter estimation
alg = BNNODE(chain;
    dataset = dataset,
    draw_samples = 1000,
    l2std = [0.1, 0.1],
    phystd = [0.1, 0.1],
    priorsNNw = (0.0, 3.0),
    param = [Normal(1,2), Normal(2,2), Normal(2,2), Normal(0,2)],
    progress = false
)

sol_pestim = solve(prob, alg; saveat = dt)

# Plot noisy data and original solution
p1 = plot(time, x_noisy, label = "Noisy x", xlabel = "time", ylabel = "population")
plot!(p1, time, y_noisy, label = "Noisy y")
plot!(p1, solution, labels = ["True x" "True y"])
savefig(p1, "noisy_and_true_solution.png")

# Plot estimated solution from BNNODE and compare with true solution
p2 = plot(time, sol_pestim.ensemblesol[1], label = "Estimated x", xlabel = "time", ylabel = "population")
plot!(p2, time, sol_pestim.ensemblesol[2], label = "Estimated y")
plot!(p2, solution, labels = ["True x" "True y"])
savefig(p2, "estimated_vs_true_solution.png")
