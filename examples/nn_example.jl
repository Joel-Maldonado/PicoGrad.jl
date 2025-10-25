include(joinpath(@__DIR__, "../src/engine.jl"))
include(joinpath(@__DIR__, "../src/nn.jl"))

using Random

# Reproducible init
Random.seed!(42)

# XOR dataset: inputs and targets
data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

# Model: 2 -> 4 -> 1
m = MLP(2, [4, 1])

# Simple SGD loop
lr = 0.1
epochs = 500
for e in 1:epochs
    # Forward: mean squared error (MSE) over dataset
    L = Value(0.0)
    for (x, t) in data
        y = m(x)              # single Value
        L = L + (y - Value(t))^2
    end
    L = L / length(data)

    # Backward and step
    zero_grad!(m)
    backward!(L)
    for p in parameters(m)
        p.data -= lr * p.grad
    end

    if e % 50 == 0
        println("epoch $e: loss = $(L.data)")
    end
end

# Evaluate final predictions
println("\nFinal predictions:")
for (x, t) in data
    y = m(x)
    println("x=$(x) -> y=$(y.data) (target=$(t))")
end

