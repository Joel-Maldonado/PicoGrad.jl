include(joinpath(@__DIR__, "../src/engine.jl"))

# Build a small scalar expression graph
a = Value(-4.0)
b = Value(2.0)

# Forward pass: compose operations and activations
c = a + b
d = a * b + b ^ 3
c = c + (c + 1)
c = c + (1 + c + (-a))
d = d + (d * 2 + relu(b + a))
d = d + (3 * d + relu(b - a))
e = c - d
f = e ^ 2
g = f / 2.0
g = g + 10.0 / f

println(g.data) # scalar output of the forward pass (~24.7041)

# Backward pass: populate gradients via reverse-mode autodiff
backward!(g)

# Inspect gradients of inputs
println(a.grad)  # ~138.8338
println(b.grad)  # ~645.5773

