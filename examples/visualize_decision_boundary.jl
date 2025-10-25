include(joinpath(@__DIR__, "../src/engine.jl"))
include(joinpath(@__DIR__, "../src/nn.jl"))

using Random
using Plots

# Train a tiny MLP on XOR
Random.seed!(42)
data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

m = MLP(2, [4, 1])
lr = 0.1
epochs = 500
for e in 1:epochs
    L = Value(0.0)
    for (x, t) in data
        y = m(x)
        L = L + (y - Value(t))^2
    end
    L = L / length(data)
    zero_grad!(m)
    backward!(L)
    for p in parameters(m)
        p.data -= lr * p.grad
    end
    if e % 100 == 0
        println("epoch $e: loss = $(L.data)")
    end
end

# Grid and decision boundary (simple and readable)
xr = range(-0.25, 1.25, length=200)
yr = range(-0.25, 1.25, length=200)
Z = Array{Float64}(undef, length(yr), length(xr))
for (j, yv) in enumerate(yr)
    for (i, xv) in enumerate(xr)
        z = m([xv, yv])
        Z[j, i] = z.data
    end
end

p = contourf(xr, yr, Z; c=:viridis, levels=20, title="XOR decision boundary")

x0 = [x[1] for (x, t) in data if t == 0.0]
y0 = [x[2] for (x, t) in data if t == 0.0]
x1 = [x[1] for (x, t) in data if t == 1.0]
y1 = [x[2] for (x, t) in data if t == 1.0]
scatter!(p, x0, y0; color=:red, marker=:circle, ms=6, label="class 0")
scatter!(p, x1, y1; color=:blue, marker=:diamond, ms=6, label="class 1")

# Save under images/
imgdir = joinpath(@__DIR__, "../images")
isdir(imgdir) || mkpath(imgdir)
outfile = joinpath(imgdir, "xor_boundary.png")
savefig(p, outfile)
println("Saved $(outfile)")

