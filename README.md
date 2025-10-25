# PicoGrad.jl

A tiny scalar autograd engine in Julia, plus a very small neural networks library on top. Based on Andrej Karpathy’s [micrograd](https://github.com/karpathy/micrograd), this project reimplements the engine in Julia, makes it functional rather than OOP, and adds modest improvements while keeping it concise, simple, and readable.

> Note: This is not a professional project, and is meant for fun and experimentation. It was also used as an educational exercise for me and can also be used like that for anyone curious to read the code. This project is kept intentionally small, concise, simple, and easy to read and understand.

### Getting started

```bash
julia examples/autograd_example.jl   # forward/backward on a scalar expression
julia examples/nn_example.jl         # train XOR with MSE + SGD
```

> Requirements: Julia (1.12 recommended). 

### Example

Below is a short Julia sketch showing the basic usage (mirrored example from micrograd for reference):

```julia
include("src/engine.jl")

a = Value(-4.0)
b = Value(2.0)
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

# Backward pass: populate gradients via autodiff
backward!(g)

# Inspect gradients of inputs
println(a.grad)  # ~138.8338
println(b.grad)  # ~645.5773
```

### Training

`nn_example.jl` trains a tiny MLP on XOR using mean squared error and a simple SGD loop. Run it with:

```bash
julia examples/nn_example.jl
```

Training output:

```
epoch 50: loss = 0.14017297341446447
epoch 100: loss = 0.0658203910391573
epoch 150: loss = 0.011419044527953035
epoch 200: loss = 0.001282916375132894
epoch 250: loss = 0.00011858058069798822
epoch 300: loss = 1.0359059272538363e-5
epoch 350: loss = 8.964231523074673e-7
epoch 400: loss = 7.771685380572547e-8
epoch 450: loss = 6.539262865335369e-9
epoch 500: loss = 5.670266403114571e-10

Final predictions:
x=[0.0, 0.0] -> y=3.983029754162726e-5 (target=0.0)
x=[0.0, 1.0] -> y=0.9999801682042224 (target=1.0)
x=[1.0, 0.0] -> y=0.9999897957666979 (target=1.0)
x=[1.0, 1.0] -> y=7.308470523104127e-6 (target=0.0)
```

### Visualization (optional)

Install Plots.jl once and run any of the scripts:

```julia
using Pkg; Pkg.add("Plots")
julia examples/visualize_decision_boundary.jl   # XOR decision boundary → images/xor_boundary.png
```

![XOR decision boundary](images/xor_boundary.png)

### Files

- `src/engine.jl` — scalar `Value`, arithmetic, `relu`, and `backward!` (stack‑based topo).
- `src/nn.jl` — `Neuron`, `Layer`, `MLP`, parameter flattening, and `zero_grad!`.
- `examples/autograd_example.jl` — expression + gradients demo.
- `examples/nn_example.jl` — XOR training demo with MSE + SGD.
- `examples/visualize_decision_boundary.jl` — optional XOR decision boundary plot (requires Plots.jl).

### Notes

- Scalars only. Vectors are plain Julia arrays of `Value`.
- Gradients accumulate across calls, so you need to reset with `zero_grad!`.
- A layer returns a single `Value` when `nout == 1`, otherwise a vector.

### License

MIT

### Thanks

Heavily inspired by Andrej Karpathy's Micrograd, implemented in Julia as an educational exercise.
