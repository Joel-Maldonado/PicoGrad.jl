using Random

struct Neuron
    w::Vector{Value}
    b::Value
    nonlin::Bool
end

function Neuron(nin::Integer; nonlin::Bool=true)
    ws = [Value(rand() * 2 - 1) for _ in 1:Integer(nin)]
    b = Value(0.0)
    return Neuron(ws, b, nonlin)
end

(n::Neuron)(x::AbstractVector) = begin
    act = n.b
    for (wi, xi) in zip(n.w, x) # truncates to min length
        act = act + wi * xi
    end
    n.nonlin ? relu(act) : act
end

parameters(n::Neuron) = [n.w...; n.b]

struct Layer
    neurons::Vector{Neuron}
end

function Layer(nin::Integer, nout::Integer; nonlin::Bool=true)
    ns = [Neuron(nin; nonlin=nonlin) for _ in 1:Integer(nout)]
    return Layer(ns)
end

(l::Layer)(x) = begin
    out = [n(x) for n in l.neurons]
    length(out) == 1 ? out[1] : out
end

parameters(l::Layer) = vcat((parameters(n) for n in l.neurons)...)

struct MLP
    layers::Vector{Layer}
end

function MLP(nin::Integer, nouts::AbstractVector{<:Integer})
    sz = vcat(Integer(nin), Integer.(nouts))
    layers = Layer[
        Layer(sz[i], sz[i+1]; nonlin = i != length(nouts)) for i in 1:length(nouts)
    ]
    return MLP(layers)
end

(m::MLP)(x) = begin
    for layer in m.layers
        x = layer(x)
    end
    x
end

parameters(m::MLP) = vcat((parameters(layer) for layer in m.layers)...)

function zero_grad!(x::Neuron)
    for p in parameters(x)
        p.grad = 0.0
    end
    nothing
end

function zero_grad!(x::Layer)
    for p in parameters(x)
        p.grad = 0.0
    end
    nothing
end

function zero_grad!(x::MLP)
    for p in parameters(x)
        p.grad = 0.0
    end
    nothing
end

