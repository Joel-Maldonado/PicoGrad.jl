mutable struct Value
    data::Float64
    grad::Float64
    prev::Vector{Value}
    op::String
    backward::Function
end

# Leaf constructor: numeric to Value
Value(x::Real) = Value(float(x), 0.0, Value[], "", ()->nothing)

import Base: +, -, *, /, ^, show

# Addition
function +(x::Value, y::Value)
    out = Value(x.data + y.data, 0.0, Value[x, y], "+", ()->nothing)
    out.backward = function()
        x.grad += out.grad
        y.grad += out.grad
    end
    return out
end
+(x::Value, y::Real) = x + Value(y)
+(x::Real, y::Value) = Value(x) + y

# Multiplication
function *(x::Value, y::Value)
    out = Value(x.data * y.data, 0.0, Value[x, y], "*", ()->nothing)
    out.backward = function()
        x.grad += y.data * out.grad
        y.grad += x.data * out.grad
    end
    return out
end
*(x::Value, y::Real) = x * Value(y)
*(x::Real, y::Value) = Value(x) * y

# Power (scalar exponent only)
function ^(x::Value, p::Real)
    out = Value(x.data ^ float(p), 0.0, Value[x], "**$(p)", ()->nothing)
    out.backward = function()
        x.grad += (float(p) * (x.data ^ (float(p) - 1))) * out.grad
    end
    return out
end

# ReLU activation
function relu(x::Value)
    d = x.data < 0.0 ? 0.0 : x.data
    out = Value(d, 0.0, Value[x], "ReLU", ()->nothing)
    out.backward = function()
        x.grad += ((out.data > 0.0) ? 1.0 : 0.0) * out.grad
    end
    return out
end

# Unary negation
-(x::Value) = x * (-1.0)

# Subtraction
-(x::Value, y::Value) = x + (-y)
-(x::Value, y::Real) = x + (-Value(y))
-(x::Real, y::Value) = Value(x) + (-y)

# Division (via reciprocal power)
/(x::Value, y::Value) = x * (y ^ -1.0)
/(x::Value, y::Real) = x * (Value(y) ^ -1.0)
/(x::Real, y::Value) = Value(x) * (y ^ -1.0)

# Reverse-mode autodiff
function backward!(self::Value)
    # Compared to micrograd's recursive topo build, this uses
    # an explicit stack-based DFS to avoid recursion limits while
    # producing the same post-order for backprop.
    topo = Value[]
    visited = Base.IdSet{Value}()
    stack = Tuple{Value,Bool}[]  # (node, expanded?)
    push!(stack, (self, false))

    while !isempty(stack)
        v, expanded = pop!(stack)
        if expanded
            push!(topo, v)
        elseif !(v in visited)
            push!(visited, v)
            push!(stack, (v, true))
            # push children in reverse
            for child in reverse(v.prev)
                if !(child in visited)
                    push!(stack, (child, false))
                end
            end
        end
    end

    # Seed gradient on the terminal node
    self.grad = 1.0
    for v in reverse(topo)
        v.backward()
    end
    nothing
end

# Display
function show(io::IO, v::Value)
    print(io, "Value(data=$(v.data), grad=$(v.grad))")
end

