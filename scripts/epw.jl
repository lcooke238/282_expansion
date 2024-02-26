using ProgressBars
using DataStructures

# Macros which control various optimizations.
USE_PARALLEL_HASH_MAP = true
AVOID_MALLOC = true

mutable struct ParallelDict{K,V} <: AbstractDict{K,V}
    subdicts::Vector{Dict{K,V}}
    locks::Vector{ReentrantLock}

    function ParallelDict{K,V}(num_subdicts::Int) where {V} where {K}
        subdicts = Vector{Dict{K,V}}(undef, num_subdicts)
        locks = Vector{ReentrantLock}(undef, num_subdicts)
        for subdict_index = 1:num_subdicts
            subdicts[subdict_index] = Dict{K,V}()
            locks[subdict_index] = ReentrantLock()
        end
        new(subdicts, locks)
    end
end

ParallelDict{K,V}() where {V} where {K} =
    ParallelDict{K,V}(USE_PARALLEL_HASH_MAP ? Threads.nthreads() * 2 : 1)

subdict_index(key, num_subdicts::Int) = hash((key, nothing)) % num_subdicts + 1

function Base.haskey(dict::ParallelDict, key)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        haskey(dict.subdicts[i], key)
    end
end

function Base.get(dict::ParallelDict, key)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        get(dict.subdicts[i], key)
    end
end

function Base.get(dict::ParallelDict, key, default)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        get(dict.subdicts[i], key, default)
    end
end

function Base.getindex(dict::ParallelDict, key)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        getindex(dict.subdicts[i], key)
    end
end

function Base.get!(dict::ParallelDict, key, default)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        get!(dict.subdicts[i], key, default)
    end
end

function Base.get!(f::Function, dict::ParallelDict, key)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        get!(f, dict.subdicts[i], key)
    end
end

function Base.setindex!(dict::ParallelDict, value, key)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        setindex!(dict.subdicts[i], value, key)
    end
end

function Base.getkey(dict::ParallelDict, key, default)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        getkey(dict.subdicts[i], key, default)
    end
end

function Base.delete!(dict::ParallelDict, key)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        delete!(dict.subdicts[i], key)
    end
end

function Base.pop!(dict::ParallelDict, key)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        pop!(dict.subdicts[i], key)
    end
end

function Base.length(dict::ParallelDict)
    len = 0
    for i in eachindex(dict.subdicts)
        len += lock(dict.locks[i]) do
            length(dict.subdicts[i])
        end
    end
    len
end

# ITERATION IS NOT THREAD SAFE!
function Base.iterate(dict::ParallelDict, state = ())
    if state !== ()
        y = iterate(Base.tail(state)...)
        y !== nothing && return (y[1], (state[1], state[2], y[2]))
    end
    x = (state === () ? iterate(dict.subdicts) : iterate(dict.subdicts, state[1]))
    x === nothing && return nothing
    y = iterate(x[1])
    while y === nothing
        x = iterate(dict.subdicts, x[2])
        x === nothing && return nothing
        y = iterate(x[1])
    end
    return y[1], (x[2], x[1], y[2])
end

Base.IteratorSize(dict::ParallelDict) = Base.HasLength()
Base.IteratorEltype(dict::ParallelDict) = Base.IteratorEltype(dict.subdicts[1])
Base.eltype(dict::ParallelDict) = Base.eltype(dict.subdicts[1])

mutable struct TimestepStateDictArray{T,N,M} <: AbstractArray{T,N}
    shape::NTuple{N,Int}
    dict::ParallelDict{Tuple{Int,Int},Array{T,M}}
    default_entry::Array{T,M}

    function TimestepStateDictArray{T,N,M}(
        default,
        shape...,
    ) where {T} where {N} where {M}
        @assert M == N - 2
        dict = ParallelDict{Tuple{Int,Int},Array{T,M}}()
        default_entry = fill(default, shape[3:end]...)
        new(shape, dict, default_entry)
    end
end

Base.size(arr::TimestepStateDictArray) = arr.shape

function Base.getindex(arr::TimestepStateDictArray, timestep, state, I...)
    @boundscheck checkbounds(arr, timestep, state, I...)
    key = (timestep, state)
    value = get(arr.dict, key, arr.default_entry)
    getindex(value, I...)
end

function Base.setindex!(arr::TimestepStateDictArray, v, timestep, state, I...)
    @boundscheck checkbounds(arr, timestep, state, I...)
    key = (timestep, state)
    value = get!(arr.dict, key) do
        copy(arr.default_entry)
    end
    setindex!(value, v, I...)
end

Base.maximum(arr::TimestepStateDictArray) = maximum(maximum(v) for (k, v) in arr.dict)


struct ValueIterationResults
    exploration_qs::AbstractArray{Float64,3}
    exploration_values::AbstractArray{Float64,2}
    optimal_qs::AbstractArray{Float64,3}
    optimal_values::AbstractArray{Float64,2}
    worst_qs::AbstractArray{Float64,3}
    worst_values::AbstractArray{Float64,2}
    visitable_states::AbstractVector{AbstractVector{Int}}
end

function value_iteration(
    transitions::Matrix{Int},
    rewards::Matrix{Float32},
    horizon::Int;
    exploration_policy::Union{Nothing,Array{Float32,3}} = nothing,
)
    num_states, num_actions = size(transitions)

    visitable_states = AbstractVector{Int}[]
    current_visitable_states = Set{Int}()
    push!(current_visitable_states, 1)
    for timestep in ProgressBar(1:horizon)
        next_visitable_states = Set{Int}()
        for state in current_visitable_states
            for action = 1:num_actions
                next_state = transitions[state, action] + 1
                push!(next_visitable_states, next_state)
            end
        end
        push!(visitable_states, collect(current_visitable_states))
        current_visitable_states = next_visitable_states
    end

    exploration_qs =
        TimestepStateDictArray{Float64,3,1}(NaN, horizon, num_states, num_actions)
    exploration_values = TimestepStateDictArray{Float64,2,0}(NaN, horizon, num_states)
    optimal_qs =
        TimestepStateDictArray{Float64,3,1}(NaN, horizon, num_states, num_actions)
    optimal_values = TimestepStateDictArray{Float64,2,0}(NaN, horizon, num_states)
    worst_qs =
        TimestepStateDictArray{Float64,3,1}(NaN, horizon, num_states, num_actions)
    worst_values = TimestepStateDictArray{Float64,2,0}(NaN, horizon, num_states)

    for timestep in ProgressBar(horizon:-1:1)
        Threads.@threads for state in visitable_states[timestep]
            for action = 1:num_actions
                next_state = transitions[state, action] + 1
                reward = rewards[state, action]
                if timestep < horizon #&& next_state != 0
                    exploration_qs[timestep, state, action] =
                        reward + exploration_values[timestep+1, next_state]
                    optimal_qs[timestep, state, action] =
                        reward + optimal_values[timestep+1, next_state]
                    worst_qs[timestep, state, action] =
                        reward + worst_values[timestep+1, next_state]
                else
                    exploration_qs[timestep, state, action] = reward
                    optimal_qs[timestep, state, action] = reward
                    worst_qs[timestep, state, action] = reward
                end
            end
            optimal_value = maximum(optimal_qs[timestep, state, :])
            worst_value = minimum(worst_qs[timestep, state, :])
            if exploration_policy === nothing
                exploration_value =
                    sum(exploration_qs[timestep, state, :]) / num_actions
            else
                exploration_value = 0
                for action = 1:num_actions
                    exploration_value +=
                        exploration_qs[timestep, state, action] *
                        exploration_policy[timestep, state, action]
                end
            end
            optimal_values[timestep, state] = optimal_value
            worst_values[timestep, state] = worst_value
            # Make sure exploration value is between worst and optimal values since
            # occasionally floating point error leads to that not being true.
            exploration_values[timestep, state] =
                min(optimal_value, max(worst_value, exploration_value))
            # Make sure we're not getting any NaNs, which would indicate a bug.
            @assert !isnan(exploration_values[timestep, state])
            @assert !isnan(optimal_values[timestep, state])
            @assert !isnan(worst_values[timestep, state])
        end
    end

    return ValueIterationResults(
        exploration_qs,
        exploration_values,
        optimal_qs,
        optimal_values,
        worst_qs,
        worst_values,
        visitable_states,
    )
end

function calculate_minimum_k(
    transitions::Matrix{Int},
    rewards::Matrix{Float32},
    horizon::Int;
    exploration_policy::Union{Nothing,Array{Float32,3}} = nothing,
    start_with_rewards::Bool = false,
)
    num_states, num_actions = size(transitions)
    vi = value_iteration(
        transitions,
        rewards,
        horizon,
        exploration_policy = exploration_policy,
    )
    if start_with_rewards
        current_qs =
            TimestepStateDictArray{Float64,3,1}(NaN, horizon, num_states, num_actions)
        for timestep = 1:horizon
            for state in vi.visitable_states[timestep]
                for action = 1:num_actions
                    current_qs[timestep, state, action] = rewards[state, action]
                end
            end
        end
    else
        current_qs = vi.exploration_qs
    end
    k = 1
    while true
        # Check if this value of k works.
        k_works = Threads.Atomic{Bool}(true)
        # states_can_be_visited = zeros(Bool, horizon, num_states)
        states_can_be_visited =
            TimestepStateDictArray{Bool,2,0}(false, horizon, num_states)
        states_can_be_visited[1, 1] = true
        timesteps_iter = ProgressBar(1:horizon)
        set_description(timesteps_iter, "Trying k = $(k)")
        for timestep in timesteps_iter
            Threads.@threads for state in vi.visitable_states[timestep]
                if states_can_be_visited[timestep, state]
                    max_q = -Inf64
                    for action = 1:num_actions
                        max_q = max(max_q, current_qs[timestep, state, action])
                    end
                    for action = 1:num_actions
                        if current_qs[timestep, state, action] >= max_q
                            # If we get here, it's possible to take this action.
                            if (
                                vi.optimal_qs[timestep, state, action] <
                                vi.optimal_values[timestep, state] - REWARD_PRECISION
                            )
                                k_works[] = false
                            end
                            next_state = transitions[state, action] + 1
                            if timestep < horizon
                                states_can_be_visited[timestep+1, next_state] = true
                            end
                        end
                    end
                end
            end
            if !k_works[]
                break
            end
        end

        if k_works[]
            return k
        end

        # Run a Bellman backup.
        for timestep in ProgressBar(1:horizon-1)
            Threads.@threads for state in vi.visitable_states[timestep]
                for action = 1:num_actions
                    next_state = transitions[state, action] + 1
                    max_next_q = -Inf64
                    for action = 1:num_actions
                        next_q = current_qs[timestep+1, next_state, action]
                        max_next_q = max(max_next_q, next_q)
                    end
                    current_qs[timestep, state, action] =
                        rewards[state, action] + max_next_q
                end
            end
        end
        k += 1
    end
end


mutable struct EffectiveHorizonResults
    ks::Vector{Int32}
    ms::Vector{BigInt}
    vars::Vector{Float64}
    gaps::Vector{Float64}
    effective_horizon::Float64
end

const REWARD_PRECISION = 1e-4

function compute_simple_effective_horizon(
    transitions::Matrix{Int},
    rewards::Matrix{Float32},
    horizon::Int;
    exploration_policy::Union{Nothing,Array{Float32,3}} = nothing,
)
    num_states, num_actions = size(transitions)
    vi = value_iteration(
        transitions,
        rewards,
        horizon,
        exploration_policy = exploration_policy,
    )

    var_bounds =
        TimestepStateDictArray{Float64,3,1}(NaN, horizon, num_states, num_actions)
    for timestep in ProgressBar(1:horizon)
        Threads.@threads for state in vi.visitable_states[timestep]
            for action = 1:num_actions
                q = vi.exploration_qs[timestep, state, action]
                worst_q = vi.worst_qs[timestep, state, action]
                optimal_q = vi.optimal_qs[timestep, state, action]
                var_bound = (q - worst_q) * (optimal_q - worst_q)
                var_bounds[timestep, state, action] = var_bound
            end
        end
    end

    current_qs = vi.exploration_qs

    results = EffectiveHorizonResults(
        Vector{Int32}(undef, 0),
        Vector{BigInt}(undef, 0),
        Vector{Float64}(undef, 0),
        Vector{Float64}(undef, 0),
        horizon,
    )

    k = 1
    while k < results.effective_horizon
        k_works = Threads.Atomic{Bool}(true)
        state_ms = TimestepStateDictArray{BigInt,2,0}(0, horizon, num_states)
        state_vars = TimestepStateDictArray{Float64,2,0}(0, horizon, num_states)
        state_gaps = TimestepStateDictArray{Float64,2,0}(0, horizon, num_states)
        states_can_be_visited =
            TimestepStateDictArray{Bool,2,0}(false, horizon, num_states)
        states_can_be_visited[1, 1] = true
        timesteps_iter = ProgressBar(1:horizon)
        set_description(timesteps_iter, "Trying k = $(k)")
        for timestep in timesteps_iter
            Threads.@threads for state in vi.visitable_states[timestep]
                if states_can_be_visited[timestep, state]
                    max_q = -Inf64
                    max_suboptimal_q = -Inf64
                    max_var = 0
                    for action = 1:num_actions
                        q = current_qs[timestep, state, action]
                        max_q = max(max_q, q)
                        if (
                            vi.optimal_qs[timestep, state, action] <
                            vi.optimal_values[timestep, state] - REWARD_PRECISION
                        )
                            max_suboptimal_q = max(max_suboptimal_q, q)
                        end
                        max_var = max(max_var, var_bounds[timestep, state, action])
                    end

                    if max_q == max_suboptimal_q
                        k_works[] = false
                    else
                        gap = max_q - max_suboptimal_q
                        state_gaps[timestep, state] = gap
                        state_vars[timestep, state] = max_var
                        m = ceil(
                            BigInt,
                            16 * max_var / (gap^2) *
                            Base.log(2 * horizon * Float64(num_actions)^k),
                        )
                        m = max(1, m)
                        state_ms[timestep, state] = m
                    end

                    for action = 1:num_actions
                        if current_qs[timestep, state, action] > max_suboptimal_q
                            next_state = transitions[state, action] + 1
                            if timestep < horizon
                                states_can_be_visited[timestep+1, next_state] = true
                            end
                        end
                    end
                end
            end
            if !k_works[]
                break
            end
        end

        if k_works[]
            push!(results.ks, k)
            highest_m, timestep_state = findmax(state_ms)
            timestep, state = Tuple(timestep_state)
            push!(results.ms, highest_m)
            H_k = k + Base.log(num_actions, highest_m)
            # println("H_$(k) = $(H_k)")
            push!(results.gaps, state_gaps[timestep, state])
            push!(results.vars, state_vars[timestep, state])
            results.effective_horizon = min(results.effective_horizon, H_k)
        end

        # Run a Bellman backup.
        for timestep in ProgressBar(1:horizon-1)
            Threads.@threads for state in vi.visitable_states[timestep]
                for action = 1:num_actions
                    next_state = transitions[state, action] + 1
                    max_next_q = -Inf64
                    max_next_var_bound = 0
                    for action = 1:num_actions
                        next_q = current_qs[timestep+1, next_state, action]
                        max_next_q = max(max_next_q, next_q)
                        next_var_bound = var_bounds[timestep+1, next_state, action]
                        max_next_var_bound = max(max_next_var_bound, next_var_bound)
                    end
                    current_qs[timestep, state, action] =
                        rewards[state, action] + max_next_q
                    var_bounds[timestep, state, action] = max_next_var_bound
                end
            end
        end
        k += 1
    end

    results
end


# init runthrough
num_states, num_actions = size(transitions)
results = calculate_minimum_k(transitions, rewards, horizon)
