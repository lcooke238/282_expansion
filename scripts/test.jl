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
        println(current_visitable_states)
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
                if timestep < horizon
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



# init runthrough
transitions = [123 123 70 28 70 28 ;
66 66 66 242 66 242 ;
138 138 138 114 138 114 ;
41 41 41 20 41 20 ;
209 209 209 108 209 108 ;
140 140 170 239 170 239 ;
54 54 185 122 185 122 ;
143 143 24 143 24 143 ;
19 19 19 173 19 173 ;
67 67 67 202 67 202 ;
16 16 45 71 45 71 ;
37 37 65 77 65 77 ;
53 53 159 141 159 141 ;
86 86 219 217 219 217 ;
184 184 2 63 2 63 ;
150 150 150 204 150 204 ;
215 215 167 50 167 50 ;
-1 -1 -1 -1 -1 -1 ;
72 72 178 145 178 145 ;
41 41 41 20 41 20 ;
-1 -1 -1 -1 -1 -1 ;
157 157 94 157 94 157 ;
238 238 66 82 66 82 ;
75 75 160 157 160 157 ;
-1 -1 -1 -1 -1 -1 ;
57 57 19 225 19 225 ;
-1 -1 -1 -1 -1 -1 ;
112 112 198 112 198 112 ;
49 49 11 49 11 49 ;
161 161 235 7 235 7 ;
101 101 168 101 168 101 ;
35 35 35 39 35 39 ;
72 72 178 145 178 145 ;
86 86 219 217 219 217 ;
170 170 170 32 170 32 ;
-1 -1 -1 -1 -1 -1 ;
143 143 24 143 24 143 ;
16 16 45 71 45 71 ;
44 44 56 74 56 74 ;
-1 -1 -1 -1 -1 -1 ;
154 154 148 27 148 27 ;
-1 -1 -1 -1 -1 -1 ;
97 97 189 201 189 201 ;
98 98 81 153 81 153 ;
18 18 188 239 188 239 ;
167 167 167 234 167 234 ;
91 91 91 29 91 29 ;
183 183 183 38 183 38 ;
64 64 180 64 180 64 ;
166 166 10 166 10 166 ;
141 141 194 141 194 141 ;
42 244 3 199 3 199 ;
145 145 195 145 195 145 ;
142 142 183 122 183 122 ;
44 44 56 74 56 74 ;
137 137 68 36 68 36 ;
170 170 170 32 170 32 ;
41 41 41 175 41 175 ;
31 31 208 236 208 236 ;
211 211 34 176 34 176 ;
191 191 191 240 191 240 ;
132 132 227 111 227 111 ;
220 220 159 85 159 85 ;
83 83 216 83 216 83 ;
217 217 55 217 55 217 ;
45 45 45 165 45 165 ;
138 138 138 114 138 114 ;
-1 -1 -1 -1 -1 -1 ;
67 67 67 152 67 152 ;
104 104 8 250 8 250 ;
252 252 252 214 252 214 ;
88 88 62 88 62 88 ;
75 75 160 157 160 157 ;
227 227 227 33 227 33 ;
239 239 187 239 187 239 ;
230 230 226 112 226 112 ;
208 208 208 99 208 99 ;
147 147 169 147 169 147 ;
134 134 150 223 150 223 ;
22 22 150 171 150 171 ;
137 137 68 36 68 36 ;
-1 -1 -1 -1 -1 -1 ;
83 83 216 83 216 83 ;
250 250 25 250 25 250 ;
40 40 248 21 248 21 ;
122 122 59 122 59 122 ;
161 161 235 7 235 7 ;
4 4 4 131 4 131 ;
141 141 194 141 194 141 ;
15 15 15 113 15 113 ;
221 221 89 112 89 112 ;
68 68 68 207 68 207 ;
115 115 138 105 138 105 ;
57 57 19 225 19 225 ;
243 243 89 218 89 218 ;
-1 -1 -1 -1 -1 -1 ;
-1 -1 -1 -1 -1 -1 ;
-1 -1 -1 -1 -1 -1 ;
-1 -1 -1 -1 -1 -1 ;
98 98 81 153 81 153 ;
193 193 120 250 120 250 ;
245 245 174 245 174 245 ;
43 43 136 237 136 237 ;
84 84 170 52 170 52 ;
110 110 19 199 19 199 ;
250 250 25 250 25 250 ;
107 107 228 101 228 101 ;
102 102 76 245 76 245 ;
43 43 136 237 136 237 ;
-1 -1 -1 -1 -1 -1 ;
109 109 41 229 41 229 ;
217 217 55 217 55 217 ;
210 210 222 210 222 210 ;
14 14 1 223 1 223 ;
193 193 120 250 120 250 ;
93 93 8 233 8 233 ;
167 167 167 234 167 234 ;
30 30 128 30 128 30 ;
-1 -1 -1 -1 -1 -1 ;
90 90 148 157 148 157 ;
19 19 19 173 19 173 ;
163 163 45 147 45 147 ;
74 74 103 74 103 74 ;
190 190 252 49 252 49 ;
4 4 4 131 4 131 ;
-1 -1 -1 -1 -1 -1 ;
241 251 196 210 196 210 ;
124 124 124 139 124 139 ;
164 164 4 200 4 200 ;
-1 -1 -1 -1 -1 -1 ;
48 48 61 48 61 48 ;
102 102 76 245 76 245 ;
80 80 91 158 91 158 ;
179 179 47 141 47 141 ;
205 205 66 63 66 63 ;
149 149 73 48 73 48 ;
35 35 35 129 35 129 ;
67 67 67 202 67 202 ;
8 8 8 51 8 51 ;
107 107 228 101 228 101 ;
119 119 248 145 248 145 ;
122 122 59 122 59 122 ;
5 5 34 74 34 74 ;
-1 -1 -1 -1 -1 -1 ;
74 74 103 74 103 74 ;
157 157 94 157 94 157 ;
106 106 87 30 87 30 ;
88 88 62 88 62 88 ;
89 89 89 126 89 126 ;
13 13 46 64 46 64 ;
66 66 66 242 66 242 ;
-1 -1 -1 -1 -1 -1 ;
-1 -1 -1 -1 -1 -1 ;
-1 -1 -1 -1 -1 -1 ;
243 243 89 218 89 218 ;
-1 -1 -1 -1 -1 -1 ;
45 45 45 165 45 165 ;
112 112 198 112 198 112 ;
7 7 9 7 9 7 ;
183 183 183 38 183 38 ;
89 89 89 126 89 126 ;
162 162 96 143 96 143 ;
-1 -1 -1 -1 -1 -1 ;
12 12 167 88 167 88 ;
58 58 209 232 209 232 ;
133 133 212 88 212 88 ;
147 147 169 147 169 147 ;
159 159 159 6 159 6 ;
58 58 209 232 209 232 ;
215 215 167 50 167 50 ;
248 248 248 23 248 23 ;
63 63 92 63 92 63 ;
18 18 188 239 188 239 ;
151 151 26 229 26 229 ;
31 31 208 236 208 236 ;
-1 -1 -1 -1 -1 -1 ;
239 239 187 239 187 239 ;
133 133 212 88 212 88 ;
148 148 148 224 148 224 ;
54 54 185 122 185 122 ;
80 80 91 158 91 158 ;
8 8 8 51 8 51 ;
211 211 34 176 34 176 ;
34 34 34 172 34 172 ;
100 100 181 83 181 83 ;
34 34 34 172 34 172 ;
-1 -1 -1 -1 -1 -1 ;
40 40 248 21 248 21 ;
248 248 248 23 248 23 ;
-1 -1 -1 -1 -1 -1 ;
121 121 65 166 65 166 ;
227 227 227 33 227 33 ;
223 223 213 223 213 223 ;
42 244 3 199 3 199 ;
182 182 183 144 183 144 ;
154 154 148 27 148 27 ;
150 150 150 204 150 204 ;
125 125 125 203 125 203 ;
79 79 15 192 15 192 ;
229 229 155 229 155 229 ;
245 245 174 245 174 245 ;
-1 -1 -1 -1 -1 -1 ;
-1 -1 -1 -1 -1 -1 ;
-1 -1 -1 -1 -1 -1 ;
184 184 2 63 2 63 ;
69 69 138 83 138 83 ;
177 177 116 147 116 147 ;
162 162 96 143 96 143 ;
35 35 35 129 35 129 ;
208 208 208 99 208 99 ;
223 223 213 223 213 223 ;
84 84 170 52 170 52 ;
159 159 159 6 159 6 ;
238 238 66 82 66 82 ;
206 206 156 166 156 166 ;
220 220 159 85 159 85 ;
93 93 8 233 8 233 ;
7 7 9 7 9 7 ;
210 210 222 210 222 210 ;
68 68 68 207 68 207 ;
182 182 183 144 183 144 ;
78 78 15 210 15 210 ;
22 22 150 171 150 171 ;
63 63 92 63 92 63 ;
230 230 226 112 226 112 ;
229 229 155 229 155 229 ;
15 15 15 113 15 113 ;
91 91 91 29 91 29 ;
209 209 209 108 209 108 ;
-1 -1 -1 -1 -1 -1 ;
241 251 196 210 196 210 ;
35 35 35 39 35 39 ;
237 237 231 237 231 237 ;
199 199 197 199 247 199 ;
179 179 47 141 47 141 ;
67 67 67 152 67 152 ;
153 153 246 153 246 153 ;
153 153 246 153 246 153 ;
115 115 138 105 138 105 ;
145 145 195 145 195 145 ;
13 13 46 64 46 64 ;
135 135 60 130 60 130 ;
100 100 181 83 181 83 ;
79 79 15 192 15 192 ;
95 95 249 17 249 17 ;
237 237 231 237 231 237 ;
-1 -1 -1 -1 -1 -1 ;
118 118 118 186 118 186 ;
148 148 148 224 148 224 ;
-1 -1 -1 -1 -1 -1 ;
199 199 197 199 247 199 ;
146 146 127 117 127 117 ;
65 65 65 253 65 253 ;
177 177 116 147 116 147]
rewards = Float32[0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
1.0 1.0 1.0 1.0 1.0 1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
1.0 1.0 1.0 1.0 1.0 1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 0.0 -1.0 0.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
-1.0 -1.0 0.0 -1.0 0.0 -1.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0]
horizon = 254
num_states, num_actions = size(transitions)
vi = value_iteration(
        transitions,
        rewards,
        horizon,
        exploration_policy = nothing,
    )
