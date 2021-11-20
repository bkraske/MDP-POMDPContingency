#Ben Kraske, 10/11/21 - Updated 11/4
#Implement General Component Failure POMDP - assumes some path is given
using POMDPs

#Define State, Observation, Action Types
# const cf_st = Tuple{Int64, Int64, Int64}
struct cf_st
    x::Int64
    f::Int64
end
struct cf_ot
    x::Int64
    f::Symbol
end
const cf_at = Symbol

##Define POMDP Struct
struct CF_POMDP <: POMDP{cf_st, cf_at, cf_ot} # POMDP{State, Action, Observation}
    ###State List and Related Parameters##################################
    all_s_actual::Array{cf_st}
    all_o_actual::Array{cf_ot}
    n_x::StepRange{Int64, Int64}
    n_f::StepRange{Int64, Int64}
    state_ind::Dict{cf_st, Int64}
    act_ind::Dict{cf_at, Int64}
    obs_ind::Dict{cf_ot, Int64}
    a_list::Vector{cf_at}
    disc::Float64
    move::Dict{cf_at, Int64}
    ###Reward States, Rewards, Terminal State##############################
    rew_state::Int64
    land_states::Tuple{Int64,Int64,Int64,Int64,Int64}
    max_rew::Float64
    move_rew::Float64
    bad_rew::Float64
    term_fail::Int64
    term_state_bad::Int64
    term_state_good::Int64
    #Other Parameters #####################################################
    i_x::Int64
    i_f::Int64
    #Probabilities
    pnorm1::Float64
    pnorm2::Float64
    pobs1::Float64
    pobs2::Float64
end

##Function to initialize POMDP
function init_cf(;
    rew_state::Int64 = 20,
    max_rew::Float64 = 1.0,#200
    move_rew::Float64 = 0.0, #-0.05
    bad_rew::Float64 = -1.0, #Set back to -10 eventually
    term_fail::Int64 = 2,
    term_state_bad::Int64 = -1,
    term_state_good::Int64 = -2,
    #Other Parameters
    disc::Float64 = 0.95,
    a_list::Vector{Symbol} = [:f,:b,:l],
    ## State Initial Variables #################################################
    i_x::Int64 = 0,
    i_f::Int64 = 0,
    pnorm1::Float64 = 0.99,
    pnorm2::Float64 = 0.99,
    pobs1::Float64 = 0.95,
    pobs2::Float64 = 0.95,
    segment_total = 20,
    land_states::Tuple{Int64,Int64,Int64} = (5,10,15)
    )

    #State List
    n_x = -2:1:segment_total
    n_f = 0:1:2 #Failure states
    n_o = (:faulty,:ok)

    all_s_actual = collect(cf_st(x,f) for x in n_x, f in n_f) #all states
    all_o_actual = collect(cf_ot(x,o) for x in n_x, o in n_o) #all observations

    #Dictionary for input based on action
    move = Dict(:f=>1,:b=>-1,:l=>1)

    state_ind =  Dict(s=>i for (i,s) in enumerate(all_s_actual)) #Better way of doing this???
    act_ind = Dict(s=>i for (i,s) in enumerate(a_list)) #Better way of doing this???
    obs_ind =  Dict(s=>i for (i,s) in enumerate(all_o_actual)) #Better way of doing this???

    land_states = (i_x,5,10,15,rew_state)
    return CF_POMDP(
    ###State List and Related Parameters##################################
    all_s_actual,
    all_o_actual,
    n_x,
    n_f,
    state_ind,
    act_ind,
    obs_ind,
    a_list,
    disc,
    move,
    ###Reward States, Rewards, Terminal State##############################
    rew_state,
    land_states,
    max_rew,
    move_rew,
    bad_rew,
    term_fail,
    term_state_bad,
    term_state_good,
    #Other Parameters #####################################################
    i_x,
    i_f,
    #Probabilities
    pnorm1,
    pnorm2,
    pobs1,
    pobs2
    )
end

##POMDP Function Defitions

POMDPs.states(m::CF_POMDP) = m.all_s_actual
POMDPs.stateindex(m::CF_POMDP, s::cf_st) = m.state_ind[s]

POMDPs.actions(m::CF_POMDP) = m.a_list
POMDPs.actionindex(m::CF_POMDP, a::cf_at) = m.act_ind[a]
POMDPs.actions(m::CF_POMDP, state::cf_st) = POMDPs.actions(m)

POMDPs.observations(m::CF_POMDP) = m.all_o_actual
POMDPs.obsindex(m::CF_POMDP, o::cf_ot) = m.obs_ind[o]
# POMDPs.obsindex(m::CF_POMDP, o::Tuple{Float64,Float64,Float64, Int64}) = m.obs_ind[o]
POMDPs.observations(m::CF_POMDP, state::cf_st) = POMDPs.observations(m)
# POMDPs.obstype(m::CF_POMDP) = Tuple{Float64, Float64, Float64, Symbol}
POMDPs.discount(m::CF_POMDP) = m.disc
POMDPs.initialstate(m::CF_POMDP) = Deterministic(cf_st(m.i_x,m.i_f))

function POMDPs.isterminal(m::CF_POMDP, s::cf_st)
    if s.x == m.rew_state #Reward State
        return true
    elseif s.x == m.term_state_bad || s.x == m.term_state_good #Could just make this check for negative if true state is important
        return true
    elseif s.f == m.term_fail
        return true
    else
        return false
    end
end
function POMDPs.isterminal(m::CF_POMDP, s::Any)
    @warn "Terminal Check: Not a $cf_st, is a $(typeof(s))"
end

function POMDPs.transition(m::CF_POMDP,s::cf_st,a::cf_at)

    #Parse Function inputs
    act = m.move[a]

    #Update transit segment and intent to land/replan (r)
    if a == :b || a == :f
        xp = s.x + act
    elseif a == :l && s.x ∉ m.land_states
        xp = m.term_state_bad #Could just make this check for negative if true state is important
    elseif a == :l && s.x ∈ m.land_states
        xp = m.term_state_good ###UPDATE THESE ABOVE
    end
    
    #Bounce when attempting to exit grid################
    if xp in m.n_x
        xp = xp
    elseif xp ∉ m.n_x
        # println("Out X $xp")
        xp = s.x
    end

    ############################################################################
    #Probability of Failure
    if s.f == 0 #probability of transition to intermediate failure
        prob = [m.pnorm1,1-m.pnorm1]
        s_vec = [cf_st(xp, 0),cf_st(xp, 1)]
    elseif s.f == 1 #probability of transition to total failure
        prob = [m.pnorm2,1-m.pnorm2]
        s_vec = [cf_st(xp, 1),cf_st(xp, 2)]
    elseif s.f == 2 #cover transition from total to total for POMDPs package
        prob = [1.0]
        s_vec = [cf_st(xp, 2)]
    end

    @assert abs(sum(prob)-1)<0.0000001  "Probabilities do not sum to 1 in transit() !"

    ht = SparseCat(s_vec, prob)
    return ht
end

function POMDPs.observation(m::CF_POMDP,sp::cf_st)

    if sp.f == 0
        prob = [m.pobs1,1-m.pobs1]
        s_vec = [cf_ot(sp.x, :ok),cf_ot(sp.x, :faulty)]
    elseif sp.f == 1
        prob = [1-m.pobs2,m.pobs2]
        s_vec = [cf_ot(sp.x, :ok),cf_ot(sp.x, :faulty)]
    elseif sp.f == 2
        prob = [1.0]
        s_vec = [cf_ot(sp.x, :faulty)]
    end

    @assert abs(sum(prob)-1)<0.0000001  "Probabilities do not sum to 1 in observation() !"

    ob = SparseCat(s_vec, prob)
    return ob
end

function POMDPs.reward(m::CF_POMDP,s::cf_st, a::cf_at, sp::cf_st)

    #Reward for Landing Locations and Speed
    if sp.x == m.rew_state
        rd = m.max_rew
    else
        rd = 0.0
    end

    #Penalty for landing not at emergency site
    if s.x ∉ m.land_states && sp.x == -1 && a == :l
        rm = m.bad_rew
    else
        rm = 0.0
    end

    #Penalty for Faults
    if sp.f == 2
        rf = m.bad_rew
    # elseif fp == 1
    #     rf = 0.05*m.bad_rew
    else
        rf = 0.0
    end

    r_tot = rd + rf + rm

    return r_tot
end
