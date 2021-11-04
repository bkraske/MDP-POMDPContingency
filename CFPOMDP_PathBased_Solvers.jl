#Ben Kraske
#Define solvers and policy structs - Updated 11/4

using POMDPs

##Previous Observation "Solver" for use with PreviousObservationUpdater
struct PrevObsSolver <: Solver
    vip::ValueIterationPolicy
end
struct PrevObsPolicy{P<:POMDP} <: Policy
    m::P
    vip::ValueIterationPolicy
end

function POMDPs.solve(cs::PrevObsSolver, p::POMDP)
    # vip = solve(cs.vis,UnderlyingMDP(p))
    return PrevObsPolicy(p,cs.vip)
end

function POMDPs.action(p::PrevObsPolicy, b)
    if isa(b,Deterministic{cf_st})
        bn = support(b)[1]
    elseif isa(b,cf_ot)
        if b.f == :faulty
            f_new = 1
        elseif b.f == :ok
            f_new = 0
        end
        bn = cf_st(b.x,f_new)
    end
    return action(p.vip,bn) #otherwise, do optimal aggresive action
end

##Maximum Likelihood "Solver"
struct MaxBSolver <: Solver
    vip::ValueIterationPolicy
end
struct MaxBPolicy{P<:POMDP} <: Policy
    m::P
    vip::ValueIterationPolicy
end

function POMDPs.solve(cs::MaxBSolver, p::POMDP)
    # vip = solve(cs.vis,UnderlyingMDP(p))
    return MaxBPolicy(p,cs.vip)
end

function POMDPs.action(p::MaxBPolicy, b)
    if isa(b,DiscreteBelief)
        ind = argmax(b.b)
        p0 = b.state_list[ind]
    end
    return action(p.vip,p0) #otherwise, do optimal aggresive action
end
