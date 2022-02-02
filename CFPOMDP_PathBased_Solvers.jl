#Ben Kraske
#Define solvers and policy structs - Updated 11/4

using POMDPs
using BeliefUpdaters


##Previous Observation "Solver" and Updater
struct PrevObsUpdater{P<:POMDP} <: Updater
    pomdp::P
end

BeliefUpdaters.initialize_belief(u::PrevObsUpdater, d::Any) = initialize_belief(DiscreteUpdater(u.pomdp),d)

function BeliefUpdaters.update(u::PrevObsUpdater, b, action, obs)
    if isa(obs,cf_ot)
        if obs.f == :faulty
            f_new = 1
        elseif obs.f == :ok
            f_new = 0
        end
        bn = cf_st(obs.x,f_new)
    end
    return initialize_belief(DiscreteUpdater(u.pomdp),Deterministic(bn))
end

struct PrevObsSolver <: Solver
    vip::ValueIterationPolicy
end
struct PrevObsPolicy{P<:POMDP} <: Policy
    pomdp::P
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
    elseif isa(b,DiscreteBelief{CF_POMDP, cf_st})
        for (state,value) in weighted_iterator(bel0)
            if value == 1
                bn = state
            end
        end
    end
    return action(p.vip,bn) #otherwise, do optimal aggresive action
end

##Maximum Likelihood "Solver"
struct MaxBSolver <: Solver
    vip::ValueIterationPolicy
end
struct MaxBPolicy{P<:POMDP} <: Policy
    pomdp::P
    vip::ValueIterationPolicy
end

function POMDPs.solve(cs::MaxBSolver, p::POMDP)
    # vip = solve(cs.vis,UnderlyingMDP(p))
    return MaxBPolicy(p,cs.vip)
end

function POMDPs.action(p::MaxBPolicy, b)
    # print(b)
    if isa(b,DiscreteBelief)
        ind = argmax(b.b)
        p0 = b.state_list[ind]
    end
    return action(p.vip,p0) #otherwise, do optimal aggresive action
end

#Updater
# struct MaxBeliefUpdater{P<:POMDP} <: Updater
#     pomdp::P
# end
#
# function BeliefUpdaters.initialize_belief(u::MaxBeliefUpdater, d::Any)
#     return initialize_belief(DiscreteUpdater(u.pomdp),d)
# end

# function BeliefUpdaters.update(u::MaxBeliefUpdater, b::DiscreteBelief, action, obs)
#     bp = update(DiscreteUpdater(u.pomdp),b,action,obs)
#     ind = argmax(b.b)
#     bn = b.state_list[ind]
#     return initialize_belief(DiscreteUpdater(u.pomdp),Deterministic(bn))
# end
