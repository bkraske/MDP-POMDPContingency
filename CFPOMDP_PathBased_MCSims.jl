#Ben Kraske, 8/5/21 - Updated 11/4
#Component Failure POMDP - Pathbased MC Runs
#Packages
using POMDPs,POMDPModelTools, POMDPSimulators, POMDPPolicies, BeliefUpdaters
using QMDP, DiscreteValueIteration, SARSOP
using Statistics, Test, DataFrames, Plots, CSV, Dates, Random
using POMDPPolicyGraphs

include("CFPOMDP_PathBased.jl")
include("CFPOMDP_PathBased_Solvers.jl")

## Sim for Plotting Each Step
function plotPOMDP(cf,planner,up)
    pos_hist_x = [cf.i_x]
    # pos_hist_r = [cf.i_r]
    site_x = [0]
    site_y = [0.0]
    for i in -1*ones(length(cf.land_states)-2)
        push!(site_y, i)
    end
    for i in cf.land_states[2:4]
        push!(site_x,i)
    end
    push!(site_x,cf.rew_state)
    push!(site_y,0.0)

    for (s, a, o, r, b, sp) in stepthrough(cf, planner, up, "s,a,o,r,b,sp")# stepthrough(cf, planner, "s,a,o,r,b,sp") #stepthrough(cf, qmdp_pol, BootstrapFilter(cf,50), "s,a,o,r,b,sp")#(cf, planner, "s,a,o,r,b,sp")#(cf, qmdp_pol, "s,a,o,r,b,sp")
        println("=====") # =======================================")
        #println("belief $(unique(particles(b)))")
        println("belief $(collect(unique((weighted_iterator(b)))))")
        # println("belief $b")
        println("in state $s")
        println("took action $a")
        println("received observation $o and reward $r")
        println("next state is $sp")
        println("============================================")
        push!(pos_hist_x,sp.x)
        # push!(pos_hist_r,sp.r)
        # plot(site_x, site_y,seriestype = :scatter;grid=true, label = "Landing Sites") #, xlims = [first(cf.n_x),last(cf.n_x)], ylims = [first(cf.n_h),last(cf.n_h)])
        # ylims!(-1,1)
        # display(plot!(pos_hist_x,pos_hist_r;grid=true, label = "UAV"))
    end
end


function plotMDP(cf,pol)
    pos_hist_x = [cf.i_x]
    # pos_hist_r = [cf.i_r]
    site_x = [0]
    site_y = [0.0]
    for i in -1*ones(length(cf.land_states)-2)
        push!(site_y, i)
    end
    for i in cf.land_states[2:4]
        push!(site_x,i)
    end
    push!(site_x,cf.rew_state)
    push!(site_y,0.0)

    for (s, a, r, sp) in stepthrough(UnderlyingMDP(cf), pol, support(initialstate(cf))[1], "s,a,r,sp")
        println("=====") # =======================================")
        println("in state $s")
        println("took action $a")
        println("received reward $r")
        println("next state is $sp")
        println("============================================")
        push!(pos_hist_x,sp.x)
        # push!(pos_hist_r,sp.r)
        plot(site_x, site_y,seriestype = :scatter;grid=true, label = "Landing Sites") #, xlims = [first(cf.n_x),last(cf.n_x)], ylims = [first(cf.n_h),last(cf.n_h)])
        ylims!(-1,1)
        # display(plot!(pos_hist_x,pos_hist_r;grid=true, label = "UAV"))
    end
end


##Stats Functions for runs
function sem(result)
    stand_dev = std(result.disc_rew)
    standard_error = stand_dev/sqrt(length(result.final_state))
    sample_mean = mean(result.disc_rew)
    return (sample_mean,standard_error)::Tuple{Float64,Float64}
end

function count_complete(m, result)
    target_count = []
    for i in 1:length(result.final_state)
        if result.final_state[i].x==m.rew_state
            push!(target_count,1)
        else
            push!(target_count,0)
        end
    end
    std_err = std(target_count)/sqrt(length(target_count))
    return (mean(target_count),std_err)::Tuple{Float64,Float64}
end

function count_fail(m, result)
    fail_count = []
    for i in 1:length(result.final_state)
        if result.final_state[i].f==m.term_fail || result.final_state[i].x == m.term_state_bad
            push!(fail_count,1)
        elseif result.final_state[i].f==1 || result.final_state[i].f ==0
            push!(fail_count,0)
        end
    end
    std_err = std(fail_count)/sqrt(length(fail_count))
    return (mean(fail_count),std_err)::Tuple{Float64,Float64}
end

function count_steps(m, updater, s0) #Count number of steps for ParticleBelief to reach some threshold
    b = initialize_belief(updater,s0)
    x = support(s0)[1].x
    c = 0
    pf1 = 0
    while (pf1 < 0.5)&&(x < 20)
        x += 1
        b = update(updater, b, :f, cf_ot(x,0,:f,:faulty))
        c += 1
        pf1 = 0
        for (s,i) in weighted_iterator(b)
            if s.f == 1
                pf1 += i
            end
            # println(pf1)
        end
        # println(pf1)
    end
    return c::Int64
end

function count_steps_bel(m) #Same as above for DiscreteBelief
    b = [1.0,0.0,0.0]
    bn = [0.0,0.0,0.0]
    c = 0
    while b[2] < 0.5 && c<500
        bn[1] = (1-m.pobs1)*m.pnorm1*b[1]
        bn[2] = m.pobs2*((1-m.pnorm1)*b[1]+m.pnorm2*b[2])
        bn[3] = 1.0*((1-m.pnorm2)*b[2])
        b[1] = bn[1]/sum(bn)
        b[2] = bn[2]/sum(bn)
        b[3] = bn[3]/sum(bn)
        c+=1

    end
    return c::Int64
end

function withintol(sim_mean,sim_sem,exact)
    within_tol2 = (sim_mean+(sim_sem)>=exact)&&(sim_mean-(sim_sem)<=exact)
    return within_tol2
end
##Generate MC Data
function run_MC(m, n_runs, solvers) #Run all policies on given problem definition #Change "solver" to "policy"
    results_list = []
    reward_mean = []
    reward_std = []
    failed = []
    failed_std = []
    target = []
    target_std = []
    names = []
    # steps = count_steps_bel(m)
    steps = 0
    for (solver,name,updater,graphtype,precision) in solvers
        # println("Solver $name")
        sims = [Sim(m,solver,updater) for i in 1:n_runs]
        # println(updater)
        result = run(sims) do sim, hist
            return [:steps=>n_steps(hist), :disc_rew=>discounted_reward(hist),:final_state=> last(state_hist(hist))]
        end
        # println(result)
        push!(names,name)
        push!(results_list, result)
        push!(reward_mean, sem(result)[1])
        push!(reward_std, sem(result)[2])
        push!(failed, count_fail(m,result)[1])
        push!(failed_std, count_fail(m,result)[2])
        push!(target, count_complete(m,result)[1])
        push!(target_std, count_complete(m,result)[2])

        # [exact_reward,exact_completed,exact_failed,exact_steps] = run_single_PG(m, solver, updater, graphtype, precision=3)
        exact_reward,exact_completed,exact_failed = run_single_PG2(m, solver, updater, graphtype, precision;disc = [0.95,0.99995,0.99995])
        push!(names,name*"-E")
        push!(reward_std, withintol(last(reward_mean),last(reward_std),exact_reward))
        push!(reward_mean, exact_reward)
        push!(failed_std, withintol(last(failed),last(failed_std),exact_failed))
        push!(failed, exact_failed)
        push!(target_std, withintol(last(target),last(target_std),exact_completed))
        push!(target, exact_completed)
    end
    return theframe = DataFrame([:Policy=>names,:Mean_Disc_Rew=>reward_mean, :SE_Disc_Rew=>reward_std, :Fract_Goal=>target, :SE_Goal=>target_std, :Fract_Fail=>failed, :SE_Fail=>failed_std,
               :pnorm1 => m.pnorm1, :pnorm2 => m.pnorm2, :pobs1 => m.pobs1, :pobs2 => m.pobs2, :rew => m.bad_rew, :stps_up => steps])
end

function run_MC_mdp(m, n_runs, solvers) #Above for MDP
    m_mdp = UnderlyingMDP(m)
    results_list = []
    reward_mean = []
    reward_std = []
    failed = []
    failed_std = []
    target = []
    target_std = []
    names = []
    # steps = count_steps_bel(m)
    steps = 0
    for (solver,name) in solvers
        # println("Solver $name")
        sims = [Sim(m_mdp,solver) for i in 1:n_runs]
        result = run(sims) do sim, hist
            return [:steps=>n_steps(hist), :disc_rew=>discounted_reward(hist),:final_state=> last(state_hist(hist))]
        end
        # println(result)
        push!(names,name)
        push!(results_list, result)
        push!(reward_mean, sem(result)[1])
        push!(reward_std, sem(result)[2])
        push!(failed, count_fail(m,result)[1])
        push!(failed_std, count_fail(m,result)[2])
        push!(target, count_complete(m,result)[1])
        push!(target_std, count_complete(m,result)[2])
    end
    return theframe = DataFrame([:Policy=>names,:Mean_Disc_Rew=>reward_mean, :SE_Disc_Rew=>reward_std, :Fract_Goal=>target, :SE_Goal=>target_std, :Fract_Fail=>failed, :SE_Fail=>failed_std,
               :pnorm1 => m.pnorm1, :pnorm2 => m.pnorm2, :pobs1 => m.pobs1, :pobs2 => m.pobs2, :rew => m.bad_rew, :stps_up => steps])
end

##Generate and Evaluate Policy Graphs
function completedReward(m::CF_POMDP,s,a,sp)
    if sp.x == m.rew_state
        return [1.0]
    else
        return [0.0]
    end
end

function failedReward(m::CF_POMDP,s,a,sp)
    if sp.f == m.term_fail || sp.x == m.term_state_bad
        return [1.0]
    else
        return [0.0]
    end
end


function stepsReward(m::CF_POMDP,s,a,sp)
    return [1.0]
end

function run_single_PG(m, policy, updater, graphtype, precision; tolerance = 0.001)
    up = updater
    # rew_fxn_list = [vectorizedReward,completedReward,failedReward,stepsReward]
    rew_fxn_list = [(vectorizedReward,true),(completedReward,true),(failedReward,true)]
    result_array = Vector{Float64}(undef,length(rew_fxn_list))
    b0 = initialize_belief(up,initialstate(m))
    if graphtype == :belief
        pg = ExtractBeliefPolicyGraph(m,updater,policy::Policy,b0::DiscreteBelief,precision)
    else
        pg = ExtractPolicyGraph(m,updater,policy::Policy,b0::DiscreteBelief)
    end
    for (i,(rew_fxn,disc)) in enumerate(rew_fxn_list)
        result_array[i] = EvalPolicyGraph(m,b0,pg;tolerance=tolerance,rewardfunction=rew_fxn,disc_io=disc)[1,3]
        # println(result_array)
    end
    return result_array
end


function allrewards(m::CF_POMDP,s,a,sp)
    r1 = reward(m,s,a,sp)
    if sp.x == m.rew_state
        r2 = 1.0
    else
        r2 = 0.0
    end
    if sp.f == m.term_fail || sp.x == m.term_state_bad
        r3 = 1.0
    else
        r3 = 0.0
    end
    return [r1,r2,r3]
end

function run_single_PG2(m, policy, updater, graphtype, precision; tolerance = 0.001, disc = discount(m))
    t_start = time()
    up = updater
    # rew_fxn_list = [vectorizedReward,completedReward,failedReward,stepsReward]
    # rew_fxn_list = [(vectorizedReward,true),(completedReward,false),(failedReward,false)]
    result_array = Vector{Float64}(undef,3)
    b0 = initialize_belief(up,initialstate(m))
    if graphtype == :belief
        pg = ExtractBeliefPolicyGraph(m,updater,policy::Policy,b0::DiscreteBelief,precision)
    else
        pg = ExtractPolicyGraph(m,updater,policy::Policy,b0::DiscreteBelief)
    end
    result_array = EvalPolicyGraph(m,b0,pg;tolerance=tolerance,rewardfunction=allrewards,disc=disc)[1,3,:]
    dt = time()-t_start
    println("$dt ellapsed for PG")
    return result_array
end
# function run_PGs(m, solvers)
#     rew_fxn_list = [vectorizedReward,completedReward,failedReward]
#     names = []
#     rewards = []
#     completions = []
#     failures = []
#     std_devs = zeros(length(solvers))
#     for (solver,name,updater) in solvers
#         pg = ExtractBeliefPolicyGraph(m,updater,pol::Policy,b0::DiscreteBelief,precision::Int64)
#         for rew_fxn in rew_fxn_list
#             values = EvalPolicyGraph(m,b0,pg;tolerance=tolerance,rewardfunction=rew_fxn) #SPECIFY A LIST OF REWARD FUNCTIONS HERE --> ARRIVAL, FAILURE, ETC.
#             push!(results_list,values)
#             push!(names,name)
#         end
#     end
#     return theframe = DataFrame([:Policy=>names,:Mean_Disc_Rew=>rewards, :SE_Disc_Rew=>std_devs, :Fract_Goal=>completions, :SE_Goal=>std_devs, :Fract_Fail=>failures, :SE_Fail=>std_devs,
#                :pnorm1 => m.pnorm1, :pnorm2 => m.pnorm2, :pobs1 => m.pobs1, :pobs2 => m.pobs2, :rew => m.bad_rew, :stps_up => std_devs])
# end



#Vary the probability of transition and observation across problems and run MC sims
function vary_probs_obs(n_runs,pobs1r=1.0:-0.2:0.5,pnorm1r=(0.99,0.96,0.93),pnorm2r=(0.99,0.9,0.75),rew= (-0.1,-0.5,-1.0,-2.5,-5.0,-7.5,-10.0,-20.0))#pnorm1r=1.0:-0.05:0.8)
    results = []
    for i in collect(pobs1r), k in collect(pobs1r), j in collect(pnorm1r), l in collect(pnorm2r), rv in collect(rew)#collect(pnorm1r)
        println("=== rew: $rv, pobs1: $i, pobs2: $k, pnorm1: $j, pnorm2: $l")
        cfv = init_cf(pobs1 = i, pobs2 = k, pnorm1 = j, pnorm2 = l, bad_rew = rv)
        mdp_pol = solve(ValueIterationSolver(max_iterations = 1000, belres = 0.002, verbose = false, include_Q = true),UnderlyingMDP(cfv)) #was 0.000001
        prevobs_solver = PrevObsSolver(mdp_pol)
        maxb_solver = MaxBSolver(mdp_pol)
        p_prevobs = solve(prevobs_solver, cfv)
        p_maxb = solve(maxb_solver, cfv)
        p_sarsop = solve(SARSOPSolver(precision=0.002,verbose = false),cfv)
        p_qmdp = AlphaVectorPolicy(cfv, mdp_pol.qmat, mdp_pol.action_map)

        solvers = [(p_sarsop, "SARSOP",DiscreteUpdater(cfv),:alpha,0),(p_qmdp, "QMDP",DiscreteUpdater(cfv),:belief,0.009),(p_prevobs,"Prev Obs MDP",PrevObsUpdater(cfv),:belief,0.009),(p_maxb, "Max Wt MDP",DiscreteUpdater(cfv),:belief,0.009)]
        result = run_MC(cfv,n_runs,solvers);
        results = vcat(results,result);
        #add UnderlyingMDP to solution set
        mdp_result = run_MC_mdp(cfv,n_runs,[(mdp_pol,"Full MDP")]);
        results = vcat(results,mdp_result);
    end
    results = reduce(vcat,results);
    CSV.write("Comp_Fail_"*Dates.format(now(),"dd_mm_HH:MM:SS")*".csv", results)
    return results
end

##Plot Results by Timestep to belief
function plot_res2(result, index) #where index varies from 0 to 30 for only varying two parameters
    # pobs = [collect(1.0:-0.1:0.5)]
    plot()
    for j in index#0:Int(sqrt((length(result)))):(length(result)-Int(sqrt((length(result)))))
    steps = []
    sarsop_g  = []
    sarsop_sem = []
    prev_o_g = []
    prev_o_sem = []
    max_b_g = []
    max_b_sem = []
        for i in (j+1):(j+Int(sqrt((length(result)))))
            # println(i)
            push!(steps,result[i].Fract_Fail[1])
            push!(prev_o_g,result[i].Fract_Goal[3])
            push!(prev_o_sem,result[i].SE_Goal[3])
            push!(sarsop_g,result[i].Fract_Goal[2])
            push!(sarsop_sem,result[i].SE_Goal[2])
            push!(max_b_g,result[i].Fract_Goal[4])
            push!(max_b_sem,result[i].SE_Goal[4])
        end
    println(sarsop_g)
    # plot(pobs,sarsop_g,seriestype = :scatter,yerror=sarsop_sem,label = "POMDP (SARSOP)")
    # plot!(pobs,prev_o_g,seriestype = :scatter,yerror=prev_o_sem,label="MDP (V.I. w/ P.O.)",xlabel="Probability of Correct Observation",ylabel="Fraction Reaching Target")
    plot!(steps,sarsop_g,yerror=sarsop_sem,label = "POMDP (SARSOP)")#,xlims=[0,15])
    plot!(steps,max_b_g,yerror=max_b_sem,label = "MDP (Max)")
    display(plot!(steps,prev_o_g,yerror=prev_o_sem,label="MDP (V.I. w/ P.O.)",xlabel="Timesteps to 50% or Greater Fault Belief",ylabel="Fraction Reaching Target"))
    # display(plot!(steps,prev_o_g,yerror=prev_o_sem,label="MDP (V.I. w/ P.O.)",xlabel="Belief on fault state after two fault observations",ylabel="Fraction Reaching Target"))#,ylims = [0,1])
    end
    #boxplot()
end

##Plot Pareto
function plot_pareto2(result,pnorm1,pnorm2,pobs1,pobs2)
    solver_list = ["SARSOP","QMDP","Prev Obs MDP","Max Wt MDP","Full MDP"]
    print_name = ["SARSOP","QMDP","Prev Obs MDP","Max Prob MDP","Omniscient"]
    line_style = [:solid,:solid,:solid,:solid,:dash]
    value_dict = Dict()
    p = plot(;size = (600,400).*1.0)
    for (i,name) in enumerate(solver_list)
        push!(value_dict, name*"_g" => result[(result.Policy.==name).&(result.pnorm1.==pnorm1).&(result.pnorm2.==pnorm2).&(result.pobs1.==pobs1).&(result.pobs2.==pobs2),:Fract_Goal])
        push!(value_dict, name*"_semg" => result[(result.Policy.==name).&(result.pnorm1.==pnorm1).&(result.pnorm2.==pnorm2).&(result.pobs1.==pobs1).&(result.pobs2.==pobs2),:SE_Goal])
        push!(value_dict, name*"_s" => ones(length(value_dict[name*"_g"]))-result[(result.Policy.==name).&(result.pnorm1.==pnorm1).&(result.pnorm2.==pnorm2).&(result.pobs1.==pobs1).&(result.pobs2.==pobs2),:Fract_Fail])
        push!(value_dict, name*"_sems" => result[(result.Policy.==name).&(result.pnorm1.==pnorm1).&(result.pnorm2.==pnorm2).&(result.pobs1.==pobs1).&(result.pobs2.==pobs2),:SE_Fail])
        rlabel = unique(result.rew)
        plot!(p,value_dict[name*"_s"],value_dict[name*"_g"],yerror=value_dict[name*"_semg"],xerror=value_dict[name*"_sems"],label = print_name[i],linestyle = line_style[i],legend = :bottomleft)
        # plot!(p,value_dict[name*"_s"],value_dict[name*"_g"],yerror=value_dict[name*"_semg"],xerror=value_dict[name*"_sems"],label = print_name[i],linestyle = line_style[i],xlims = [0.8875,1.0],ylims = [0.0,1.0],legend = :bottomleft) #For custom axis limits
        # plot!(p,value_dict[name*"_s"],value_dict[name*"_g"],label = print_name[i], series_annotations = text.(rlabel;pointsize=7))
    end
    # return p
    return plot!(p,title="pnorm1:$pnorm1, pnorm2:$pnorm2, pobs1:$pobs1, pobs2:$pobs2",xlabel="Fraction Safe",ylabel="Fraction Reaching Target")
    # return plot!(p,title="Prob. Remain in: Normal:$pnorm1, Fault:$pnorm2 \n Prob. Correct Obs in: Normal:$pobs1, Fault:$pobs2",xlabel="Fraction Safe",ylabel="Fraction Reaching Target")
    # plot!(p,title="Safety-Efficiency Trade-off \n Failure Unlikely, Noisy Observations",xlabel="Fraction Safe",ylabel="Fraction Reaching Target")
    # savefig(p,"ForCUAS_"*@Name(result)*"_$pnorm1,$pnorm2,$pobs1,$pobs2"*".svg") #For Specific Names
    # return p
end

macro Name(arg)
   string(arg)
end

function multi_pareto(result) #Plot a series of Preto Curves from CSV/DF
    for pnorm1 in unique(result.pnorm1), pnorm2 in unique(result.pnorm2), pobs1 in unique(result.pobs1), pobs2 in unique(result.pobs2)
        plot = plot_pareto2(result, pnorm1, pnorm2, pobs1, pobs2)
        display(plot)
        # savefig(@Name(result)*"_$pnorm1,$pnorm2,$pobs1,$pobs2"*".svg")
    end
end
