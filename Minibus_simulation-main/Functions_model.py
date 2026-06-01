import pandas as pd
import numpy as np
import gurobipy as gp
import seaborn as sns
import matplotlib.pyplot as plt
import random
import copy
import plotly.express as px
from tqdm.auto import tqdm
import time
import os
import json
from datetime import datetime

# Define the calibration function
def auto_calibrate_weights(costs_list, max_possible_detour_mins):
    # Find the most expensive bus in the fleet to base our penalties on
    max_cost = max(costs_list)  
    
    # Travel cost weight: Set to 1 so the objective function evaluates in actual currency
    b_1 = 1.0 
    
    # Lateness penalty: Make 1 minute of delay 15x more expensive than 1 minute of bus.
    b_3 = max_cost * 15.0 
    
    # Calculate the absolute worst-case scenario cost to serve a passenger:
    # (Longest possible detour driving cost) + (Penalty for being up to 5 minutes late)
    max_service_cost = (max_possible_detour_mins * max_cost) + (5 * b_3)
    
    # Rejection cost: Must be unquestionably worse than the worst-case service cost.
    # We double it to guarantee the solver always tries to pick the passenger up.
    # c_0 = max_service_cost * 1.2 #2.0  
    
    c_0 = max_cost * 20.0 
    
    # Rejection weight multiplier (keep at 1, as c_0 handles the actual magnitude)
    b_2 = 1.0
    
    return b_1, b_2, b_3, c_0

def get_travel_time(orig, dest, current_t, travel_dict):
    if orig == 0 or dest == 0: return 0.0 # Virtual nodes take 0 mins
    if orig == dest: return 0.0
    
    # Query the new dictionary using the current time step
    return travel_dict.get((current_t, orig, dest), 10.0)

def my_callback(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        # This triggers only when a new BEST integer solution is found
        obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
        bound = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
        current_time = time.time()
        time_since_last = current_time - model._last_sol_time
        model._last_sol_time = current_time
        #tqdm.write(f"   → New Best Obj: {obj:.2f} (Bound: {bound:.2f})")
        model._pbar.set_postfix({
            "Best Obj": f"{obj:.1f}", 
            "Bound": f"{bound:.1f}", 
            "Gap": f"{abs((obj-bound)/obj)*100:.1f}%" if obj != 0 else "N/A",
            "Sol_Time": f"{time_since_last:.1f}s"
            })

def generate_requests(params, current_req_id, passenger_history, status, t, origin_stations, stations):
    # Generate a certain number of requests with certain passenger sizes
    R_wait = []
    req_t_p_wait = []
    if status == "initial":
        num_initial_requests = np.random.poisson(params['init_reqs_num_poisson_lambda'])
        passenger_sizes = np.clip(np.random.poisson(lam=params['reqs_pax_size_poisson_lambda'], size=num_initial_requests), params['poisson_clip_lb'], params['poisson_clip_ub'])

        # For each request, randomly assign an origin and destination station
        for i in range(num_initial_requests):
            pax = int(passenger_sizes[i])
            orig = random.choice(origin_stations)
            dest = random.choice(stations)
        
            while dest == orig:
                dest = random.choice(stations)
            
            R_wait.append([pax, orig, dest, "wait", current_req_id])
            req_time = random.randint(0, params["interval"])
            req_t_p_wait.append(req_time)

            passenger_history[current_req_id] = {
                "status": "waiting",
                "orig": orig,
                "dest": dest,
                "time_requested": req_time,
                "time_picked_up": None,
                "time_dropped_off": None,
                "assigned_bus": None,
                "passenger_count": pax
            }

            current_req_id += 1

    elif status == "new_interval":
        num_new_requests = np.random.poisson(params["reqs_num_poisson_lambda"]) 
        passenger_sizes = np.clip(np.random.poisson(lam=params["reqs_pax_size_poisson_lambda"], size=num_new_requests), params["poisson_clip_lb"], params["poisson_clip_ub"])
        
        for i in range(num_new_requests):
            pax = int(passenger_sizes[i])
            orig = random.choice(origin_stations) 
            dest = random.choice(stations) 
            while dest == orig:
                dest = random.choice(stations)
            
            req_time = t + params["interval"] + random.randint(0, params["interval"])
            R_wait.append([pax, orig, dest, "wait", current_req_id])
            
            req_t_p_wait.append(req_time)

            passenger_history[current_req_id] = {
                "status": "waiting",
                "orig": orig,
                "dest": dest,
                "time_requested": req_time,
                "time_picked_up": None,
                "time_dropped_off": None,
                "assigned_bus": None,
                "passenger_count": pax
            }

            current_req_id += 1
    
    return R_wait, req_t_p_wait, current_req_id

def filter_waiting_passengers(R_wait, req_t_p_wait, params, t, passenger_history):
    # Determine the patience limit for passengers
    patience_limit = params['pax_max_wait']

    #
    filtered_R_wait = []
    filtered_req_t_p_wait = []
    n_abandonned = 0
    
    for req, rtp in zip(R_wait, req_t_p_wait):
        waited_so_far = t - rtp # how long they have been waiting
        req_id = req[4] # the unique request ID we assigned when generating requests
        
        if waited_so_far >= patience_limit:
            #  Passenger gives up — count as abandoned
            n_abandonned += 1
            tqdm.write(f"Passenger abandoned after {waited_so_far} min wait at t={t}")
            passenger_history[req_id]["status"] = "abandoned"
            
        else:
            # Still wihin patience window - carry over unchanged
            filtered_R_wait.append(req)
            filtered_req_t_p_wait.append(rtp)

    return filtered_R_wait, filtered_req_t_p_wait, n_abandonned

def model_construction(params, N, P_nodes, P_sched, P_wait, D_nodes, P_and_D, S_nodes, Z_nodes, Q, 
                       s_dict, t_dict, e_dict, l_dict, A_m, ub_dict, u_dict, n_req, M_stations, 
                       current_ghost_reqs, len_R_wait, bus_idx, bus_cost):

    # Initializing the model
    model_MILP_base = gp.Model("MILP_base")
    model_MILP_base.Params.OutputFlag = 0

    # Stop if the best solution is within 5% of the theoretical best
    model_MILP_base.Params.MIPGap = params["mip_gap"]
    
    # Stop after x seconds no matter what, and take the best found so far
    model_MILP_base.Params.TimeLimit = params["max_opt_time_seconds"]

    # 1. Pump up the heuristics. Default is 0.05 (5%). Set to 20% or higher.
    # This forces Gurobi to spend 20% of its time running quick algorithms 
    # to guess feasible solutions rather than purely doing branch-and-bound math.
    model_MILP_base.Params.Heuristics = params["heuristics"]
    
    # 2. Aggressive Symmetry Breaking. 
    # Routing problems have massive symmetry (Bus A doing Route X is the same cost as Bus B doing Route X).
    model_MILP_base.Params.Symmetry = params["symmetry"]
    
    # 3. Method selection for the root node.
    # Concurrent method (3) runs primal simplex, dual simplex, and barrier simultaneously.
    # It takes more RAM but usually finishes the root relaxation much faster.
    model_MILP_base.Params.Method = params["method"]

    # Initializing the decision variables
    x_base = model_MILP_base.addVars(N, N, bus_idx, vtype=gp.GRB.BINARY, name="x")
    q_k = model_MILP_base.addVars(N, bus_idx, vtype=gp.GRB.INTEGER, lb=0, ub=ub_dict, name="q_k")
    w_k = model_MILP_base.addVars(N, bus_idx, vtype=gp.GRB.CONTINUOUS, lb=0, ub=params["w_max"], name="w_k")
    a_k = model_MILP_base.addVars(N, bus_idx, vtype=gp.GRB.CONTINUOUS, name="a_k")
    #mu = model_MILP_base.addVars(N, bus_idx, vtype=gp.GRB.CONTINUOUS, lb=1, ub=len(N), name="mu")
    y = model_MILP_base.addVars(P_nodes, vtype=gp.GRB.BINARY, name="y")
    late_slack = model_MILP_base.addVars(N, bus_idx, vtype=gp.GRB.CONTINUOUS, lb=0, name="late_slack")

    for i in N:
        for j in N:
            if t_dict[i, j] > params["pax_max_wait"]:  
                for k in bus_idx:
                    x_base[i, j, k].UB = 0 # NEW

    # ---> WARM START GOES HERE <---
    # 1. Reject all new waiting passengers to guarantee a baseline
    for i in P_wait:
        y[i].Start = 0
        y[i].VarHintVal = 0
        y[i].VarHintPri = 100 # NEW
        
        # 2. Explicitly tell the buses NOT to drive to or from these rejected nodes
        # This prevents Gurobi from wasting 120 seconds trying to figure out the routes!
        for k in bus_idx:
            #for j in N:
            #    x_base[i, j, k].Start = 0           # Don't leave pickup
            #    x_base[j, i, k].Start = 0           # Don't enter pickup
            #    x_base[i+n_req, j, k].Start = 0     # Don't leave drop-off
            #    x_base[j, i+n_req, k].Start = 0     # Don't enter drop-off
            for i in P_sched: # NEW
                if u_dict.get((k, i), 0) > 0.5:
                    # Give Gurobi a massive hint: "Bus k served passenger i last time, do it again."
                    y[i].VarHintVal = 1
                    y[i].VarHintPri = 100
                    
                    # We also hint that bus k should definitely drive from the pickup to the dropoff
                    d = i + n_req
                    # We don't know the exact sequence, but we know this link must exist in the route
                    x_base[i, d, k].VarHintVal = 1 
                    x_base[i, d, k].VarHintPri = 50
    
    # Objective funtion to be minimized
    obj_expr_trav_cost = gp.quicksum(
        params["b_1"] * bus_cost[k] * t_dict[i, j] * x_base[i, j, k]
        for i in N
        for j in N
        for k in bus_idx
    )

    obj_expr_reject_cost = gp.quicksum(
        params["b_2"] * params["c_0"] * (1 - y[i]) for i in P_nodes
    )

    obj_late_penalty = gp.quicksum(params["b_3"] * late_slack[i, k] for i in N for k in bus_idx)

    model_MILP_base.setObjective(obj_expr_trav_cost + obj_expr_reject_cost + obj_late_penalty, gp.GRB.MINIMIZE)

    # Constraints
    for i in P_nodes:
        model_MILP_base.addConstr(
            gp.quicksum(x_base[i, j, k] for j in P_and_D for k in bus_idx) == y[i]
        ) # DO NOT REPLACE WITH .sum "*", WILL SUM OVER ALL NODES INSTEAD OF JUST P_AND_D

    # Previous scheduled request served
    for i in P_sched:
        for k in bus_idx:
            model_MILP_base.addConstr(
                x_base.sum(i, "*", k) == u_dict[k, i]
            )

    # Each request served at most once
    for i in P_wait:
        model_MILP_base.addConstr(
            x_base.sum(i, "*", "*") <= 1
        )

    # Ensures same vehicle visits pickup and drop-off nodes of same request
    for i in P_nodes:
        d = i + n_req
        for k in bus_idx:
            model_MILP_base.addConstr(
                x_base.sum(i, "*", k) - x_base.sum(d, "*", k) == 0
            )

    # Flow conservation constraints
    for i in P_nodes + D_nodes:
        for k in bus_idx:
            model_MILP_base.addConstr(
                x_base.sum(i, "*", k) - x_base.sum("*", i, k) == 0
            )

    for k in bus_idx:
        model_MILP_base.addConstr(
            x_base.sum(S_nodes[k], "*", k) - x_base.sum("*", S_nodes[k], k) == 1
        )

    for k in bus_idx:
        model_MILP_base.addConstr(
            x_base.sum(Z_nodes[k], "*", k) - x_base.sum("*", Z_nodes[k], k) == -1
        )

    # NEW: Anchor initial passenger load to 0 for all buses
    for k in bus_idx:
        model_MILP_base.addConstr(
            q_k[S_nodes[k], k] == 0,
            name=f"force_empty_start_{k}"
        )

    # Capacity constraints
    for i in N:
        for j in N:
            for k in bus_idx:
                model_MILP_base.addConstr(
                    q_k[i, k] + Q[i] - params["M_cap"] * (1 - x_base[i, j, k]) <= q_k[j, k]
                )
                model_MILP_base.addConstr(
                    q_k[j, k] <= q_k[i, k] + Q[i] + params["M_cap"] * (1 - x_base[i, j, k])
                )

    # Time constraints
    for i in N:
        for j in N:
            for k in bus_idx:
                M_time_window_ij = l_dict[i] + params["max_late"] + params["w_max"] + s_dict[i] + t_dict[i, j] - e_dict[j]
                if M_time_window_ij < 0:
                    continue

                model_MILP_base.addConstr(
                    a_k[i, k] + w_k[i, k] + s_dict[i] + t_dict[i, j] - M_time_window_ij * (1 - x_base[i, j, k]) <= a_k[j, k]
                )

    #for m in M_stations:
    #    model_MILP_base.addConstr(
    #        gp.quicksum(A_m[i, m] * w_k[i, k] for i in P_and_D for k in bus_idx) <= params["w_max"]
    #    )

    for i in P_nodes:
        for k in bus_idx:
            # Lower bound remains strict (you can't arrive before the request exists)
            model_MILP_base.addConstr(
                e_dict[i] <= a_k[i, k] + w_k[i, k] + params["M_time_window"] * (1 - x_base.sum(i, '*', k))
            )
            # SOFT Upper bound: allow arrival > l_dict, but it will cost the objective function
            model_MILP_base.addConstr(
                a_k[i, k] <= l_dict[i] + late_slack[i, k] + params["M_time_window"] * (1 - x_base.sum(i, '*', k))
            ) #-> led to some people exceeding their maximum wait time at the stop
            # HARD Upper bound: For new passengers, if you can't arrive by 13 mins, reject them!
            #model_MILP_base.addConstr(
            #    a_k[i, k] <= l_dict[i] + params["M_time_window"] * (1 - x_base.sum(i, '*', k))
            #) # NEW -> but crashes more often due to infeasibility


    for i in D_nodes:
        for k in bus_idx:
            # SOFT Upper bound for drop-offs
            model_MILP_base.addConstr(
                a_k[i, k] <= l_dict[i] + late_slack[i, k] + params["M_time_window"] * (1 - x_base.sum(i, '*', k))
            )

    for i in P_nodes:
        for k in bus_idx:
            model_MILP_base.addConstr(
                a_k[i+n_req, k] - (a_k[i, k] + w_k[i,k] + s_dict[i]) <= params["a_max"] * t_dict[i, i+n_req] 
                + params["M_time_window"] * (1 - x_base.sum(i, '*', k))
            )

    for i in P_nodes:
        for k in bus_idx:
            model_MILP_base.addConstr(
                a_k[i, k] + w_k[i, k] + s_dict[i] + t_dict[i, i+n_req] <= a_k[i+n_req, k] 
                + params["M_time_window"] * (1 - x_base.sum(i, '*', k))
            )

    #for i in N:
    #    for j in N:
    #        for k in bus_idx:
    #            if i != j and i in P_nodes + D_nodes and j in P_nodes + D_nodes:
    #                model_MILP_base.addConstr(
    #                    mu[i, k] - mu[j, k] + params["M"] * x_base[i, j, k] <= params["M"] - 1,
    #                    name=f"subtour_{i}_{j}_{k}"
    #                )

    # Logical constraints to reduce symmetry and infeasibility
    for i in N:
            for k in bus_idx:
                model_MILP_base.addConstr(
                    x_base[i, i, k] == 0
                )

    # Ensure that each vehicle starts at its assigned start node and ends at its assigned end node
    for k in bus_idx:
        model_MILP_base.addConstr(
            x_base.sum("*", S_nodes[k], k) == 0
        ) # Vehicle cannot enter start node


    for k in bus_idx: # vehicle cannot enter or leave any other vehicle's start or end node
        for k_other in bus_idx:
            if k != k_other:
                # forbid vehicle k from visiting start of other vehicles
                model_MILP_base.addConstr(
                    x_base.sum('*', S_nodes[k_other], k) == 0
                )
                model_MILP_base.addConstr(
                    x_base.sum(S_nodes[k_other], '*', k) == 0
                )

                # forbid vehicle k from visiting end of other vehicles
                model_MILP_base.addConstr(
                    x_base.sum('*', Z_nodes[k_other], k) == 0
                )
                model_MILP_base.addConstr(
                    x_base.sum(Z_nodes[k_other], '*', k) == 0
                )

    # We need to know which logical node ID corresponds to which ghost request.
    # Because ghosts are pre-pended to R_sched, their P_node IDs start exactly after P_wait.
    current_p_idx = len_R_wait 
    
    for k in bus_idx:
        # For each vehicle, check if there are ghost requests that need to be served
        ghosts_for_k = current_ghost_reqs.get(k, [])
        
        if ghosts_for_k:
            previous_node = S_nodes[k] # Start at the vehicle's starting node
            
            for _ in ghosts_for_k:
                ghost_p_node = current_p_idx
                
                # 1. Force a direct link from the previous node to this ghost node
                model_MILP_base.addConstr(x_base[previous_node, ghost_p_node, k] == 1)
                
                # 2. Advance the pointers
                previous_node = ghost_p_node
                current_p_idx += 1

    # NEW
    # Force vehicles to start their current route at or after the current time step
    for k in bus_idx:
        model_MILP_base.addConstr(
            e_dict[S_nodes[k]] <= a_k[S_nodes[k], k],
            name=f"force_start_time_{k}"
        )
    
    model_MILP_base.Params.StartNodeLimit = params["start_node_limit"]

    # Focus on finding feasible solutions quickly rather than proving optimality
    model_MILP_base.Params.MIPFocus = params["mip_focus_feasibility"]

    # Aggressively pre-solve the model to shrink it before the root relaxation
    model_MILP_base.Params.Presolve = params["presolve_aggressive"]

    # Controls the generation of cutting planes
    model_MILP_base.Params.Cuts = params["cuts"]

    # Controls the branch variable selection strategy
    model_MILP_base.Params.VarBranch = params["varbranch"]

    model_MILP_base.Params.NoRelHeurTime = params["no_rel_heur_time"] # NEW

    return model_MILP_base, x_base, q_k, w_k, a_k, y, late_slack#, mu

def run_simulation(params, stations, origin_stations, initial_K, bus_idx, bus_cost, get_travel_time_func):

    # 1. Initialization
    # Set random seeds for reproducibility
    random.seed(params['seed'])
    np.random.seed(params['seed'])

    # Setup time stamps for the simulation
    time_stamps = range(params["t_start"], params["t_end"] + 1, params["interval"])

    # Initialize the fleet
    K = copy.deepcopy(initial_K)  # Deep copy to avoid modifying the original

    # Generate initial requests
    passenger_history = {}
    global_req_id = 0
    R_wait, req_t_p_wait, global_req_id = generate_requests(params, global_req_id, passenger_history, "initial", 0, origin_stations, stations)

    R_sched = []  # List to hold scheduled requests
    req_t_p_sched = []  # List to hold scheduled request times

    R = R_wait + R_sched  # All requests (waiting + scheduled)
    req_t_p = req_t_p_wait + req_t_p_sched  # Corresponding times
    n_req = len(R)
    u_dict_assignments_carryover = [] 
    
    current_ghost_reqs = {k: [] for k in bus_idx}

    history_routes = {k: [] for k in bus_idx}
    history_stats = {
        "time_step": [],
        "new_reqs_presented": [],
        "reqs_rejected": [],
        "reqs_abandonned": [],
        "obj_cost": [],
        "in_transit_carried_over": [],
        "solve_time_seconds": [],
        "mip_gap": [],
        "node_count": [],
        "solve_status": []
    }

    pbar = tqdm(time_stamps, desc="Simulation Progress", unit="interval")

    #for t in tqdm(time_stamps, desc="Simulation Progress", unit="interval"):
    for t in pbar:
        t_next = t + params["interval"]

        # 1. Filter for waiting passengers who haven't left the system
        R_wait, req_t_p_wait, n_abandonned = filter_waiting_passengers(R_wait, req_t_p_wait, params, t, passenger_history)

        R = R_wait + R_sched
        req_t_p = req_t_p_wait + req_t_p_sched
        n_req = len(R)

        # 2. Generate nodes for each request
        # 2.1 Pickup nodes
        P_nodes = list(range(n_req))
        P_wait = list(range(len(R_wait)))
        P_sched = list(range(len(R_wait), len(R_wait) + len(R_sched)))

        u_dict = {}
        for idx, node_id in enumerate(P_sched):
            assigned_k = u_dict_assignments_carryover[idx]
            for k in bus_idx:
                u_dict[k, node_id] = 1 if (assigned_k is not None and k == assigned_k) else 0

        # 2.2 Dropoff nodes
        D_nodes = list(range(n_req, 2*n_req))

        # 2.3 Virtual start and end nodes
        S_nodes = list(range(2*n_req, 2*n_req + len(K)))
        Z_nodes = list(range(2*n_req + len(K), 2*n_req + 2*len(K)))

        # 2.4 Create node sets
        P_and_D = P_nodes + D_nodes
        N = P_nodes + D_nodes + S_nodes + Z_nodes

        # 2.5 Create mapping from modeling to physical nodes
        P_loc = {i: R[i][1] for i in range(n_req)}
        D_loc = {i + n_req: R[i][2] for i in range(n_req)}
        S_loc = {S_nodes[k]: K[k][1] for k in range(len(K))}
        Z_loc = {Z_nodes[k]: K[k][2] for k in range(len(K))}

        node_to_loc = {}
        node_to_loc.update(P_loc)
        node_to_loc.update(D_loc)
        node_to_loc.update(S_loc)
        node_to_loc.update(Z_loc)

        # 3. Trip time definition
        # 3.1 Calculate travel times between all nodes keyed by logical nodes
        t_dict = {(i,j): get_travel_time_func(node_to_loc[i], node_to_loc[j], t) for i in N for j in N}

        for i in N:
            for j in Z_nodes:
                t_dict[i, j] = 0

        # 3.2 Boarding/alighting time
        s_dict = {i: params['board_alight_time'] if i in P_nodes + D_nodes else 0 for i in N}
        
        # If in transit, no service time
        for idx, i in enumerate(P_nodes):
            if len(R[idx]) > 3 and R[idx][-1] == "ghost":
                s_dict[i] = 0  # No service time for in-transit passengers

        # 3.3 Time windows for pickups and dropoffs
        # Earliest pickup time is request time, latest pickup time is request time + max wait
        # No constraint on earliest dropoff, latest dropoff is t_end + max_late
        tep, tlp, ted, tld = {}, {}, {}, {}

        for idx, i in enumerate(P_nodes):
            tep[i] = req_t_p[idx]  
            tlp[i] = tep[i] + params['pax_max_wait']

        for idx, i in enumerate(D_nodes):
            pickup_node = i - n_req
            travel_time = t_dict[pickup_node, i]
            ted[i] = 0
            tld[i] = tlp[pickup_node] + travel_time * params['a_max']
        
        e_dict, l_dict = {}, {}

        for i in P_nodes:
            e_dict[i] = tep[i]
            l_dict[i] = tlp[i]

        for i in D_nodes:
            e_dict[i] = ted[i]
            l_dict[i] = tld[i]
        
        for i in S_nodes:
            e_dict[i] = t
            l_dict[i] = 1440

        for i in Z_nodes:
            e_dict[i] = params["t_end"] # Force vehicles to only return by the end of the day and not before
            l_dict[i] = 1440

        # 4. Capacities
        Q = {}

        for i in P_nodes:
            Q[i] = R[i][0]
        
        for i in D_nodes:
            Q[i] = -R[i - n_req][0]

        for i in S_nodes + Z_nodes:
            Q[i] = 0

        Q_max = [vehicle[0] for vehicle in K]

        ub_dict = {(i, k): Q_max[k] for i in N for k in bus_idx}

        # 5. Mapping logical to physical nodes in matrix
        M_stations = list(set(node_to_loc.values()))

        A_m = {}

        for i in P_and_D:
            for m in M_stations:
                if node_to_loc[i] == m:
                    A_m[i, m] = 1
                else:                    
                    A_m[i, m] = 0
        
        # 6. Build the optimization model
        model_MILP_base, x_base, q_k, w_k, a_k, y, late_slack = model_construction(params, N, P_nodes, P_sched, P_wait, D_nodes, P_and_D, S_nodes, Z_nodes, 
                                                                                       Q, s_dict, t_dict, e_dict, l_dict, A_m, ub_dict, u_dict, n_req, M_stations, 
                                                                                       current_ghost_reqs, len(R_wait), bus_idx, bus_cost)

        if 'x_sol' in locals(): # Only if a solution exists from the previous t
            # Reset all start values to 0 (or undefined)
            for v in x_base.values():
                v.Start = 0
            
            # We use u_dict (the one built at the end of the PREVIOUS loop)
            # to suggest which P-nodes belong to which vehicles.
            for (k, i_new) in u_dict:
                if u_dict[k, i_new] > 0.5:
                    # Suggest that this vehicle k visits its assigned pickup
                    # and corresponding dropoff.
                    # Note: This is a 'partial' start. Gurobi will try to 
                    # fill in the path (S -> P -> D -> Z) to make it feasible.
                    
                    # Find a j for the path. This is tricky because the sequence might change.
                    # Usually, setting y[i] and partial x is enough for Gurobi to find the rest.
                    if i_new in y:
                        y[i_new].Start = 1
        
        model_MILP_base._pbar = pbar
        model_MILP_base._last_sol_time = time.time()

        # 7. Optimize the model
        model_MILP_base.optimize(my_callback)

        if model_MILP_base.status == gp.GRB.INFEASIBLE: # NEW
            print(f"\n🚨 MODEL INFEASIBLE AT t={t} 🚨")
            print("Computing IIS (Irreducible Inconsistent Subsystem)...")
            model_MILP_base.computeIIS()
            
            # Save the exact contradicting rules to a text file
            iis_filename = f"infeasible_t{t}.ilp"
            model_MILP_base.write(iis_filename)
            print(f"✅ Saved conflicting constraints to {iis_filename}. Open this file to see the exact contradiction.")
            
            # Stop the simulation so you can investigate
            break

        if model_MILP_base.status == gp.GRB.TIME_LIMIT and model_MILP_base.SolCount == 0: # NEW
            print(f"⚠️ t={t}: SOLVER OVERWHELMED. Triggering Fallback Plan...")
            
            # 1. Force the solver to reject all waiting passengers for this interval
            for i in P_wait:
                model_MILP_base.addConstr(y[i] == 0, name=f"panic_reject_{i}")
            
            # 2. Give Gurobi 30 seconds to quickly string together the remaining ghosts/scheduled
            model_MILP_base.Params.TimeLimit = 120
            print("   -> Re-routing only active/ghost passengers...")
            model_MILP_base.optimize()

        # 8. Solution extraction and system update
        if model_MILP_base.status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT] and model_MILP_base.SolCount > 0:
            # 8.1 Extract the solution values for decision variables
            x_sol = model_MILP_base.getAttr('X', x_base)
            y_sol = model_MILP_base.getAttr('X', y)
            q_sol = model_MILP_base.getAttr('X', q_k)
            w_sol = model_MILP_base.getAttr('X', w_k)
            a_sol = model_MILP_base.getAttr('X', a_k)

            # 8.2 Calculate rejection rate
            presented = len(P_wait)
            rejected = sum(1 for i in P_wait if y_sol[i] < 0.5)

            opt_runtime = model_MILP_base.Runtime
            opt_nodecount = model_MILP_base.NodeCount
            opt_status = model_MILP_base.Status

            try:
                opt_mipgap = model_MILP_base.MIPGap
            except AttributeError:
                opt_mipgap = 0.0

            #tqdm.write(f"✓ t={t}: Solved. Obj: {model_MILP_base.ObjVal:.1f} | Rejected: {rejected}/{presented}")
            pbar.set_postfix({
                "Last_t": t,
                "Final_Obj": f"{model_MILP_base.ObjVal:.1f}",
                "Rejected": f"{rejected}/{presented}"
            })

            # 8.3 Update the history statistics and bus route statistics
            history_stats["time_step"].append(t)
            history_stats["new_reqs_presented"].append(presented)
            history_stats["reqs_rejected"].append(rejected)
            history_stats["reqs_abandonned"].append(n_abandonned)
            history_stats["obj_cost"].append(model_MILP_base.ObjVal)
            history_stats["solve_time_seconds"].append(opt_runtime)
            history_stats["mip_gap"].append(opt_mipgap)
            history_stats["node_count"].append(opt_nodecount)
            history_stats["solve_status"].append(opt_status)

            for k in bus_idx:
                route_for_k = []
                curr_node = S_nodes[k]
                
                while curr_node != Z_nodes[k]:
                    is_pickup = curr_node < n_req 
                    actual_arrival = a_sol.get((curr_node, k), t)

                    if is_pickup:
                        # Look up original requested time (req_t_p[curr_node]) to handle early arrivals
                        action_time = max(actual_arrival, req_t_p[curr_node]) + w_sol.get((curr_node, k), 0)
                    else:
                        action_time = actual_arrival + w_sol.get((curr_node, k), 0)

                    route_for_k.append({
                        "logical node": curr_node,
                        "location": node_to_loc[curr_node],
                        "arrival_time": action_time,
                        "passenger_load": q_sol.get((curr_node, k), 0) + Q.get(curr_node, 0)
                    })
                
                    next_node = None
                    for j in N:
                        if x_sol[curr_node, j, k] > 0.5:
                            next_node = j
                            break
                    
                    if next_node is not None:
                        curr_node = next_node
                    else:
                        break

                history_routes[k].append({
                    "interval": t,
                    "route": route_for_k
                })

            # 8.4 Update the system state for the next iteration
            next_R_sched, next_req_t_p_sched, u_dict_carryover_k = [], [], []

            # 8.4.1 Create ghost requests for in-transit passengers, to correctly carry them over to the next iteration.
            next_ghost_reqs = {k: [] for k in bus_idx}
            next_ghost_times = {k: [] for k in bus_idx}

            for idx, i in enumerate(P_nodes):
                d_node = i + n_req
                req_id = R[idx][4]
                
                # We check whether request was scheduled in a previous iteration or is newly scheduled in this iteration
                is_active = (i in P_sched) or (i in P_wait and y_sol[i] > 0.5)
                
                if is_active:
                    # 8.4.1.1Find when the drop-off happens
                    d_time = -1
                    for k in bus_idx:
                        if sum(x_sol[d_node, j, k] for j in N) > 0.5: # Check if bus k visits drop-off node
                            d_time = a_sol[d_node, k] + w_sol[d_node, k] # Extract arrival time at drop-off node #NEW added w_sol
                            break
                    
                    # 8.4.1.2 If drop-off happens after the current time step, it carries over
                    if d_time > t + params["interval"]:
                        # Find when and who picked them up
                        p_time = -1
                        assigned_k = None
                        for k in bus_idx:
                            if sum(x_sol[i, j, k] for j in N) > 0.5:
                                p_time = max(a_sol[i, k], req_t_p[idx]) + w_sol.get((i, k), 0) # NEW added
                                assigned_k = k
                                break
                        
                        carried_time = req_t_p[idx] # Original pickup time request

                        # If pickup already happened, move their origin to the vehicle's location and set pickup time to now.
                        if p_time <= t + params["interval"] and assigned_k is not None:
                            passenger_history[req_id]["status"] = "in_transit"
                            if passenger_history[req_id]["time_picked_up"] is None:
                                passenger_history[req_id]["time_picked_up"] = p_time
                            passenger_history[req_id]["assigned_bus"] = assigned_k
                            # IN-TRANSIT: Create a ghost request
                            ghost_req = R[idx].copy()
                            
                            # Update origin to vehicle's current node. 
                            last_visited = S_nodes[assigned_k]
                            max_a = -1
                            for n_idx in N:
                                if sum(x_sol[n_idx, j, assigned_k] for j in N) > 0.5:
                                    # This node is on the route of assigned_k, check if it's the last one before t_next
                                    if a_sol[n_idx, assigned_k] <= t + params["interval"] and a_sol[n_idx, assigned_k] > max_a:
                                        max_a = a_sol[n_idx, assigned_k]
                                        last_visited = n_idx

                            # Update the ghost request's origin to the last visited node's location
                            ghost_req[1] = node_to_loc[last_visited]
                            ghost_req.append("ghost") 
                            
                            next_ghost_reqs[assigned_k].append(ghost_req)
                            next_ghost_times[assigned_k].append(carried_time)
                        else:
                            # Scheduled but not picked up yet (normal carryover)
                            next_R_sched.append(R[idx].copy())
                            next_req_t_p_sched.append(carried_time)
                            u_dict_carryover_k.append(assigned_k)

                    # If drop-off happens within the current time step, we consider the request completed and do not carry it over
                    else:
                        passenger_history[req_id]["status"] = "completed"
                        if passenger_history[req_id]["time_dropped_off"] is None:
                            passenger_history[req_id]["time_dropped_off"] = d_time

                        for k in bus_idx:
                            if sum(x_sol[d_node, j, k] for j in N) > 0.5:
                                passenger_history[req_id]["assigned_bus"] = k
                            if sum(x_sol[i, j, k] for j in N) > 0.5:
                                if passenger_history[req_id]["time_picked_up"] is None:
                                    # ADD w_sol[i, k] so it logs Departure/Boarding time, not just Arrival NEW added w_sol
                                    passenger_history[req_id]["time_picked_up"] = max(a_sol[i, k], req_t_p[idx]) + w_sol.get((i, k), 0) # NEW

            # 8.4.2 Sort the next_R_sched so that ghost requests are in front of their corresponding normal requests, to ensure they get assigned to the same vehicle in the next iteration
            final_next_R_sched = []
            final_next_req_t_p_sched = []
            final_u_dict_assignments = []

            # 8.4.2.1 Add all ghosts in strict vehicle order
            for k in bus_idx:
                for g_req, g_time in zip(next_ghost_reqs[k], next_ghost_times[k]):
                    final_next_R_sched.append(g_req)
                    final_next_req_t_p_sched.append(g_time)
                    final_u_dict_assignments.append(k) # Lock ghost to this vehicle

            # 8.4.2.2 Add normal carryovers
            for n_req_item, n_time, n_k in zip(next_R_sched, next_req_t_p_sched, u_dict_carryover_k):
                final_next_R_sched.append(n_req_item)
                final_next_req_t_p_sched.append(n_time)
                final_u_dict_assignments.append(n_k)

            # Replace the old arrays with the newly sorted ones
            next_R_sched = final_next_R_sched
            next_req_t_p_sched = final_next_req_t_p_sched

            # 8.4.3 Update Vehicle Positions (K)
            for k in bus_idx:
                last_visited_node = S_nodes[k]
                max_a = -1
                for i in N:
                    if i not in Z_nodes:
                        if sum(x_sol[i, j, k] for j in N) > 0.5:
                            if a_sol[i, k] <= t_next and a_sol[i, k] > max_a:
                                max_a = a_sol[i, k]
                                last_visited_node = i

                K[k][1] = node_to_loc[last_visited_node]

            # 8.4.4 Generate New Requests (R_wait)
            new_R_wait, new_req_t_p_wait, global_req_id = generate_requests(params, global_req_id, passenger_history, "new_interval", t_next, origin_stations=origin_stations, stations=stations)
            
            # 8.4.5 Rebuild u_dict with new logical indices 
            # Next iteration's P_nodes will be ordered as: [ ...new_R_wait..., ...next_R_sched... ]
            # Unserved waiting passengers carry forward.
            # They already passed the patience check at the TOP of this loop,
            # so any survivor here still has remaining patience.
            unserved_R_wait       = [R_wait[i] for i in P_wait if y_sol[i] < 0.5]
            unserved_req_t_p_wait = [req_t_p_wait[i] for i in P_wait if y_sol[i] < 0.5]
            
            # New arrivals + passengers the optimizer rejected but still have patience
            R_wait        = new_R_wait + unserved_R_wait
            req_t_p_wait  = new_req_t_p_wait + unserved_req_t_p_wait
            R_sched       = next_R_sched
            req_t_p_sched = next_req_t_p_sched
            
            # Rebuild full request list
            R             = R_wait + R_sched
            req_t_p       = req_t_p_wait + req_t_p_sched
            n_req         = len(R)

            # ---> Pass the raw assignments forward to the next loop <---
            u_dict_assignments_carryover = final_u_dict_assignments

            current_ghost_reqs = next_ghost_reqs
 
            history_stats['in_transit_carried_over'].append(len(next_R_sched))

        else:
            # --- 4. THE FALLBACK FAILED ---
            print(f"❌ FATAL: Even the fallback plan failed to find a solution at t={t}. Halting simulation.")
            break
        #else:
        #    if model_MILP_base.SolCount == 0:
        #        print(f"No solution found at t={t} within time limit or model is infeasible.")
        #    break

        # ==========================================
        # 9. MEMORY CLEANUP (Forget the old model)
        # ==========================================
        # 1. Safely tell Gurobi to destroy the backend C-memory
        if 'model_MILP_base' in locals():
            model_MILP_base.dispose()
        
        # 2. Delete the massive Python dictionaries holding variables
        del x_base, q_k, w_k, a_k, y, late_slack#, mu
        
        # 3. Delete the solution dictionaries (if they exist)
        if 'x_sol' in locals():
            del x_sol, y_sol, q_sol, w_sol, a_sol
    
    return history_routes, history_stats, passenger_history

def save_experiment(SIM_PARAMS, df_stats, df_routes, df_pax, experiment_name=None, base_folder="MILP_Experiments"):
    
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"Run_{timestamp}_n1_{SIM_PARAMS['n1']}_n2_{SIM_PARAMS['n2']}_n3_{SIM_PARAMS['n3']}_max_opt_time_seconds_{SIM_PARAMS['max_opt_time_seconds']}_start_node_limit_{SIM_PARAMS['start_node_limit']}_mip_focus_feasibiility_{SIM_PARAMS['mip_focus_feasibility']}_presolve_aggressive_{SIM_PARAMS['presolve_aggressive']}_mip_gap_{SIM_PARAMS['mip_gap']}_heuristics_{SIM_PARAMS['heuristics']}_symmetry_{SIM_PARAMS['symmetry']}_method_{SIM_PARAMS['method']}"

    run_folder = os.path.join(base_folder, experiment_name)
    os.makedirs(run_folder, exist_ok=True)
    print(f"\n📁 Saving Experiment to: {run_folder}")

    df_stats.to_csv(os.path.join(run_folder, "stats.csv"), index=False)
    df_routes.to_csv(os.path.join(run_folder, "routes.csv"), index=False)
    df_pax.to_csv(os.path.join(run_folder, "passengers.csv"), index=False)
    
    # Safely write JSON (converting numpy ints/floats to standard Python types)
    clean_params = {}
    for k, v in SIM_PARAMS.items():
        # Convert numpy types to native python types so JSON doesn't crash
        if hasattr(v, 'item'): 
            clean_params[k] = v.item()
        else:
            clean_params[k] = v
            
    with open(os.path.join(run_folder, "parameters.json"), "w") as f:
        json.dump(clean_params, f, indent=4)
        
    total_pax = len(df_pax)
    served_pax = len(df_pax[df_pax['status'] == 'completed'])
    service_rate = (served_pax / total_pax) * 100 if total_pax > 0 else 0
    abandoned_pax = len(df_pax[df_pax['status'] == 'abandoned'])
    
    completed_pax_df = df_pax[df_pax['status'] == 'completed']
    avg_wait = completed_pax_df['wait_time_mins'].mean() if not completed_pax_df.empty else 0
    
    avg_solve_time = df_stats['solve_time_seconds'].mean() if not df_stats.empty else 0
    total_rejections = df_stats['reqs_rejected'].sum() if not df_stats.empty else 0
    final_obj_cost = df_stats['obj_cost'].iloc[-1] if not df_stats.empty else 0
    
    # Calculate how many intervals the simulation should have run
    expected_intervals = len(range(SIM_PARAMS["t_start"], SIM_PARAMS["t_end"] + 1, SIM_PARAMS["interval"]))
    actual_intervals = len(df_stats)
    
    if actual_intervals < expected_intervals:
        # If it didn't finish all steps, it broke early due to infeasibility
        feasibility_status = "No" 
    else:
        # It made it through the whole day successfully
        feasibility_status = "Yes"
    
    kpi_row = {
        "Experiment_Name": experiment_name,
        "Total_Requests": total_pax,
        "Service_Rate_%": round(service_rate, 2),
        "Abandoned_Pax": abandoned_pax,
        "Avg_Wait_Mins": round(avg_wait, 2),
        "Total_Solver_Rejections": total_rejections,
        "Avg_Solve_Time_Secs": round(avg_solve_time, 2),
        "Final_Obj_Cost": round(final_obj_cost, 2),
        "Overall_Feasible": feasibility_status
    }
    
    for key, value in clean_params.items():
        kpi_row[f"Param_{key}"] = value
    
    df_kpi = pd.DataFrame([kpi_row])
    master_tracker_path = os.path.join(base_folder, "MASTER_EXPERIMENT_TRACKER.csv")
    
    if os.path.exists(master_tracker_path):
        existing_tracker = pd.read_csv(master_tracker_path)
        updated_tracker = pd.concat([existing_tracker, df_kpi], ignore_index=True)
        updated_tracker.to_csv(master_tracker_path, index=False)
    else:
        df_kpi.to_csv(master_tracker_path, index=False)
        
    print(f"✅ Logged KPIs to Master Tracker: {master_tracker_path}")