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

def extract_simulation_params(csv_path):
    # 1. Read the CSV file containing bus arrival data at each station
    df = pd.read_csv(csv_path)
    
    # 2. Convert the 'arrival_time' column from plain text strings into  datetime objects
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    
    # 3. Sort the data by station_id and chronologically: Every bus that stops at Station A in the exact order they arrive
    df_sorted = df.sort_values(by=['station_id', 'arrival_time'])
    
    # 4. Shift the arrival times up by one row within each station group: Each row now contains the arrival time of the current bus and the 'next_arrival' of the following bus
    df_sorted['next_arrival'] = df_sorted.groupby('station_id')['arrival_time'].shift(-1)
    
    # 5. Subtract the current bus time from the next bus time to find the gap
    df_sorted['headway_mins'] = (df_sorted['next_arrival'] - df_sorted['arrival_time']).dt.total_seconds() / 60.0 # Divided by 60 to convert seconds to minutes
    
    # 6. Filter out gaps longer than 2 hours: Removes overnight/service breaks which shouldn't count as standard passenger wait time
    valid_headways = df_sorted[df_sorted['headway_mins'] < 120]
    
    # 7. Take the 90th percentile. 
    pax_max_wait = valid_headways['headway_mins'].quantile(0.90)
    
    # 8. Sort by bus_id and stop_sequence to track a physical bus as it drives its route
    df_route = df.sort_values(by=['bus_id', 'stop_sequence'])
    
    # 9. Shift the arrival time up by one row to bring the arrival time at the next stop into the current row
    df_route['next_station_time'] = df_route.groupby('bus_id')['arrival_time'].shift(-1)
    
    # 10. Subtract the times to find out how many minutes it took the bus to drive from Stop A to Stop B
    df_route['travel_time_mins'] = (df_route['next_station_time'] - df_route['arrival_time']).dt.total_seconds() / 60.0
    
    # 11. Calculate the mean of all these short station-to-station hops: Gives a baseline for how long direct travel takes, helping for detour penality calibration (a_max)
    avg_travel_time = df_route[df_route['travel_time_mins'] > 0]['travel_time_mins'].mean()
    
    return {
        "pax_max_wait": int(round(pax_max_wait)), 
        "avg_travel_time": avg_travel_time        
    }

def auto_calibrate_weights(costs_list, max_possible_detour_mins):
    # 1. Find the most expensive bus in the fleet to base our penalties on
    max_cost = max(costs_list)  
    
    # 2. Travel cost weight: Set to 1 so the objective function evaluates in actual currency
    b_1 = 1.0 

    # 3. Rejection weight multiplier (keep at 1, as c_0 handles the actual magnitude)
    b_2 = 1.0
    
    # 4. Lateness penalty: Make 1 minute of delay 15x more expensive than 1 minute of bus.
    b_3 = max_cost * 15.0 
    max_service_cost = (max_possible_detour_mins * max_cost) + (5 * b_3)
    
    # 5. Base rejection cost: Set to 20x to strongly discourage rejections and make it the highest penalty
    c_0 = max_cost * 20.0 
    
    
    
    return b_1, b_2, b_3, c_0

def get_travel_time(orig, dest, current_t, travel_dict):
    if orig == 0 or dest == 0: return 0.0 # Virtual nodes take 0 mins
    if orig == dest: return 0.0
    
    return travel_dict.get((current_t, orig, dest), 10.0)

def my_callback(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
        bound = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
        current_time = time.time()
        time_since_last = current_time - model._last_sol_time
        model._last_sol_time = current_time
        model._pbar.set_postfix({
            "Best Obj": f"{obj:.1f}", 
            "Bound": f"{bound:.1f}", 
            "Gap": f"{abs((obj-bound)/obj)*100:.1f}%" if obj != 0 else "N/A",
            "Sol_Time": f"{time_since_last:.1f}s"
            })

def generate_requests(params, current_req_id, passenger_history, status, t, origin_stations, stations):
    R_wait = []
    req_t_p_wait = []
    if status == "initial":
        num_initial_requests = np.random.poisson(params['init_reqs_num_poisson_lambda'])
        passenger_sizes = np.clip(np.random.poisson(lam=params['reqs_pax_size_poisson_lambda'], size=num_initial_requests), params['poisson_clip_lb'], params['poisson_clip_ub'])

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
    patience_limit = params['pax_max_wait']

    filtered_R_wait = []
    filtered_req_t_p_wait = []
    n_abandonned = 0
    
    for req, rtp in zip(R_wait, req_t_p_wait):
        waited_so_far = t - rtp 
        req_id = req[4] # unique request ID assigned when generating requests
        
        if waited_so_far >= patience_limit:
            n_abandonned += 1
            tqdm.write(f"Passenger abandoned after {waited_so_far} min wait at t={t}")
            passenger_history[req_id]["status"] = "abandoned"
            
        else:
            # Still wihin patience window - carry over unchanged
            filtered_R_wait.append(req)
            filtered_req_t_p_wait.append(rtp)

    return filtered_R_wait, filtered_req_t_p_wait, n_abandonned

def model_construction(params, N, P_nodes, P_sched, P_wait, D_nodes, P_and_D, S_nodes, Z_nodes, Q, 
                       s_dict, t_dict, e_dict, l_dict, ub_dict, u_dict, n_req, 
                       current_ghost_reqs, len_R_wait, bus_idx, bus_cost):

    # 1. Model initialization
    model_MILP_base = gp.Model("MILP_base")
    
    # 2. Model parameters and settings
    model_MILP_base.Params.OutputFlag = 0
    model_MILP_base.Params.MIPGap = params["mip_gap"]
    model_MILP_base.Params.TimeLimit = params["max_opt_time_seconds"]
    model_MILP_base.Params.Heuristics = params["heuristics"]
    model_MILP_base.Params.Symmetry = params["symmetry"]
    model_MILP_base.Params.Method = params["method"]
    model_MILP_base.Params.StartNodeLimit = params["start_node_limit"]
    model_MILP_base.Params.MIPFocus = params["mip_focus_feasibility"]
    model_MILP_base.Params.Presolve = params["presolve_aggressive"]
    model_MILP_base.Params.Cuts = params["cuts"]
    model_MILP_base.Params.VarBranch = params["varbranch"]
    model_MILP_base.Params.NoRelHeurTime = params["no_rel_heur_time"]

    # 3. Initializing the decision variables
    x_base = model_MILP_base.addVars(N, N, bus_idx, vtype=gp.GRB.BINARY, name="x")
    a_k = model_MILP_base.addVars(N, bus_idx, vtype=gp.GRB.CONTINUOUS, lb=0, name="a_k")
    w_k = model_MILP_base.addVars(N, bus_idx, vtype=gp.GRB.CONTINUOUS, lb=0, ub=params["w_max"], name="w_k")
    q_k = model_MILP_base.addVars(N, bus_idx, vtype=gp.GRB.INTEGER, lb=0, ub=ub_dict, name="q_k")
    y = model_MILP_base.addVars(P_nodes, vtype=gp.GRB.BINARY, name="y")
    late_slack = model_MILP_base.addVars(N, bus_idx, vtype=gp.GRB.CONTINUOUS, lb=0, name="late_slack")

    # 4. Warm start (reject all new waiting passengers to guarantee a baseline)
    for i in P_wait:
        y[i].Start = 0
        y[i].VarHintVal = 0
        y[i].VarHintPri = 100 
        
        for k in bus_idx:
            for i in P_sched: 
                if u_dict.get((k, i), 0) > 0.5:
                    y[i].VarHintVal = 1
                    y[i].VarHintPri = 100
                    
                    d = i + n_req
                    x_base[i, d, k].VarHintVal = 1 
                    x_base[i, d, k].VarHintPri = 50
    
    # 5. Objective function
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

    # 6. Constraints

    ## 6.1 Flow and Pairing Constraints
    ### 6.1.1 Linking constraint between decision to serve a passenger the routing out of their pickup node. If y[i] = 0, no outgoing route from pickup i; if y[i] = 1, at least one outgoing route from pickup i.
    for i in P_nodes:
        model_MILP_base.addConstr(
            gp.quicksum(x_base[i, j, k] for j in P_and_D for k in bus_idx) == y[i]
        ) # DO NOT REPLACE WITH .sum "*", WILL SUM OVER ALL NODES INSTEAD OF JUST P_AND_D

    ### 6.1.2 Previous scheduled request served
    for i in P_sched:
        for k in bus_idx:
            model_MILP_base.addConstr(
                x_base.sum(i, "*", k) == u_dict[k, i]
            )

    ### 6.1.3 Each request served at most once
    for i in P_wait:
        model_MILP_base.addConstr(
            x_base.sum(i, "*", "*") <= 1
        )

    ### 6.1.4 Ensures same vehicle visits pickup and drop-off nodes of same request
    for i in P_nodes:
        d = i + n_req
        for k in bus_idx:
            model_MILP_base.addConstr(
                x_base.sum(i, "*", k) - x_base.sum(d, "*", k) == 0
            )

    ### 6.1.5 Flow conservation constraints
    for i in P_nodes + D_nodes:
        for k in bus_idx:
            model_MILP_base.addConstr(
                x_base.sum(i, "*", k) - x_base.sum("*", i, k) == 0
            )

    ### 6.1.6 Each vehicle starts at its assigned start node and ends at its assigned end node
    for k in bus_idx:
        model_MILP_base.addConstr(
            x_base.sum(S_nodes[k], "*", k) - x_base.sum("*", S_nodes[k], k) == 1
        )
    for k in bus_idx:
        model_MILP_base.addConstr(
            x_base.sum(Z_nodes[k], "*", k) - x_base.sum("*", Z_nodes[k], k) == -1
        )

    ## 6.2 Capacity Constraints
    ### 6.2.1 Empty start constraint, such that buses start with 0 passengers on board
    for k in bus_idx:
        model_MILP_base.addConstr(
            q_k[S_nodes[k], k] == 0,
            name=f"force_empty_start_{k}"
        )

    ### 6.2.2 Capacity constraints
    for i in N:
        for j in N:
            for k in bus_idx:
                model_MILP_base.addConstr(
                    q_k[i, k] + Q[i] - params["M_cap"] * (1 - x_base[i, j, k]) <= q_k[j, k]
                )
                model_MILP_base.addConstr(
                    q_k[j, k] <= q_k[i, k] + Q[i] + params["M_cap"] * (1 - x_base[i, j, k])
                )

    ## 6.3 Time Window Constraints
    ### 6.3.1 Time propagation constraint
    for i in N:
        for j in N:
            for k in bus_idx:
                M_time_window_ij = l_dict[i] + params["max_late"] + params["w_max"] + s_dict[i] + t_dict[i, j] - e_dict[j]
                if M_time_window_ij < 0:
                    continue

                model_MILP_base.addConstr(
                    a_k[i, k] + w_k[i, k] + s_dict[i] + t_dict[i, j] - M_time_window_ij * (1 - x_base[i, j, k]) <= a_k[j, k]
                )

    ### 6.3.2 Time windows for earliest and latest arrival of a bus for pickups
    for i in P_nodes:
        for k in bus_idx:
            model_MILP_base.addConstr(
                e_dict[i] <= a_k[i, k] + w_k[i, k] + params["M_time_window"] * (1 - x_base.sum(i, '*', k))
            )
            model_MILP_base.addConstr(
                a_k[i, k] <= l_dict[i] + late_slack[i, k] + params["M_time_window"] * (1 - x_base.sum(i, '*', k))
            ) 

    ### 6.3.3 Time window for latest dropoff
    for i in D_nodes:
        for k in bus_idx:
            model_MILP_base.addConstr(
                a_k[i, k] <= l_dict[i] + late_slack[i, k] + params["M_time_window"] * (1 - x_base.sum(i, '*', k))
            )

    ### 6.3.4 Detour constraint
    for i in P_nodes:
        for k in bus_idx:
            model_MILP_base.addConstr(
                a_k[i+n_req, k] - (a_k[i, k] + w_k[i,k] + s_dict[i]) <= params["a_max"] * t_dict[i, i+n_req] 
                + params["M_time_window"] * (1 - x_base.sum(i, '*', k))
            )

    ### 6.3.5 Minimum travel time constraint between paired pickup and dropoff nodes
    for i in P_nodes:
        for k in bus_idx:
            model_MILP_base.addConstr(
                a_k[i, k] + w_k[i, k] + s_dict[i] + t_dict[i, i+n_req] <= a_k[i+n_req, k] 
                + params["M_time_window"] * (1 - x_base.sum(i, '*', k))
            )

    ## 6.4 Logical and Depot Constraints
    ### 6.4.1 Self-loop prevention constraint
    for i in N:
            for k in bus_idx:
                model_MILP_base.addConstr(
                    x_base[i, i, k] == 0
                )

    ### 6.4.2 Prevent vehicle from visiting its own start node after departing
    for k in bus_idx:
        model_MILP_base.addConstr(
            x_base.sum("*", S_nodes[k], k) == 0
        ) 

    ### 6.4.3 Prevent vehicles from visiting the start and end nodes of other vehicles
    for k in bus_idx: 
        for k_other in bus_idx:
            if k != k_other:
                model_MILP_base.addConstr(
                    x_base.sum('*', S_nodes[k_other], k) == 0
                )
                model_MILP_base.addConstr(
                    x_base.sum(S_nodes[k_other], '*', k) == 0
                )

                model_MILP_base.addConstr(
                    x_base.sum('*', Z_nodes[k_other], k) == 0
                )
                model_MILP_base.addConstr(
                    x_base.sum(Z_nodes[k_other], '*', k) == 0
                )

    ## 6.5 Ghost Request Constraints
    ### 6.5.1 Ghost Request Pickup Constraints 
    current_p_idx = len_R_wait 
    for k in bus_idx:
        ghosts_for_k = current_ghost_reqs.get(k, [])
        
        if ghosts_for_k:
            previous_node = S_nodes[k] 

            for _ in ghosts_for_k:
                ghost_p_node = current_p_idx
                
                model_MILP_base.addConstr(x_base[previous_node, ghost_p_node, k] == 1)
                
                previous_node = ghost_p_node
                current_p_idx += 1

    ### 6.5.2 Current Horizon Start Time Constraint
    for k in bus_idx:
        model_MILP_base.addConstr(
            e_dict[S_nodes[k]] <= a_k[S_nodes[k], k],
            name=f"force_start_time_{k}"
        )
    
    return model_MILP_base, x_base, q_k, w_k, a_k, y, late_slack#, mu

def run_simulation(params, stations, origin_stations, initial_K, bus_idx, bus_cost, get_travel_time_func):

    # 1. Algorithm Initialization
    ## 1.1 Set random seeds for reproductibility
    random.seed(params['seed'])
    np.random.seed(params['seed'])

    ## 1.2 Timestamp list generation for the simulation loop
    time_stamps = range(params["t_start"], params["t_end"] + 1, params["interval"])

    ## 1.3 Initial fleet configuration
    K = copy.deepcopy(initial_K)  

    ## 1.4 Request list generation
    ### 1.4.1 Initial request generation
    passenger_history = {}
    global_req_id = 0
    R_wait, req_t_p_wait, global_req_id = generate_requests(params, global_req_id, passenger_history, "initial", 0, origin_stations, stations)

    ### 1.4.2 Empty list generation for scheduled requests
    R_sched = []  # List to hold scheduled requests
    req_t_p_sched = []  # List to hold scheduled request times

    R = R_wait + R_sched  # All requests (waiting + scheduled)
    req_t_p = req_t_p_wait + req_t_p_sched  # Corresponding times
    n_req = len(R)

    ## 1.5 Ghost request tracking initialization for carryover
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

    for t in pbar:
        # 2. Timestep Update
        t_next = t + params["interval"]

        # 3. Passenger Management
        ## 3.1 Filter waiting passengers who haven't left the system
        R_wait, req_t_p_wait, n_abandonned = filter_waiting_passengers(R_wait, req_t_p_wait, params, t, passenger_history)

        ## 3.2 Update list of requests and corresponding times
        R = R_wait + R_sched
        req_t_p = req_t_p_wait + req_t_p_sched
        n_req = len(R)

        # 4. Network and Model Setup
        ## 4.1 Generate pickup nodes for all requests
        P_nodes = list(range(n_req))
        P_wait = list(range(len(R_wait)))
        P_sched = list(range(len(R_wait), len(R_wait) + len(R_sched)))

        ## 4.2 Dictionnary for carried over passengers scheduled in a previous iteration (u(k, node_id) = 1 if scheduled previously, 0 else)
        u_dict = {}
        for idx, node_id in enumerate(P_sched):
            assigned_k = u_dict_assignments_carryover[idx]
            for k in bus_idx:
                u_dict[k, node_id] = 1 if (assigned_k is not None and k == assigned_k) else 0

        ## 4.3 Dropoff nodes
        D_nodes = list(range(n_req, 2*n_req))

        ## 4.4 Virtual start and end nodes
        S_nodes = list(range(2*n_req, 2*n_req + len(K)))
        Z_nodes = list(range(2*n_req + len(K), 2*n_req + 2*len(K)))

        ## 4.5 Create node sets
        P_and_D = P_nodes + D_nodes
        N = P_nodes + D_nodes + S_nodes + Z_nodes

        ## 4.6 Create mapping from modeling to physical nodes
        P_loc = {i: R[i][1] for i in range(n_req)}
        D_loc = {i + n_req: R[i][2] for i in range(n_req)}
        S_loc = {S_nodes[k]: K[k][1] for k in range(len(K))}
        Z_loc = {Z_nodes[k]: K[k][2] for k in range(len(K))}

        node_to_loc = {}
        node_to_loc.update(P_loc)
        node_to_loc.update(D_loc)
        node_to_loc.update(S_loc)
        node_to_loc.update(Z_loc)

        ## 4.7 Calculate travel times between all nodes keyed by logical nodes
        t_dict = {(i,j): get_travel_time_func(node_to_loc[i], node_to_loc[j], t) for i in N for j in N}

        for i in N:
            for j in Z_nodes:
                t_dict[i, j] = 0 # No travel time between any node and depot

        ## 4.8 Boarding/alighting time
        s_dict = {i: params['board_alight_time'] if i in P_nodes + D_nodes else 0 for i in N}
        
        for idx, i in enumerate(P_nodes):
            if len(R[idx]) > 3 and R[idx][-1] == "ghost":
                s_dict[i] = 0  # No service time for in-transit passengers/ "ghosts"

        ## 4.9 Time windows for pickups and dropoffs
        tep, tlp, ted, tld = {}, {}, {}, {}

        for idx, i in enumerate(P_nodes):
            tep[i] = req_t_p[idx]  # earliest pickup time is request time
            tlp[i] = tep[i] + params['pax_max_wait'] # latest pickup time is request time + max wait

        for idx, i in enumerate(D_nodes):
            pickup_node = i - n_req
            travel_time = t_dict[pickup_node, i]
            ted[i] = 0      # No earliest dropoff time
            tld[i] = tlp[pickup_node] + travel_time * params['a_max'] # Latest dropoff time is latest pickup + max detour time
        
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
            e_dict[i] = params["t_end"] 
            l_dict[i] = 1440

        ## 4.10 Capacities and demand for each node
        Q = {}

        for i in P_nodes:
            Q[i] = R[i][0]
        
        for i in D_nodes:
            Q[i] = -R[i - n_req][0]

        for i in S_nodes + Z_nodes:
            Q[i] = 0

        Q_max = [vehicle[0] for vehicle in K]

        ub_dict = {(i, k): Q_max[k] for i in N for k in bus_idx}
        
        ## 4.11 Build the optimization model
        model_MILP_base, x_base, q_k, w_k, a_k, y, late_slack = model_construction(params, N, P_nodes, P_sched, P_wait, D_nodes, P_and_D, S_nodes, Z_nodes, 
                                                                                       Q, s_dict, t_dict, e_dict, l_dict, ub_dict, u_dict, n_req, 
                                                                                       current_ghost_reqs, len(R_wait), bus_idx, bus_cost)

        ## 4.12 Warm start
        if 'x_sol' in locals(): # If a solution exists from the previous t
            ### 4.12.1 Reset all start values to 0
            for v in x_base.values():
                v.Start = 0
            
            ### 4.12.2 Use u_dict from the previous loop to suggest which P-nodes belong to which vehicles.
            for (k, i_new) in u_dict:
                if u_dict[k, i_new] > 0.5:
                    if i_new in y:
                        y[i_new].Start = 1
        
        model_MILP_base._pbar = pbar
        model_MILP_base._last_sol_time = time.time()

        # 5. Optimization Execution
        ## 5.1 Optimize the Gurobi model
        model_MILP_base.optimize(my_callback)

        ## 5.2 Check for infeasibility and save feasibility report
        if model_MILP_base.status == gp.GRB.INFEASIBLE:
            print(f"\n🚨 MODEL INFEASIBLE AT t={t} 🚨")
            print("Computing IIS (Irreducible Inconsistent Subsystem)...")
            model_MILP_base.computeIIS()
            
            iis_filename = f"infeasible_t{t}.ilp" # Save contradicting rules to a text file
            model_MILP_base.write(iis_filename)
            print(f"✅ Saved conflicting constraints to {iis_filename}. Open this file to see the exact contradiction.")
            
            break
        
        ## 5.3 Fallback plan: If solver hits time limit, reject all new passengers and solve for already scheduled/in-transit passengers only
        if model_MILP_base.status == gp.GRB.TIME_LIMIT and model_MILP_base.SolCount == 0: 
            print(f"⚠️ t={t}: SOLVER OVERWHELMED. Triggering Fallback Plan...")
            
            for i in P_wait:
                model_MILP_base.addConstr(y[i] == 0, name=f"panic_reject_{i}")
            
            model_MILP_base.Params.TimeLimit = 120
            print("   -> Re-routing only active/ghost passengers...")
            model_MILP_base.optimize()

        # 6. System State Update
        if model_MILP_base.status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT] and model_MILP_base.SolCount > 0:
            # 6.1 Extract the solution values for decision variables
            x_sol = model_MILP_base.getAttr('X', x_base)
            y_sol = model_MILP_base.getAttr('X', y)
            q_sol = model_MILP_base.getAttr('X', q_k)
            w_sol = model_MILP_base.getAttr('X', w_k)
            a_sol = model_MILP_base.getAttr('X', a_k)

            # 6.2 Calculate rejection rate
            presented = len(P_wait)
            rejected = sum(1 for i in P_wait if y_sol[i] < 0.5)

            opt_runtime = model_MILP_base.Runtime
            opt_nodecount = model_MILP_base.NodeCount
            opt_status = model_MILP_base.Status

            try:
                opt_mipgap = model_MILP_base.MIPGap
            except AttributeError:
                opt_mipgap = 0.0

            pbar.set_postfix({
                "Last_t": t,
                "Final_Obj": f"{model_MILP_base.ObjVal:.1f}",
                "Rejected": f"{rejected}/{presented}"
            })

            # 6.3 Update the history statistics and bus route statistics
            history_stats["time_step"].append(t)
            history_stats["new_reqs_presented"].append(presented)
            history_stats["reqs_rejected"].append(rejected)
            history_stats["reqs_abandonned"].append(n_abandonned)
            history_stats["obj_cost"].append(model_MILP_base.ObjVal)
            history_stats["solve_time_seconds"].append(opt_runtime)
            history_stats["mip_gap"].append(opt_mipgap)
            history_stats["node_count"].append(opt_nodecount)
            history_stats["solve_status"].append(opt_status)

            ## 6.4 Extract and save the routes for each bus
            for k in bus_idx:
                route_for_k = []
                curr_node = S_nodes[k]
                
                while curr_node != Z_nodes[k]:
                    ### 6.4.1 Determine if current node is a pickup or dropoff, to correctly calculate arrival time and passenger load
                    is_pickup = curr_node < n_req 
                    actual_arrival = a_sol.get((curr_node, k), t)

                    ### 6.4.2 Calculate action time
                    if is_pickup:
                        action_time = max(actual_arrival, req_t_p[curr_node]) + w_sol.get((curr_node, k), 0)
                    else:
                        action_time = actual_arrival + w_sol.get((curr_node, k), 0)

                    ### 6.4.3 Append the current node's information to the route
                    route_for_k.append({
                        "logical node": curr_node,
                        "location": node_to_loc[curr_node],
                        "arrival_time": action_time,
                        "passenger_load": q_sol.get((curr_node, k), 0) + Q.get(curr_node, 0)
                    })

                    ### 6.4.4 Find the next node in the route
                    next_node = None
                    for j in N:
                        if x_sol[curr_node, j, k] > 0.5:
                            next_node = j
                            break
                    
                    ### 6.4.5 Move to the next node or end if no next node is found
                    if next_node is not None:
                        curr_node = next_node
                    else:
                        break
                
                ### 6.4.6 Save the extracted route for bus k at interval t in the history
                history_routes[k].append({
                    "interval": t,
                    "route": route_for_k
                })

            ## 6.5 Update the system state for the next iteration
            next_R_sched, next_req_t_p_sched, u_dict_carryover_k = [], [], []

            ### 6.5.1 Create ghost requests for in-transit passengers, to correctly carry them over to the next iteration
            next_ghost_reqs = {k: [] for k in bus_idx}
            next_ghost_times = {k: [] for k in bus_idx}

            for idx, i in enumerate(P_nodes):
                d_node = i + n_req
                req_id = R[idx][4]
                
                ### 6.5.2 Check whether request was scheduled in a previous iteration or is newly scheduled in this iteration
                is_active = (i in P_sched) or (i in P_wait and y_sol[i] > 0.5)
                
                if is_active:
                    ### 6.5.3 Find when the drop-off happens
                    d_time = -1
                    for k in bus_idx:
                        if sum(x_sol[d_node, j, k] for j in N) > 0.5: # Check if bus k visits drop-off node
                            d_time = a_sol[d_node, k] + w_sol[d_node, k] # Extract arrival time at drop-off node
                            break
                    
                    ### 6.5.4 If drop-off happens after the current time step, it carries over
                    if d_time > t + params["interval"]:
                        p_time = -1
                        assigned_k = None
                        for k in bus_idx:
                            if sum(x_sol[i, j, k] for j in N) > 0.5:
                                p_time = max(a_sol[i, k], req_t_p[idx]) + w_sol.get((i, k), 0)
                                assigned_k = k
                                break
                        
                        carried_time = req_t_p[idx] # Original pickup time request

                        ### 6.5.5 If pickup already happened, move their origin to the vehicle's location and set pickup time to help
                        if p_time <= t + params["interval"] and assigned_k is not None:
                            passenger_history[req_id]["status"] = "in_transit"
                            if passenger_history[req_id]["time_picked_up"] is None:
                                passenger_history[req_id]["time_picked_up"] = p_time
                            passenger_history[req_id]["assigned_bus"] = assigned_k
                            ghost_req = R[idx].copy()
                            
                            ### 6.5.6 Update origin to vehicle's current node. 
                            last_visited = S_nodes[assigned_k]
                            max_a = -1
                            for n_idx in N:
                                if sum(x_sol[n_idx, j, assigned_k] for j in N) > 0.5:
                                    if a_sol[n_idx, assigned_k] <= t + params["interval"] and a_sol[n_idx, assigned_k] > max_a:
                                        max_a = a_sol[n_idx, assigned_k]
                                        last_visited = n_idx

                            ghost_req[1] = node_to_loc[last_visited]
                            ghost_req.append("ghost") 
                            
                            next_ghost_reqs[assigned_k].append(ghost_req)
                            next_ghost_times[assigned_k].append(carried_time)
                        else:
                            ### 6.5.7 Scheduled but not picked up yet (normal carryover)
                            next_R_sched.append(R[idx].copy())
                            next_req_t_p_sched.append(carried_time)
                            u_dict_carryover_k.append(assigned_k)

                    ### 6.5.8 If drop-off happens within the current time step, we consider the request completed and do not carry it over
                    else:
                        passenger_history[req_id]["status"] = "completed"
                        if passenger_history[req_id]["time_dropped_off"] is None:
                            passenger_history[req_id]["time_dropped_off"] = d_time

                        for k in bus_idx:
                            if sum(x_sol[d_node, j, k] for j in N) > 0.5:
                                passenger_history[req_id]["assigned_bus"] = k
                            if sum(x_sol[i, j, k] for j in N) > 0.5:
                                if passenger_history[req_id]["time_picked_up"] is None:
                                    passenger_history[req_id]["time_picked_up"] = max(a_sol[i, k], req_t_p[idx]) + w_sol.get((i, k), 0) # NEW

            ### 6.5.9 Sort the next_R_sched so that ghost requests are in front of their corresponding normal requests, to ensure they get assigned to the same vehicle in the next iteration
            final_next_R_sched = []
            final_next_req_t_p_sched = []
            final_u_dict_assignments = []

            ### 6.5.10 Add all ghosts in strict vehicle order
            for k in bus_idx:
                for g_req, g_time in zip(next_ghost_reqs[k], next_ghost_times[k]):
                    final_next_R_sched.append(g_req)
                    final_next_req_t_p_sched.append(g_time)
                    final_u_dict_assignments.append(k) # Lock ghost to this vehicle

            ### 6.5.11 Add normal carryovers
            for n_req_item, n_time, n_k in zip(next_R_sched, next_req_t_p_sched, u_dict_carryover_k):
                final_next_R_sched.append(n_req_item)
                final_next_req_t_p_sched.append(n_time)
                final_u_dict_assignments.append(n_k)

            ## 6.6 Replace the old arrays with the newly sorted ones
            next_R_sched = final_next_R_sched
            next_req_t_p_sched = final_next_req_t_p_sched

            ## 6.7 Update Vehicle Positions (K)
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

            ## 6.8 Generate new requests for the next iteration
            new_R_wait, new_req_t_p_wait, global_req_id = generate_requests(params, global_req_id, passenger_history, "new_interval", t_next, origin_stations=origin_stations, stations=stations)
            
            ## 6.9 Rebuild u_dict with new logical indices 
            unserved_R_wait       = [R_wait[i] for i in P_wait if y_sol[i] < 0.5]
            unserved_req_t_p_wait = [req_t_p_wait[i] for i in P_wait if y_sol[i] < 0.5]
            
            R_wait        = new_R_wait + unserved_R_wait
            req_t_p_wait  = new_req_t_p_wait + unserved_req_t_p_wait
            R_sched       = next_R_sched
            req_t_p_sched = next_req_t_p_sched
            
            R             = R_wait + R_sched
            req_t_p       = req_t_p_wait + req_t_p_sched
            n_req         = len(R)

            u_dict_assignments_carryover = final_u_dict_assignments

            current_ghost_reqs = next_ghost_reqs
 
            history_stats['in_transit_carried_over'].append(len(next_R_sched))

        else:
            print(f"❌ FATAL: Even the fallback plan failed to find a solution at t={t}. Halting simulation.")
            break
        
        # 7. Memory Cleanup
        if 'model_MILP_base' in locals():
            model_MILP_base.dispose()
        
        del x_base, q_k, w_k, a_k, y, late_slack
        
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
    
    clean_params = {}
    for k, v in SIM_PARAMS.items():
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
        feasibility_status = "No" 
    else:
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