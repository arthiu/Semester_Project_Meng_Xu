import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_executive_summary(df_pax, df_routes):
    # 1. Normalize passenger status strings (handles 'Served', 'served', 'completed', 'Abandoned')
    if not df_pax.empty and 'status' in df_pax.columns:
        df_pax['status_clean'] = df_pax['status'].astype(str).str.strip().str.lower()
    else:
        print("No passenger data available for summary.")
        return

    # 2. Calculate Passenger Metrics
    total_pax = len(df_pax)
    served_df = df_pax[df_pax['status_clean'].isin(['served', 'completed'])]
    abandoned_df = df_pax[df_pax['status_clean'] == 'abandoned']
    
    served_count = len(served_df)
    abandoned_count = len(abandoned_df)
    
    service_rate = (served_count / total_pax * 100) if total_pax > 0 else 0
    abandon_rate = (abandoned_count / total_pax * 100) if total_pax > 0 else 0
    
    # Clip negative waits/travels to 0 just in case
    avg_wait = np.clip(served_df['wait_time_mins'], 0, None).mean() if not served_df.empty else 0
    avg_travel = np.clip(served_df['travel_time_mins'], 0, None).mean() if not served_df.empty else 0

    # 3. Calculate Fleet Metrics
    avg_occupancy = df_routes['passenger_load'].mean() if not df_routes.empty else 0
    
    # Calculate average time/distance travelled per bus (Max arrival time - min arrival time per bus)
    if not df_routes.empty:
        active_times = df_routes.groupby('bus_id')['arrival_time'].max() - df_routes.groupby('bus_id')['arrival_time'].min()
        avg_bus_active_time = active_times.mean()
    else:
        avg_bus_active_time = 0

    # 4. Build and Print the Summary Table
    summary_data = {
        "Metric": [
            "Total Requests Generated",
            "Passengers Serviced",
            "Service Rate (%)",
            "Passengers Abandoned",
            "Abandonment Rate (%)",
            "Avg. Passenger Wait Time (mins)",
            "Avg. Passenger Travel Time (mins)",
            "Avg. Bus Occupancy (Pax/Bus)",
            "Avg. Bus Active Travel Time (mins)"
        ],
        "Value": [
            f"{total_pax}",
            f"{served_count}",
            f"{service_rate:.1f}%",
            f"{abandoned_count}",
            f"{abandon_rate:.1f}%",
            f"{avg_wait:.1f}",
            f"{avg_travel:.1f}",
            f"{avg_occupancy:.1f}",
            f"{avg_bus_active_time:.1f}"
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    
    print("\n" + "="*50)
    print(" 📊 EXECUTIVE SIMULATION SUMMARY")
    print("="*50)
    # Print without the index for a cleaner look
    print(df_summary.to_string(index=False, justify='left'))
    print("="*50 + "\n")

def get_realized_route(df_routes, bus_id):
    raw_route = []
    
    # Filter down to just this bus and sort chronologically
    bus_df = df_routes[df_routes['bus_id'] == bus_id].sort_values(['decision_interval', 'step_sequence'])
    
    if bus_df.empty:
        return []
    
    # Get the unique optimization intervals this bus participated in
    intervals = sorted(bus_df['decision_interval'].unique())
    
    for i, current_t in enumerate(intervals):
        # Determine the cutoff time for this specific plan
        next_t = intervals[i+1] if i + 1 < len(intervals) else float('inf') 
            
        # Isolate the plan generated at current_t
        snapshot = bus_df[bus_df['decision_interval'] == current_t]
        
        for _, row in snapshot.iterrows():
            arr_time = row['arrival_time']
            
            # Keep nodes visited BEFORE the next optimization triggers
            if current_t <= arr_time < next_t:
                raw_route.append({
                    "executed_during_interval": current_t,
                    "logical_node": row["logical_node"],
                    "location": row["location"],
                    "arrival_time": round(arr_time, 2),
                    "passenger_load": round(row["passenger_load"])
                })
                
    # --- CLEANUP BLOCK ---
    # Remove the 0-load dips caused by S_nodes and Ghost carry-overs
    cleaned_route = []
    for i in range(len(raw_route)):
        curr = raw_route[i]
        is_artifact = False
        
        # If the NEXT event happens at the exact same time AND location,
        # then the CURRENT event is just an intermediate MILP step (like an S_node).
        # We skip it so we only capture the final, true passenger load.
        if i < len(raw_route) - 1:
            nxt = raw_route[i+1]
            if abs(curr['arrival_time'] - nxt['arrival_time']) < 0.001 and curr['location'] == nxt['location']:
                is_artifact = True
                
        if not is_artifact:
            cleaned_route.append(curr)
            
    return cleaned_route
    
def plot_bus_load_evolution(realized_route, bus_id):
    if not realized_route:
        print("No route data to plot.")
        return

    # Extract data for plotting
    times = [step['arrival_time'] for step in realized_route]
    loads = [step['passenger_load'] for step in realized_route]
    locations = [step['location'] for step in realized_route]

    plt.figure(figsize=(14, 6))
    
    # Step plot for load evolution (where='post' means the load changes immediately after the arrival time)
    plt.step(times, loads, where='post', color='royalblue', linewidth=2.5, alpha=0.8, label=f'Bus {bus_id} Load')
    
    # Track labels added to legend so we don't duplicate them
    added_to_legend = set()

    # Plot markers and annotations for pick-ups and drop-offs
    for i in range(len(realized_route)):
        t = times[i]
        l = loads[i]
        loc = locations[i]
        
        if i == 0:
            # Start node
            plt.plot(t, l, marker='s', color='black', markersize=8, label='Start Node')
            added_to_legend.add('Start Node')
            plt.text(t, l + 0.15, f"Start\nt={t:.1f}", ha='left', va='bottom', fontsize=9, fontweight='bold')
        else:
            prev_l = loads[i-1]
            diff = l - prev_l
            
            if diff > 0:  # Pick-up
                label = 'Pick-up' if 'Pick-up' not in added_to_legend else ""
                plt.plot(t, l, marker='^', color='forestgreen', markersize=10, zorder=5, label=label)
                if label: added_to_legend.add('Pick-up')
                
                plt.text(t, l + 0.15, f"+{int(diff)} pax\n@{t:.1f}", ha='center', va='bottom', fontsize=9, color='darkgreen')
                
            elif diff < 0:  # Drop-off
                label = 'Drop-off' if 'Drop-off' not in added_to_legend else ""
                plt.plot(t, l, marker='v', color='crimson', markersize=10, zorder=5, label=label)
                if label: added_to_legend.add('Drop-off')
                
                plt.text(t, l - 0.15, f"{int(diff)} pax\n@{t:.1f}", ha='center', va='top', fontsize=9, color='darkred')
                
            else:  # Visited a node but no load change (e.g. End depot or zero-load ghost transition)
                label = 'Empty Stop / Depot' if 'Empty Stop / Depot' not in added_to_legend else ""
                plt.plot(t, l, marker='o', color='gray', markersize=6, zorder=5, label=label)
                if label: added_to_legend.add('Empty Stop / Depot')

    # Formatting the plot
    plt.title(f"Passenger Load Evolution Timeline: Bus {bus_id}", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Simulation Time (minutes)", fontsize=12, fontweight='bold')
    plt.ylabel("Passenger Load (Number of People)", fontsize=12, fontweight='bold')
    
    # Ensure the Y-axis only shows whole numbers (you can't have half a passenger)
    max_load = max(loads) if loads else 5
    plt.yticks(range(0, int(max_load) + 2))
    
    # Add a subtle grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.grid(True, axis='x', linestyle=':', alpha=0.3)
    
    # Cleanup legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Filter out empty labels
    valid_handles = [h for h, l in zip(handles, labels) if l]
    valid_labels = [l for l in labels if l]
    plt.legend(valid_handles, valid_labels, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    # Add a horizontal line at Y=0 for visual grounding
    plt.axhline(0, color='black', linewidth=1)

    plt.tight_layout()
    plt.show()