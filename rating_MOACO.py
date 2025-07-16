import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# Mở file output để lưu kết quả
output_file = open("MOACO_output/output_updated_with_improvements.txt", "w", encoding="utf-8")

def print_and_save(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=output_file)

# ===================== Data Loading and Preparation =====================
coordinates = pd.read_csv("data/coordinates_3.csv")
id_to_idx = {id_val: idx for idx, id_val in enumerate(coordinates['id'])}
idx_to_id = {idx: id_val for idx, id_val in enumerate(coordinates['id'])}
num_nodes = len(coordinates)

station_idxs = [
    id_to_idx[id_val]
    for id_val in coordinates[coordinates['name'].str.lower().str.contains("railway station")]['id'].tolist()
]

poi_data = coordinates[~coordinates['id'].isin([coordinates['id'][idx] for idx in station_idxs])]
top_10_pois = poi_data.nlargest(10, 'rating')
top_10_ids = top_10_pois['id'].tolist()
top_10_idxs = [id_to_idx[id_val] for id_val in top_10_ids]


distance_data = pd.read_csv("data/formatted_distance_matrix_taxi_with_service_cost.csv")
distance_matrix_np = np.full((num_nodes, num_nodes), np.inf)
time_matrix_np = np.full((num_nodes, num_nodes), np.inf)
cost_matrix_np = np.full((num_nodes, num_nodes), np.inf)

for idx, row in distance_data.iterrows():
    from_idx = id_to_idx[row['from_id']]
    to_idx = id_to_idx[row['to_id']]
    distance_matrix_np[from_idx, to_idx] = row['distance_km']
    time_matrix_np[from_idx, to_idx] = row['time_min']
    cost_matrix_np[from_idx, to_idx] = row['cost_baht']

# ===================== ACO Parameters and Archive =====================
num_ants = 20  # Tăng số lượng kiến
num_iterations = 100  # Tăng số vòng lặp
evaporation_rate = 0.5
alpha = 1
beta = 2
Q = 100
MAX_ARCHIVE_SIZE = 200  # Tăng kích thước archive

# Danh sách lưu trữ kết quả cho mỗi ga tàu
all_pareto_archives = []

# ===================== Dominance and Archive Update =====================
def dominates(a, b):
    return (
        a['time'] <= b['time'] and a['cost'] <= b['cost'] and
        (a['time'] < b['time'] or a['cost'] < b['cost'])
    )

def update_archive(sol, archive):
    archive = [x for x in archive if not dominates(sol, x)]
    if not any(dominates(x, sol) for x in archive):
        archive.append(sol)
    temp_archive = []
    for s1 in archive:
        is_dominated = False
        for s2 in archive:
            if s1 != s2 and dominates(s2, s1):
                is_dominated = True
                break
        if not is_dominated:
            temp_archive.append(s1)
    archive = temp_archive
    if len(archive) > MAX_ARCHIVE_SIZE:
        # Cắt bớt archive dựa trên khoảng cách thời gian để duy trì đa dạng
        archive = sorted(archive, key=lambda s: s['time'])
        selected = [archive[0]]
        for i in range(1, len(archive)):
            if abs(archive[i]['time'] - selected[-1]['time']) > 5:
                selected.append(archive[i])
            if len(selected) >= MAX_ARCHIVE_SIZE:
                break
        archive = selected
    return archive

# ===================== Cost and Heuristic Functions =====================
def calculate_cost(route_idxs: list[int]) -> tuple[float, float, float]:
    total_distance = 0
    total_time = 0
    total_cost = 0
    for i in range(len(route_idxs) - 1):
        from_idx = route_idxs[i]
        to_idx = route_idxs[i + 1]
        total_distance += distance_matrix_np[from_idx, to_idx]
        total_time += time_matrix_np[from_idx, to_idx]
        total_cost += cost_matrix_np[from_idx, to_idx]
    return total_distance, total_time, total_cost

def heuristic(from_idx, to_idx, w_dist=1.0, w_time=1.0, w_cost=1.0):
    dist = distance_matrix_np[from_idx, to_idx]
    time = time_matrix_np[from_idx, to_idx]
    cost = cost_matrix_np[from_idx, to_idx]
    if dist == np.inf or time == np.inf or cost == np.inf or dist == 0:
        return 0.0001
    return 1 / (w_dist * dist + w_time * time + w_cost * cost)

# ===================== ACO Main Algorithm =====================
def aco_tsp(station_idx, num_ants, num_iterations, alpha, beta, evaporation_rate, Q):
    pheromones = {
        'time': np.ones((num_nodes, num_nodes)) * 0.1,
        'cost': np.ones((num_nodes, num_nodes)) * 0.1
    }
    archive = []
    all_solutions = []
    best_overall_route = None
    min_combined_objective = float('inf')

    for iteration in range(num_iterations):
        iteration_solutions = []

        for ant in range(num_ants):
            # Trọng số ngẫu nhiên cho mỗi kiến
            w_time = random.uniform(0, 1)
            w_cost = 1 - w_time
            route_idxs = [station_idx]
            visited_idxs = set(route_idxs)
            remaining_pois = list(top_10_idxs)
            while remaining_pois:
                current_idx = route_idxs[-1]
                possible_next_nodes = [node_idx for node_idx in remaining_pois if node_idx not in visited_idxs]
                if not possible_next_nodes:
                    break
                probabilities = []
                total_numerator = 0
                numerator_values = []
                for next_idx in possible_next_nodes:
                    combined_pheromone = (
                        w_time * pheromones['time'][current_idx, next_idx] +
                        w_cost * pheromones['cost'][current_idx, next_idx]
                    ) ** alpha
                    heuristic_value = heuristic(current_idx, next_idx) ** beta
                    numerator = combined_pheromone * heuristic_value
                    numerator_values.append(numerator)
                    total_numerator += numerator
                if total_numerator == 0:
                    next_idx = random.choice(possible_next_nodes)
                else:
                    probabilities = [val / total_numerator for val in numerator_values]
                    probabilities_sum = sum(probabilities)
                    if probabilities_sum != 0:
                        probabilities = [p / probabilities_sum for p in probabilities]
                    else:
                        next_idx = random.choice(possible_next_nodes)
                    next_idx = np.random.choice(possible_next_nodes, p=probabilities)
                route_idxs.append(next_idx)
                visited_idxs.add(next_idx)
                remaining_pois.remove(next_idx)

            if len(route_idxs) == len(top_10_idxs) + 1:
                route_idxs.append(station_idx)
                distance, time, cost = calculate_cost(route_idxs)
                if time == np.inf or cost == np.inf:
                    continue
                sol = {'route_idxs': route_idxs, 'time': time, 'cost': cost}
                iteration_solutions.append(sol)
                all_solutions.append(sol)
                archive = update_archive(sol, archive)
                combined_objective = time + cost
                if combined_objective < min_combined_objective:
                    min_combined_objective = combined_objective
                    best_overall_route = [idx_to_id[i] for i in route_idxs]

        pheromones['time'] *= (1 - evaporation_rate)
        pheromones['cost'] *= (1 - evaporation_rate)
        for sol in archive:
            time_pheromone_deposit = Q / sol['time'] if sol['time'] > 0 else Q
            cost_pheromone_deposit = Q / sol['cost'] if sol['cost'] > 0 else Q
            for i in range(len(sol['route_idxs']) - 1):
                from_idx = sol['route_idxs'][i]
                to_idx = sol['route_idxs'][i + 1]
                pheromones['time'][from_idx, to_idx] += time_pheromone_deposit
                pheromones['cost'][from_idx, to_idx] += cost_pheromone_deposit
                pheromones['time'][from_idx, to_idx] = min(pheromones['time'][from_idx, to_idx], 1000.0)
                pheromones['cost'][from_idx, to_idx] = min(pheromones['cost'][from_idx, to_idx], 1000.0)

        if archive:
            current_best_time = min(s['time'] for s in archive)
            current_best_cost = min(s['cost'] for s in archive)
            print_and_save(f"Station {station_idx} - Iteration {iteration + 1}: Best Time = {current_best_time:.2f}, Best Cost = {current_best_cost:.2f}")
        else:
            print_and_save(f"Station {station_idx} - Iteration {iteration + 1}: Archive is empty.")
        print_and_save(f"Station {station_idx} - Archive size: {len(archive)}")

    return archive, all_solutions, best_overall_route

# ===================== Run ACO for each station =====================
for station_idx in station_idxs:
    print_and_save(f"\nRunning MOACO for station {station_idx}...")
    archive, all_solutions, best_route_ids = aco_tsp(
        station_idx, num_ants, num_iterations, alpha, beta, evaporation_rate, Q
    )
    all_pareto_archives.append((station_idx, archive, all_solutions, best_route_ids))

# ===================== Plot Pareto Fronts Before 2-Opt for Each Station =====================
for station_idx, archive, _, _ in all_pareto_archives:
    if archive:
        times = [sol['time'] for sol in archive]
        costs = [sol['cost'] for sol in archive]
        plt.figure(figsize=(8, 6))
        plt.scatter(times, costs, label=f'Station {station_idx} (Before 2-Opt)')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Cost (Baht)')
        plt.title(f'Pareto Front for Station {station_idx} Before 2-Opt (MOACO)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"MOACO_output/pareto_front_station_{station_idx}_before_2opt.png")
        plt.close()

# ===================== Plot Combined Pareto Fronts Before 2-Opt =====================
plt.figure(figsize=(10, 7))
for station_idx, archive, _, _ in all_pareto_archives:
    if archive:
        times = [sol['time'] for sol in archive]
        costs = [sol['cost'] for sol in archive]
        plt.scatter(times, costs, label=f'Station {station_idx}')
plt.xlabel('Time (minutes)')
plt.ylabel('Cost (Baht)')
plt.title('Pareto Fronts for Different Stations Before 2-Opt (MOACO)')
plt.grid(True)
plt.legend()
plt.savefig("MOACO_output/pareto_fronts_before_2opt.png")
plt.close()

# ===================== Print Best Solutions Before 2-Opt =====================
print_and_save("\n--- Best Solutions for Each Station (Before 2-Opt) ---")
best_solutions = []
for station_idx, archive, _, best_route_ids in all_pareto_archives:
    if archive:
        best_sol = min(archive, key=lambda s: s['time'] + s['cost'])
        best_time = best_sol['time']
        best_cost = best_sol['cost']
        best_route_ids = [idx_to_id[i] for i in best_sol['route_idxs']]
        best_solutions.append({
            'station_idx': station_idx,
            'route_ids': best_route_ids,
            'time': best_time,
            'cost': best_cost,
            'route_idxs': best_sol['route_idxs']
        })
        print_and_save(f"\nStation {station_idx}:")
        print_and_save(f"  Route (IDs): {best_route_ids}")
        print_and_save(f"  Time: {best_time:.2f} minutes")
        print_and_save(f"  Cost: {best_cost:.2f} Baht")

# ===================== Multi-Objective 2-Opt Refinement =====================
def two_opt_fixed_endpoints_multiobjective(route_idxs, max_iterations=50):
    start_idx, end_idx = route_idxs[0], route_idxs[-1]
    middle = route_idxs[1:-1]
    archive = []

    _, initial_time, initial_cost = calculate_cost(route_idxs)
    if initial_time != np.inf and initial_cost != np.inf:
        archive.append({'route_idxs': route_idxs, 'time': initial_time, 'cost': initial_cost})

    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(len(middle) - 1):
            for j in range(i + 1, len(middle)):
                new_middle = middle[:i] + middle[i:j + 1][::-1] + middle[j + 1:]
                new_route = [start_idx] + new_middle + [end_idx]
                _, new_time, new_cost = calculate_cost(new_route)
                if new_time == np.inf or new_cost == np.inf:
                    continue
                new_sol = {'route_idxs': new_route, 'time': new_time, 'cost': new_cost}
                prev_archive_len = len(archive)
                archive = update_archive(new_sol, archive)
                if len(archive) > prev_archive_len:
                    middle = new_middle
                    improved = True
                    break  # để tránh nhiều cập nhật cùng lúc
            if improved:
                break
    return archive

# ===================== Run Multi-Objective 2-Opt and Print Results =====================
print_and_save("\n--- Optimized Solutions after Multi-Objective 2-Opt ---")
optimized_solutions = []
for sol in best_solutions:
    archive_2opt = two_opt_fixed_endpoints_multiobjective(sol['route_idxs'])
    if archive_2opt:
        # Chọn giải pháp tốt nhất từ archive 2-opt (dựa trên tổng time + cost)
        best_2opt_sol = min(archive_2opt, key=lambda s: s['time'] + s['cost'])
        opt_distance, opt_time, opt_cost = calculate_cost(best_2opt_sol['route_idxs'])
        optimized_route_ids = [idx_to_id[idx] for idx in best_2opt_sol['route_idxs']]
        optimized_solutions.append({
            'station_idx': sol['station_idx'],
            'route_ids': optimized_route_ids,
            'time': opt_time,
            'cost': opt_cost,
            'route_idxs': best_2opt_sol['route_idxs']
        })
        print_and_save(f"\nStation {sol['station_idx']} (Optimized):")
        print_and_save(f"  Route (IDs): {optimized_route_ids}")
        print_and_save(f"  Distance: {opt_distance:.2f} km")
        print_and_save(f"  Time: {opt_time:.2f} minutes")
        print_and_save(f"  Cost: {opt_cost:.2f} Baht")
        print_and_save(f"  Number of solutions in 2-opt archive: {len(archive_2opt)}")
    else:
        print_and_save(f"\nStation {sol['station_idx']} (Optimized): No valid solutions found after 2-opt.")

# ===================== Plot Optimized Pareto Fronts After 2-Opt for Each Station =====================
for sol in optimized_solutions:
    station_idx = sol['station_idx']
    archive_2opt = two_opt_fixed_endpoints_multiobjective(sol['route_idxs'])
    if archive_2opt:
        times = [s['time'] for s in archive_2opt]
        costs = [s['cost'] for s in archive_2opt]
        plt.figure(figsize=(8, 6))
        plt.scatter(times, costs, label=f'Station {station_idx} (After 2-Opt)')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Cost (Baht)')
        plt.title(f'Pareto Front for Station {station_idx} After Multi-Objective 2-Opt (MOACO)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"MOACO_output/pareto_front_station_{station_idx}_after_2opt.png")
        plt.close()

# ===================== Plot Combined Pareto Fronts Before and After 2-Opt =====================
plt.figure(figsize=(10, 7))
for station_idx, archive, _, _ in all_pareto_archives:
    if archive:
        times = [sol['time'] for sol in archive]
        costs = [sol['cost'] for sol in archive]
        plt.scatter(times, costs, label=f'Station {station_idx} (Before 2-Opt)', marker='o')
for sol in optimized_solutions:
    plt.scatter(sol['time'], sol['cost'], label=f'Station {sol["station_idx"]} (After 2-Opt)', marker='x')
plt.xlabel('Time (minutes)')
plt.ylabel('Cost (Baht)')
plt.title('Pareto Fronts Before and After Multi-Objective 2-Opt (MOACO)')
plt.grid(True)
plt.legend()
plt.savefig("MOACO_output/pareto_fronts_before_after_2opt.png")
plt.close()

# ===================== Close Output File =====================
output_file.close()