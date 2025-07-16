import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

output_file = open("output/output.txt", "w", encoding="utf-8")

def print_and_save(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=output_file)

# ===================== Data Loading and Preparation =====================

# Đọc dữ liệu tọa độ
coordinates = pd.read_csv("coordinates_3.csv")

# Ánh xạ ID của điểm đến chỉ số liên tục và ngược lại
id_to_idx = {id_val: idx for idx, id_val in enumerate(coordinates['id'])}
idx_to_id = {idx: id_val for idx, id_val in enumerate(coordinates['id'])}
num_nodes = len(coordinates)

# Lọc các điểm là ga tàu dựa vào chuỗi "railway station" trong cột name (không phân biệt hoa thường)
station_idxs = [
    id_to_idx[id_val]
    for id_val in coordinates[coordinates['name'].str.lower().str.contains("railway station")]['id'].tolist()
]

# Đọc ma trận khoảng cách và chi phí

# Tạo ma trận khoảng cách, thời gian và chi phí từ dữ liệu CSV dưới dạng NumPy arrays
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

num_ants = 10
num_iterations = 100
evaporation_rate = 0.5
alpha = 1  # Hệ số pheromone
beta = 2   # Hệ số độ hấp dẫn
Q = 100    # Lượng pheromone

all_solutions = []
archive = []
MAX_ARCHIVE_SIZE = 100

# ===================== Dominance and Archive Update =====================

def dominates(a, b):
    return (
        a['time'] <= b['time'] and a['cost'] <= b['cost'] and
        (a['time'] < b['time'] or a['cost'] < b['cost'])
    )

def update_archive(sol):
    global archive
    archive = [x for x in archive if not dominates(sol, x)]
    if not any(dominates(x, sol) for x in archive):
        archive.append(sol)
    # Remove dominated solutions
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
    # Limit archive size
    if len(archive) > MAX_ARCHIVE_SIZE:
        archive = sorted(archive, key=lambda s: s['time'] + s['cost'])[:MAX_ARCHIVE_SIZE]

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

# ===================== Pheromone Initialization =====================

pheromones = {
    'time': np.ones((num_nodes, num_nodes)) * 0.1,
    'cost': np.ones((num_nodes, num_nodes)) * 0.1
}

# ===================== Ant Route Selection =====================

def select_next_node(current_idx, visited_idxs, pheromone_matrices, alpha, beta, w_time=0.5, w_cost=0.5):
    probabilities = []
    possible_next_nodes = [node_idx for node_idx in range(num_nodes) if node_idx not in visited_idxs]
    if not possible_next_nodes:
        return -1

    total_numerator = 0
    numerator_values = []

    for next_idx in possible_next_nodes:
        combined_pheromone = (
            w_time * pheromone_matrices['time'][current_idx, next_idx] +
            w_cost * pheromone_matrices['cost'][current_idx, next_idx]
        ) ** alpha
        heuristic_value = heuristic(current_idx, next_idx) ** beta
        numerator = combined_pheromone * heuristic_value
        numerator_values.append(numerator)
        total_numerator += numerator

    if total_numerator == 0:
        return random.choice(possible_next_nodes)
    probabilities = [val / total_numerator for val in numerator_values]
    probabilities_sum = sum(probabilities)
    if probabilities_sum != 0:
        probabilities = [p / probabilities_sum for p in probabilities]
    else:
        return random.choice(possible_next_nodes)
    return np.random.choice(possible_next_nodes, p=probabilities)

# ===================== ACO Main Algorithm =====================

def aco_tsp(num_ants, num_iterations, alpha, beta, evaporation_rate, Q):
    best_overall_route = None
    min_combined_objective = float('inf')

    for iteration in range(num_iterations):
        iteration_solutions = []

        for ant in range(num_ants):
            start_idx = random.choice(station_idxs)
            route_idxs = [start_idx]
            visited_idxs = set(route_idxs)

            while len(route_idxs) < num_nodes:
                current_idx = route_idxs[-1]
                next_idx = select_next_node(current_idx, visited_idxs, pheromones, alpha, beta)
                if next_idx == -1:
                    break
                route_idxs.append(next_idx)
                visited_idxs.add(next_idx)

            if len(route_idxs) == num_nodes:
                route_idxs.append(start_idx)
                distance, time, cost = calculate_cost(route_idxs)
                if time == np.inf or cost == np.inf:
                    continue
                sol = {'route_idxs': route_idxs, 'time': time, 'cost': cost}
                iteration_solutions.append(sol)
                all_solutions.append(sol)
                update_archive(sol)
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
            current_best_time_in_archive = min(s['time'] for s in archive)
            current_best_cost_in_archive = min(s['cost'] for s in archive)
            print_and_save(f"Iteration {iteration + 1}: Best Time in Archive = {current_best_time_in_archive:.2f}, Best Cost in Archive = {current_best_cost_in_archive:.2f}")
        else:
            print_and_save(f"Iteration {iteration + 1}: Archive is empty.")
        print_and_save(f"Archive size: {len(archive)}")

    if archive:
        best_sol_from_archive = min(archive, key=lambda s: s['time'] + s['cost'])
        final_best_route_ids = [idx_to_id[i] for i in best_sol_from_archive['route_idxs']]
        final_best_time = best_sol_from_archive['time']
        final_best_cost = best_sol_from_archive['cost']
    else:
        final_best_route_ids = None
        final_best_time = float('inf')
        final_best_cost = float('inf')

    return final_best_route_ids, final_best_time, final_best_cost

# ===================== Run ACO =====================

best_route_ids, best_time, best_cost = aco_tsp(
    num_ants, num_iterations, alpha, beta, evaporation_rate, Q
)

print_and_save(f"\n--- Kết quả ACO ---")
print_and_save(f"Best route (IDs): {best_route_ids}")
print_and_save(f"Best time: {best_time:.2f} minutes")
print_and_save(f"Best cost: {best_cost:.2f} Baht")

# ===================== Plot Pareto Front =====================

times = [sol['time'] for sol in archive]
costs = [sol['cost'] for sol in archive]

plt.figure(figsize=(10, 7))
plt.scatter(times, costs, label='Pareto Front Solutions')
plt.xlabel('Time (minutes)')
plt.ylabel('Cost (Baht)')
plt.title('Pareto Front of Travel Routes (ACO)')
plt.grid(True)
plt.legend()
plt.savefig("output/ACO_pareto_front.png")
plt.close()

# ===================== Plot all solutions =====================

times = [sol['time'] for sol in all_solutions]
costs = [sol['cost'] for sol in all_solutions]

plt.figure(figsize=(10, 7))
plt.scatter(times, costs, label='Pareto Front Solutions')
plt.xlabel('Time (minutes)')
plt.ylabel('Cost (Baht)')
plt.title('Pareto Front of Travel Routes (ACO)')
plt.grid(True)
plt.legend()
plt.savefig("output/ACO_all_solutions.png")
plt.close()

# ===================== 2-Opt Route Refinement =====================

def refine_route_with_2opt(
    best_route_ids, coordinates, id_to_idx, idx_to_id,
    distance_matrix_np, time_matrix_np, cost_matrix_np
):
    if best_route_ids is None:
        print("Không có tuyến đường để tinh chỉnh.")
        return None, (np.inf, np.inf, np.inf)

    station_idxs = [
        id_to_idx[id_val]
        for id_val in coordinates[coordinates['name'].str.lower().str.contains("railway station")]['id'].tolist()
    ]
    best_route_idxs_full = [id_to_idx[id_val] for id_val in best_route_ids]
    poi_route_idxs = [node_idx for node_idx in best_route_idxs_full if node_idx not in station_idxs]
    poi_info_df = coordinates[coordinates['id'].isin([idx_to_id[i] for i in poi_route_idxs])].copy()

    def calculate_cost_internal(route_idxs):
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

    start_station_idx = best_route_idxs_full[0]

    def get_total_travel_cost_for_poi_selection(poi_id):
        poi_idx = id_to_idx[poi_id]
        cost_from_station = cost_matrix_np[start_station_idx, poi_idx] if cost_matrix_np[start_station_idx, poi_idx] != np.inf else 0
        cost_to_station = cost_matrix_np[poi_idx, start_station_idx] if cost_matrix_np[poi_idx, start_station_idx] != np.inf else 0
        return cost_from_station + cost_to_station

    poi_info_df['total_travel_cost'] = poi_info_df['id'].apply(get_total_travel_cost_for_poi_selection)
    selected_poi_ids = poi_info_df.sort_values(by=['rating', 'total_travel_cost'], ascending=[False, True]).head(11)['id'].tolist()
    selected_poi_idxs = [id_to_idx[id_val] for id_val in selected_poi_ids]
    reduced_route_idxs = [start_station_idx] + selected_poi_idxs + [start_station_idx]

    def two_opt_fixed_endpoints(route_idxs_opt, cost_fn):
        start_idx_opt, end_idx_opt = route_idxs_opt[0], route_idxs_opt[-1]
        middle_opt = route_idxs_opt[1:-1]
        best_middle_opt = list(middle_opt)
        improved = True
        while improved:
            improved = False
            current_best_cost = cost_fn([start_idx_opt] + best_middle_opt + [end_idx_opt])
            for i in range(len(best_middle_opt) - 1):
                for j in range(i + 1, len(best_middle_opt)):
                    new_middle_opt = best_middle_opt[:i] + best_middle_opt[i:j + 1][::-1] + best_middle_opt[j + 1:]
                    new_route_opt = [start_idx_opt] + new_middle_opt + [end_idx_opt]
                    new_cost = cost_fn(new_route_opt)
                    if new_cost < current_best_cost:
                        best_middle_opt = new_middle_opt
                        current_best_cost = new_cost
                        improved = True
        return [start_idx_opt] + best_middle_opt + [end_idx_opt]

    optimized_route_idxs = two_opt_fixed_endpoints(
        reduced_route_idxs, lambda r: calculate_cost_internal(r)[2]
    )
    opt_distance, opt_time, opt_cost = calculate_cost_internal(optimized_route_idxs)
    optimized_route_ids = [idx_to_id[idx] for idx in optimized_route_idxs]

    print_and_save(f"\n--- Kết quả sau tinh chỉnh 2-Opt ---")
    print_and_save("📍 Optimized Route (IDs):", optimized_route_ids)
    print_and_save(f"📏 Distance: {opt_distance:.2f} km")
    print_and_save(f"⏱ Time: {opt_time:.2f} minutes")
    print_and_save(f"💰 Cost: {opt_cost:.2f} Baht")

    plt.figure(figsize=(8, 6))
    plt.scatter(opt_time, opt_cost, color='blue', label='Optimized Route (2-Opt)')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Cost (Baht)')
    plt.title('Pareto Point After Route Reduction + 2-Opt')
    plt.legend()
    plt.grid(True)
    plt.savefig("output/2opt_pareto_point.png")
    plt.close()

    return optimized_route_ids, (opt_distance, opt_time, opt_cost)

# ===================== Run 2-Opt Refinement =====================

optimized_route_ids, (opt_distance, opt_time, opt_cost) = refine_route_with_2opt(
    best_route_ids, coordinates, id_to_idx, idx_to_id,
    distance_matrix_np, time_matrix_np, cost_matrix_np
)

# ===================== Close Output File =====================

output_file.close()