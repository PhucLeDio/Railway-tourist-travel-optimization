import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
import matplotlib.pyplot as plt
import os

# ===================== Data Loading and Preparation =====================
# Specify the directory for output
output_directory = "NSGA_2_output"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)  # Create the directory if it doesn't exist

output_file_path = os.path.join(output_directory, "output_updated_with_improvements.txt")
output_file = open(output_file_path, "w", encoding="utf-8")

def print_and_save(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=output_file)

# Đọc dữ liệu tọa độ
coordinates = pd.read_csv("data/coordinates_3.csv")

# Ánh xạ ID của điểm đến chỉ số liên tục và ngược lại
id_to_idx = {id_val: idx for idx, id_val in enumerate(coordinates['id'])}
idx_to_id = {idx: id_val for idx, id_val in enumerate(coordinates['id'])}
num_nodes = len(coordinates)

# Lọc các điểm là ga tàu
station_idxs = [
    id_to_idx[id_val]
    for id_val in coordinates[coordinates['name'].str.lower().str.contains("railway station")]['id'].tolist()
]

# Chọn 10 địa danh có rating cao nhất, loại trừ các ga tàu
poi_data = coordinates[~coordinates['id'].isin([coordinates['id'][idx] for idx in station_idxs])]
top_10_pois = poi_data.nlargest(10, 'rating')

# Các ID và chỉ số của địa danh top 10
top_10_ids = top_10_pois['id'].tolist()
top_10_idxs = [id_to_idx[id_val] for id_val in top_10_ids]
num_pois_to_visit = len(top_10_idxs)

# Đọc ma trận khoảng cách, thời gian và chi phí
distance_data = pd.read_csv("data/formatted_distance_matrix_taxi_with_service_cost.csv")
distance_matrix_np = np.full((num_nodes, num_nodes), np.inf)
time_matrix_np = np.full((num_nodes, num_nodes), np.inf)
cost_matrix_np = np.full((num_nodes, num_nodes), np.inf)

# Cập nhật các ma trận với dữ liệu từ CSV
for idx, row in distance_data.iterrows():
    from_idx = id_to_idx[row['from_id']]
    to_idx = id_to_idx[row['to_id']]
    distance_matrix_np[from_idx, to_idx] = row['distance_km']
    time_matrix_np[from_idx, to_idx] = row['time_min']
    cost_matrix_np[from_idx, to_idx] = row['cost_baht']

# Hàm tính chi phí tuyến đường (giữ nguyên từ mã gốc)
def calculate_path_metrics(route_idxs: list[int]) -> tuple[float, float, float]:
    total_distance = 0
    total_time = 0
    total_cost = 0
    for i in range(len(route_idxs) - 1):
        from_idx = route_idxs[i]
        to_idx = route_idxs[i + 1]
        if time_matrix_np[from_idx, to_idx] == np.inf or cost_matrix_np[from_idx, to_idx] == np.inf:
            return np.inf, np.inf, np.inf
        total_distance += distance_matrix_np[from_idx, to_idx]
        total_time += time_matrix_np[from_idx, to_idx]
        total_cost += cost_matrix_np[from_idx, to_idx]
    return total_distance, total_time, total_cost

# Định nghĩa lớp RoutingProblem với ga tàu cố định
class RoutingProblem(Problem):
    def __init__(self,
                 num_pois_to_visit,
                 station_idx,
                 top_10_idxs,
                 time_matrix,
                 cost_matrix,
                 **kwargs):
        super().__init__(n_var=num_pois_to_visit,
                         n_obj=2,
                         n_constr=0,
                         xl=0,
                         xu=num_pois_to_visit - 1,
                         type_var=np.integer,
                         **kwargs)
        self.station_idx = station_idx
        self.top_10_idxs = top_10_idxs
        self.time_matrix = time_matrix
        self.cost_matrix = cost_matrix

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.full((X.shape[0], self.n_obj), np.inf)
        for i, individual_X in enumerate(X):
            route_poi_idxs = [self.top_10_idxs[int(idx)] for idx in individual_X]
            full_route_idxs = [self.station_idx] + route_poi_idxs + [self.station_idx]
            _, total_time, total_cost = calculate_path_metrics(full_route_idxs)
            F[i, 0] = total_time
            F[i, 1] = total_cost
        out["F"] = F

# Chạy NSGA-II cho từng ga tàu
all_pareto_fronts = []

for station_idx in station_idxs:
    print_and_save(f"\nChạy NSGA-II cho ga tàu {station_idx}...")
    problem = RoutingProblem(
        num_pois_to_visit=num_pois_to_visit,
        station_idx=station_idx,
        top_10_idxs=top_10_idxs,
        time_matrix=time_matrix_np,
        cost_matrix=cost_matrix_np
    )
    algorithm = NSGA2(
        pop_size=100,
        sampling=PermutationRandomSampling(),
        crossover=OrderCrossover(),
        mutation=InversionMutation(),
        eliminate_duplicates=True
    )
    res = minimize(problem, algorithm, ('n_gen', 100), seed=1, verbose=True, save_history=True)
    all_pareto_fronts.append((station_idx, res.F, res.X))

# In kết quả
print_and_save("\n--- Kết quả NSGA-II ---")
for station_idx, F, X in all_pareto_fronts:
    print_and_save(f"\nGa tàu {station_idx}:")
    print_and_save(f"Số lượng giải pháp trên mặt Pareto: {len(F)}")
    if len(F) > 0:
        print_and_save("Một số giải pháp trên mặt Pareto (Thời gian, Chi phí):")
        for i, f in enumerate(F[:5]):
            print_and_save(f"Giải pháp {i+1}: Thời gian={f[0]:.2f} phút, Chi phí={f[1]:.2f} Baht")

# ===================== Trực quan hóa cho mỗi ga tàu =====================
for station_idx, F, _ in all_pareto_fronts:
    plt.figure(figsize=(10, 7))  # Create a new figure for each station
    plt.scatter(F[:, 0], F[:, 1], s=30, label=f'Station {station_idx}')
    plt.title(f"Pareto Front for Station {station_idx}")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Cost (Baht)")
    plt.grid(True)
    plt.legend()

    # Define the path to save the plot for each station
    plot_file_path = os.path.join(output_directory, f"Pareto_front_station_{station_idx}.png")

    # Save the plot to the specified directory
    plt.savefig(plot_file_path)
    plt.close()  # Close the plot to avoid overlapping with the next station

# Tìm giải pháp tốt nhất (ví dụ: dựa trên tổng thời gian + chi phí)
best_solutions = []
for station_idx, F, X in all_pareto_fronts:
    if len(F) > 0:
        combined_objective_values = F[:, 0] + F[:, 1]
        best_solution_idx = np.argmin(combined_objective_values)
        best_X = X[best_solution_idx]
        route_poi_idxs = [top_10_idxs[int(idx)] for idx in best_X]
        full_route_idxs = [station_idx] + route_poi_idxs + [station_idx]
        _, best_time, best_cost = calculate_path_metrics(full_route_idxs)
        best_route_ids = [idx_to_id[idx] for idx in full_route_idxs]
        best_solutions.append({
            'station_idx': station_idx,
            'route_ids': best_route_ids,
            'time': best_time,
            'cost': best_cost
        })

# In giải pháp tốt nhất cho mỗi ga tàu
print_and_save("\n--- Giải pháp tốt nhất cho mỗi ga tàu (dựa trên tổng thời gian + chi phí) ---")
for sol in best_solutions:
    print_and_save(f"\nGa tàu {sol['station_idx']}:")
    print_and_save(f"  Tuyến đường (IDs): {sol['route_ids']}")
    print_and_save(f"  Thời gian: {sol['time']:.2f} phút")
    print_and_save(f"  Chi phí: {sol['cost']:.2f} Baht")

# close the output file
output_file.close()