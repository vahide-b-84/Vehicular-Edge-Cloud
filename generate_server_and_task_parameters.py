import pandas as pd
import random
from configuration import parameters
import numpy as np
import json
from scipy.stats import truncnorm

NUM_CLOUD_SERVERS = parameters.NUM_CLOUD_SERVERS
state_permutations = parameters.generate_state_permutations()

def generate_processing_frequencies(number_of_server, server_type):
    """Generate processing frequencies for edge or cloud servers."""
    if server_type == "edge":
        return [round(random.uniform(*parameters.EDGE_PROCESSING_FREQ_RANGE), 2) for _ in range(number_of_server)]
    else:
        return [round(random.uniform(*parameters.CLOUD_PROCESSING_FREQ_RANGE), 2) for _ in range(number_of_server)]

def load_rsu_data(json_filename="graph_data.json"):
    """Load RSU data from the JSON file."""
    with open(json_filename, 'r') as f:
        data = json.load(f)
    return data['rsus']  # Extract RSU data as a dictionary

def load_task_queue(json_filename="taskQueue.json"):
    """Load task queue data from the JSON file."""
    with open(json_filename, 'r') as f:
        return json.load(f)  # Return list of tasks

def generate_server_info_per_permutation(scenario_type, filename, rsu_data):
    failure_rates = parameters.compute_failure_rates()
    for perm_num, permutation in enumerate(state_permutations, start=1):
        server_counter = 1  # Reset the server counter for each sheet
        server_info = []
        columns = ['Server_ID', 'Server_Type', 'Processing_Frequency']
        for state in permutation:
            columns.extend([f'Failure_Rate_{state}', f'Failure_Model_{state}'])
        columns.append('RSU_ID')  # Adding RSU_ID column at the end

        for rsu_id, rsu_info in rsu_data.items():
            num_edge_servers = rsu_info['edge_server_numbers']
            edge_frequencies = generate_processing_frequencies(num_edge_servers, "edge")
            for i in range(num_edge_servers):
                server_id = server_counter
                server_type = "Edge"
                processing_frequency = edge_frequencies[i]
                row = [server_id, server_type, processing_frequency]
                for state in permutation:
                    state_type = parameters.STATES[state][0].lower()
                    failure_model = "Permanent" if random.random() < parameters.STATES[state][1] else "Transient"
                    failure_rate_interval = (failure_rates['edge']['homogeneous'][state_type] if scenario_type == "homogeneous"
                                             else failure_rates['edge']['heterogeneous'][state_type])
                    failure_rate = round(random.uniform(*failure_rate_interval), 6)
                    row.extend([failure_rate, failure_model])
                row.append(rsu_id)
                server_info.append(row)
                server_counter += 1

        for i in range(1, NUM_CLOUD_SERVERS + 1):
            server_id = server_counter
            server_type = "Cloud"
            cloud_frequencies = generate_processing_frequencies(NUM_CLOUD_SERVERS, "cloud")
            processing_frequency = cloud_frequencies[i-1]
            row = [server_id, server_type, processing_frequency]
            for state in permutation:
                state_type = parameters.STATES[state][0].lower()
                failure_model = "Permanent" if random.random() < parameters.STATES[state][1] else "Transient"
                failure_rate_interval = (failure_rates['cloud']['homogeneous'][state_type] if scenario_type == "homogeneous"
                                         else failure_rates['cloud']['heterogeneous'][state_type])
                failure_rate = round(random.uniform(*failure_rate_interval), 6)
                row.extend([failure_rate, failure_model])
            row.append("cloud")
            server_info.append(row)
            server_counter += 1

        server_df = pd.DataFrame(server_info, columns=columns)
        sheet_name = f'{scenario_type.capitalize()}_Permutation_{perm_num}'
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a' if perm_num > 1 else 'w') as writer:
            server_df.to_excel(writer, sheet_name=sheet_name, index=False)

def generate_task_params(json_task_queue="taskQueue.json"):
    """Generate task parameters and save them to an Excel file."""
    task_queue = load_task_queue(json_task_queue)
    task_info = []
    TASK_SIZE_RANGE = parameters.TASK_SIZE_RANGE
    a, b = parameters.Low_demand, parameters.High_demand
    mu = (a + b) / 2
    sigma = (b - a) / 6
    lower, upper = (a - mu) / sigma, (b - mu) / sigma

    for task_id, task in enumerate(task_queue, start=1):
        task_size = np.random.randint(*TASK_SIZE_RANGE)
        computation_demand = truncnorm.rvs(lower, upper, loc=mu, scale=sigma)
        task_info.append({
            "Task_ID": task_id,
            "Vehicle_ID": task['vehicle_id'],
            "Time": task['time'],
            "Interarrival_Time": task['interarrival_time'],
            "Task_Size": task_size,
            "Computation_Demand": computation_demand
        })

    task_df = pd.DataFrame(task_info)
    task_df.to_excel('task_parameters.xlsx', index=False)

def main():
    rsu_data = load_rsu_data()
    generate_server_info_per_permutation('homogeneous', 'homogeneous_server_info.xlsx', rsu_data)
    generate_server_info_per_permutation('heterogeneous', 'heterogeneous_server_info.xlsx', rsu_data)
    generate_task_params()
    print("Parameters defined in Excel files!")

