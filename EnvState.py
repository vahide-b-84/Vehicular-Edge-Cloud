
#EnvState.py
import numpy as np
import pandas as pd
from server import Server
from params import params

class EnvironmentState:
    def __init__(self, env, RSU_and_Vehicle):
        self.env=env
        self.RSU_and_Vehicle = RSU_and_Vehicle
        self.servers = {}  # Dictionary to store server objects {server_id: server_object, 'failure_rate': failure_rate, 'load': load}
        self.tasks = {}  # Dictionary to store generated task objects {task_id: task_object}
        self.num_completed_tasks = 0  # Number of completed tasks at all servers

        self.setServers() # set the RSUs' server parameters 
        
        gn = self.RSU_and_Vehicle.graph_network  # 
        self.RSU_Pairs_failure_rate = gn.RSU_Pairs_failure_rate
        self.RSU_distances = gn.RSU_distances
        self.RSU_link_bandwidths = gn.RSU_link_bandwidths
        
    def calculate_e2e_delay(self, rsu_id_i, rsu_id_j, task_size):
        bandwidth = self.RSU_link_bandwidths[(rsu_id_i, rsu_id_j)]
        distance = self.RSU_distances[rsu_id_i][rsu_id_j]  # distance between RSUs
        propagation_speed = params.network_speed  # signal speed
        L_j = self.RSU_and_Vehicle.RSUs[rsu_id_j].taskCounter  # dest RSU load
        L_max = max(rsu.taskCounter for rsu in self.RSU_and_Vehicle.RSUs.values())     
        # Transmission Delay: D_transmission = S / B
        D_transmission = task_size / bandwidth
        
        # Propagation Delay: D_propagation = d / v
        D_propagation = distance / propagation_speed
        
        # Queuing Delay: D_queuing = α * (L / L_max)
        D_queuing = params.Q_alpha * (L_j / (L_max + 1e-8)) # Q sec maximum time
        
        # Retransmission Delay: D_retransmission = E[N] * (D_transmission + D_propagation)
        P_base = self.RSU_Pairs_failure_rate[(rsu_id_i, rsu_id_j)]
        P = P_base + params.beta * (L_j / (L_max + 1e-8))
        # Limit P to a maximum value
        P = min(P, 0.7)
        E_N = (1 / (1 - P)) - 1  # Expected number of retransmissions
        D_retransmission = E_N * (D_transmission + D_propagation)
        
        # Total End-to-End Delay
        D_e2e = D_transmission + D_propagation + D_queuing + D_retransmission
        return D_e2e
        
    def calculate_all_e2e_delays_flat(self, task_size):
        e2e_vector = []

        rsu_ids = sorted(self.RSU_and_Vehicle.RSUs.keys())
        for rsu_id_i in rsu_ids:
            for rsu_id_j in rsu_ids:
                if rsu_id_i != rsu_id_j:
                    e2e_delay = self.calculate_e2e_delay(
                        rsu_id_i, rsu_id_j,
                        task_size
                    )
                    e2e_vector.append(e2e_delay)
        
        return e2e_vector

    def add_server_and_init_environment(self, server_object):
        """Add a server object to the environment state."""
        server_id = server_object.server_id  # Extract the server ID from the server object
        #print(f"Adding server with ID {server_id}")
        self.servers[server_id] = {
            'server_object': server_object,
            'tasks_assigned': [],  # List of task objects assigned to this server
            'primary_failure_count': 1000000 * server_object.failure_rate,  # Initialize failure time for the server if it is selected as primary
            'backup_failure_count': 1000000 * server_object.failure_rate,  # Initialize failure time for the server if it is selected as backup
            'primary_executed_time': 1000000,  # Initialize executed tasks time for the server if it is selected as primary
            'backup_executed_time': 1000000,  # Initialize executed tasks time for the server if it is selected as backup
            'load': 0  # Initialize load for the server (sum of computation demands of tasks assigned to it)
        }

    def print_servers(self):
        """Print information about all servers in the environment."""
        if not self.servers:
            print("No servers available.")
            return
        
        for server_id, server_info in self.servers.items():
            server_object = server_info['server_object']
            failure_rate = server_info.get('failure_rate', 0)
            load = server_info.get('load', 0)
            
            print(f"Server ID: {server_id}")
            print(f"Failure Rate: {failure_rate}")
            print(f"Load: {load}")
            # Print additional information as needed

    def assign_task_to_server(self, server_id, task, selection):
        """Assign a task object to a server based on the selection (primary or backup)."""
          
        self.servers[server_id]['tasks_assigned'].append({'task': task, 'selection': selection})

        # Update 'load'
        self.servers[server_id]['load'] += task.computation_demand
        
    def complete_task(self, server_id, task, selection, execute_time):
        """Set parameters about completed task in the environment state."""
        tasks_assigned = self.servers[server_id]['tasks_assigned']
        
        for assigned_task in tasks_assigned:
            if assigned_task['task'] == task and assigned_task['selection'] == selection:
                # Update 'load'
                self.servers[server_id]['load'] -= task.computation_demand

                if selection == "primary" :
                    # Update 'primary_executed_tasks'
                    self.servers[server_id]['primary_executed_time'] += execute_time
                    if task.primaryStat == "failure":
                        # Update 'primary_failure_count'
                        self.servers[server_id]['primary_failure_count'] += execute_time
                    

                elif selection == "backup": 
                    # Update 'backup_executed_tasks'
                    self.servers[server_id]['backup_executed_time'] += execute_time
                    if task.backupStat == "failure":
                        # Update 'backup_failure_count'
                        self.servers[server_id]['backup_failure_count'] += execute_time

                self.num_completed_tasks += 1
                
                break
   
    def get_server_by_id(self, server_id):
        """Get a server object by its ID."""
        server_info = self.servers.get(server_id)
        if server_info:
            return server_info['server_object']
        else:
            return None
    
    def add_task(self, task_object):
        """Add a task object to the environment state."""
        task_id = task_object.id  # Extract the task ID from the task object
        self.tasks[task_id] = task_object

    '''def remove_task(self, task_id):
        """Remove a task object from the environment state."""
        if task_id in self.tasks:
            del self.tasks[task_id]
        else:
            print(f"Task with ID {task_id} not found in the task dictionary.")'''

    def get_task_by_id(self, task_id):
        """Get a task object by its ID."""
        return self.tasks.get(task_id)
   
    '''def get_min_computation_demand(self):
        """Get the minimum computation demand among all tasks."""
        if not self.tasks:
            print("No tasks available.")
            return None
        
        min_demand = float('inf')  # Initialize min_demand with positive infinity
        
        for task_id, task_obj in self.tasks.items():
            if task_obj.computation_demand < min_demand:
                min_demand = task_obj.computation_demand
        
        return min_demand
    '''    
    def reset(self, new_simpy_env):
        """Reset the environment state."""
        self.env=new_simpy_env
        self.RSU_and_Vehicle.env=self.env
        #self.servers = {}
        self.tasks= {}
        self.num_completed_tasks = 0
        for vehicle in self.RSU_and_Vehicle.vehicles.values():
            vehicle.reset(self.env)
        for rsu in self.RSU_and_Vehicle.RSUs.values():
            rsu.env_state = self
            rsu.reset(self.env)
        
        # Reset server statistics
        for server_id, server_data in self.servers.items():
            server_object = server_data['server_object']
            server_object.reset_queue(self.env)
            server_data['tasks_assigned'] = []
            server_data['primary_failure_count'] = 1000000 * server_object.failure_rate
            server_data['backup_failure_count'] = 1000000 * server_object.failure_rate
            server_data['primary_executed_time'] = 1000000
            server_data['backup_executed_time'] = 1000000
            server_data['load'] = 0
        
    def setServers(self):
        # Choose the appropriate file based on the scenario type
        excel_file = 'homogeneous_server_info.xlsx' if params.SCENARIO_TYPE == 'homogeneous' else 'heterogeneous_server_info.xlsx'

        # Read the specific sheet based on permutation number
        sheet_name = f'{params.SCENARIO_TYPE.capitalize()}_Permutation_{params.Permutation_Number}'
        server_info_df = pd.read_excel(excel_file, sheet_name=sheet_name)

        # Iterate over the DataFrame from the second row to create server objects
        for index, row in server_info_df.iterrows():
            server_id = row['Server_ID']
            RSU_id = row['RSU_ID']
            server_type = row['Server_Type'] 
            processing_frequency = row['Processing_Frequency'] 
            # Extract Failure Rate and Failure Model from the 4th and 5th columns
            failure_rate = row.iloc[3]  #  4th column contains first Failure Rate
            failure_model = row.iloc[4]  #  5th column contains first Failure Model
            
            # Create a Server object
            server = Server(self.env, server_type, server_id, RSU_id, processing_frequency, failure_rate, failure_model)
            
            # Add server to the environment state and initialize state
            self.add_server_and_init_environment(server)
            rsu = self.RSU_and_Vehicle.get_rsu_by_id(RSU_id)
            
            if rsu is not None:
                rsu.serverIDs.append(server_id)

            # If the server is a cloud server, add it to all RSUs
            if server_type == "Cloud":
                for rsu in self.RSU_and_Vehicle.RSUs.values():
                    rsu.serverIDs.append(server_id) 
        
        for rsu in self.RSU_and_Vehicle.RSUs.values():
            
            rsu.generate_combinations()

    def set_RSU_env_state(self):
        for rsu in self.RSU_and_Vehicle.RSUs.values():
            rsu.env_state = self

    def normalize(self, val, min_val, max_val):
        return (val - min_val) / (max_val - min_val + 1e-8)

    def get_state(self, task, RSU_ID=None):
        primary_failure_rate = []
        backup_failure_rate = []
        frequency = []
        load = []
        in_path = []

        if RSU_ID is not None:  # Local state
            for server_id, server_info in self.servers.items():
                server_object = server_info['server_object']

                if server_object.RSU_id == RSU_ID or server_object.RSU_id == 'cloud':
                    primary_failure_rate.append(server_info['primary_failure_count'] / server_info['primary_executed_time'])
                    backup_failure_rate.append(server_info['backup_failure_count'] / server_info['backup_executed_time'])
                    frequency.append(server_object.processing_frequency)
                    load.append(server_info['load']) # sum computation demands of tasks assigned to this server

            # normalization
            norm_primary = self.normalize(np.array(primary_failure_rate), params.alpha_cloud[0], params.alpha_edge[1])
            norm_backup = self.normalize(np.array(backup_failure_rate), params.alpha_cloud[0], params.alpha_edge[1])
            norm_frequency = self.normalize(np.array(frequency), params.EDGE_PROCESSING_FREQ_RANGE[0],
                                        params.CLOUD_PROCESSING_FREQ_RANGE[1])  
            
            max_local_load = max(load) if load else 1
            norm_load = np.array(load) / max_local_load if max_local_load > 0 else np.zeros_like(load)


            norm_task_size = self.normalize(task.task_size, params.TASK_SIZE_RANGE[0], params.TASK_SIZE_RANGE[1])
            norm_demand = self.normalize(task.computation_demand, params.Low_demand, params.High_demand)

            normalized_arr = np.concatenate([
                norm_primary,
                norm_backup,
                norm_frequency,
                norm_load,
                [norm_task_size, norm_demand]
            ], dtype=np.float32)

        else:  # Global state
            vehicle = task.vehicle
            vehicle_speed = vehicle.speed
            max_task = max(rsu.taskCounter for rsu in self.RSU_and_Vehicle.RSUs.values())
            rsu_features = []

            for rsu_id, rsu in self.RSU_and_Vehicle.RSUs.items():
                in_path = 1 if rsu_id in vehicle.rsu_subgraph else 0
                primary_fail = backup_fail = exec_time = freq_total = load_total = 0
                edge_count = 0

                for sid in rsu.serverIDs:
                    s_info = self.servers.get(sid)
                    server = s_info['server_object']
                    if server.server_type != 'Edge':
                        continue

                    primary_fail += s_info['primary_failure_count']
                    backup_fail += s_info['backup_failure_count']
                    exec_time += s_info['primary_executed_time'] + s_info['backup_executed_time']
                    freq_total += server.processing_frequency
                    load_total += s_info['load'] # sum computation demands of tasks assigned to servers of this RSU
                    edge_count += 1

                failure_rate = (primary_fail + backup_fail) / (exec_time + 1e-8) if exec_time > 0 else 0.0
                task_count = rsu.taskCounter  # property of RSU, total number of tasks assigned to this RSU

                norm_fail = self.normalize(failure_rate, params.alpha_edge[0], params.alpha_edge[1])
                norm_freq = self.normalize(freq_total,
                                        params.EDGE_PROCESSING_FREQ_RANGE[0],
                                        params.EDGE_PROCESSING_FREQ_RANGE[1] * max(edge_count, 1))
                
                norm_load = self.normalize(load_total, 0, max_task * params.High_demand)
                norm_task_count = self.normalize(task_count, 0, max_task)

                rsu_features.append([
                    norm_fail, norm_freq, norm_load, norm_task_count, in_path
                ])

            rsu_features = np.array(rsu_features).flatten()

            # e2e delay 
            e2e_delays = self.calculate_all_e2e_delays_flat(task.task_size)
            norm_e2e = self.normalize(np.array(e2e_delays), 0, params.Max_e2e_Delay)

            # task
            norm_task_size = self.normalize(task.task_size, params.TASK_SIZE_RANGE[0], params.TASK_SIZE_RANGE[1])
            norm_demand = self.normalize(task.computation_demand, params.Low_demand, params.High_demand)
            norm_speed = self.normalize(vehicle_speed, 20, params.MAX_VEHICLE_SPEED)

            normalized_arr = np.concatenate([
                rsu_features,
                norm_e2e,
                [norm_task_size, norm_demand, norm_speed]
            ], dtype=np.float32)

        return normalized_arr

    def print_state(self):
        for server_state in self.get_state():
            print(f"Server ID: {server_state['server_id']}")
            print(f"Frequency: {server_state['frequency']}")
            print(f"Load: {server_state['load']}")
            print(f"primary_failure_rate: {server_state['primary_failure_rate']}")
            print(f"backup_failure_rate: {server_state['backup_failure_rate']}")
