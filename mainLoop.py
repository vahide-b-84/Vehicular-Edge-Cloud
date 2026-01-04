# mainLoop.py
from Sumo_Graph import GraphNetwork
from task import Task
from EnvState import EnvironmentState
from params import params
import simpy
import pandas as pd
from RSU_Vehicle_Setup import RSU_and_Vehicle_setup 
from Global_model import global_model
import traci
from save import save_params_and_logs  # Import from current directory

class MainLoop:
    def __init__(self):
        
        self.network = GraphNetwork() 
        self.network.load_graph() # load the object data from sumo data that saved in json file before!
        #self.network.plot_graph() # if you want to see the graph
        self.env = simpy.Environment()
        self.iteration_complete_event =  self.env.event()  # Event to signal the completion of iteration
        self.RSU_and_Vehicle = RSU_and_Vehicle_setup(self.env, self.network) # setup the RSUs and vehicles 
        self.env_state = EnvironmentState(self.env, self.RSU_and_Vehicle)
        
        # Initialize the level-1 model
        self.G_model = global_model(self.env, self.env_state, self.network.num_rsus)
        self.task_params_df = pd.read_excel('task_parameters.xlsx')
        params.min_computation_demand=self.task_params_df['Computation_Demand'].min()
        self.inter_arrival_times = self.task_params_df['Interarrival_Time'].tolist()
        
        # Initialize the simulation parameters
        self.this_episode = 0
        self.total_episodes = params.total_episodes
        self.taskCounter = 1  # Initialize taskCounter



    def EP(self):
        
        self.this_episode = 0
        
        while self.this_episode < self.total_episodes:

            self.this_episode = self.this_episode + 1
            print(f"Starting episode {self.this_episode}...")
            self.reset_setting()

            self.env.process(self.iteration()) # execute as a process
            self.env.process(self.RSU_and_Vehicle.Start_SUMO(use_gui=False)) #use_gui=True
            self.env.run(until=self.iteration_complete_event)  
            traci.close()
            #print(f"Sumo simulation for one episode finished.")
            
        rsu_logs_dict, rsu_assignments_dict = self.RSU_and_Vehicle.extract_rsu_logs_and_assignments()    
        save_params_and_logs(params, self.G_model.log_data, self.G_model.task_Assignments_info, rsu_logs_dict, rsu_assignments_dict)
        self.clear_logs(rsu_logs_dict, rsu_assignments_dict)

    
    def iteration(self):
        
        for inter_arrival_time in self.inter_arrival_times:    
            
            yield self.env.timeout(inter_arrival_time)  # Wait for the inter-arrival time
            
            # Create a task object
            task = Task(self.env, self.env_state, self.taskCounter, self.RSU_and_Vehicle.vehicles, self.task_params_df) # Generate a task object
            
            # Add the task to the environment state
            self.env_state.add_task(task)

            # Process the task independently
            self.env.process(self.task_submition(task))

            # Update the task counter
            self.taskCounter +=1
        
        
        while True:
            all_tasks_ready = all(
                task.selected_rsu_start_time is not None for task in self.env_state.tasks.values()
            )
            if all_tasks_ready:
                break
            else:
                yield self.env.timeout(5)  # Wait for a short simulation time

        #Process pending tasks for all RSUs and the global model
        processes = []
        for rsu in self.RSU_and_Vehicle.RSUs.values():
            processes.append(self.env.process(rsu.process_pendingList_and_log_result(self.this_episode)))
        if params.model_summary == "dqn":
            processes.append(self.env.process(self.G_model.process_pendingList_and_log_result(self.this_episode)))  # dqn
        else:
            raise ValueError(f"Unknown model summary: {params.model_summary}")
        
        yield self.env.all_of(processes)
        print("the task iteration completed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.iteration_complete_event.succeed() 
          
    def task_submition(self, task):
        max_wait_time = 1000  # Define a maximum wait time to prevent infinite waiting
        waited_time = 0
        while task.vehicle.Current_RSU is None and waited_time < max_wait_time:
            yield self.env.timeout(1)  # Wait for 1 time unit
            waited_time += 1
        if task.vehicle.Current_RSU is None:
            print(f"Task {task.id} could not find a valid RSU within the maximum wait time.")
            return  # Exit the process if no RSU is found
        task.original_RSU = task.vehicle.Current_RSU
        task.submitted_time = self.env.now # in original RSU
        task.vehicle.add_pending_task(task)
        
        if params.model_summary =="dqn":
            selected_RSU = self.G_model.Recommend_RSU(task, self.RSU_and_Vehicle.RSUs, self.this_episode)
        else:
            raise ValueError(f"Unknown model summary: {params.model_summary}")        
          
        yield self.env.process(selected_RSU.receive_task(task))  # Wait for RSU to receive task
        X, Y, Z = selected_RSU.Recommend_XYZ(task, self.this_episode)  # Get execution plan
        self.env.process(task.execute_task(X, Y, Z))  # Start task execution
    
    def clear_logs(self, rsu_logs_dict, rsu_assignments_dict):
        self.G_model.log_data.clear()
        self.G_model.task_Assignments_info.clear()
        for rsu_id in rsu_logs_dict:
            rsu_logs_dict[rsu_id].clear()
            rsu_assignments_dict[rsu_id].clear()

    def reset_setting(self):
        
        self.taskCounter = 1 
        del self.env # Delete the old environment to reset the simulation
        self.env = simpy.Environment()
        self.env_state.reset(self.env, self.this_episode) #  set server  reset local models and vehicles
        self.G_model.reset(self.env,self.this_episode) # reset global model 
        self.iteration_complete_event = self.env.event()  # Reset the event at the beginning of each episode