import csv
import math
#import logging
import traci  # Importing TraCI for SUMO interactions

'''logger = logging.getLogger(__name__)'''

class Vehicle:
    def __init__(self, rsu_and_vehicle, vehicle_id, env, graph_network):
        self.vehicle_id = vehicle_id
        self.rsu_and_vehicle = rsu_and_vehicle
        self.env = env
        self.graph_network = graph_network
 
        self.speed = self.graph_network.vehicles[self.vehicle_id]["speed"]  # Get initial speed from graph_data
        self.Current_RSU = None
        self.position=None
        self.rsu_subgraph = self.graph_network.vehicles[self.vehicle_id]["rsu_subgraph"]  # Get rsu_subgraph from graph_data
        self.pending_tasks = []  # List of pending tasks
      
        #logger.info(f"Vehicle {self.vehicle_id} initialized.")
        

    def add_pending_task(self, task):
        """Add a task to the pending task list."""
        task.deadline = task.submitted_time + 2.0 * (task.computation_demand / 10 + task.task_size / 20)  # یا فرمول دلخواه

        self.pending_tasks.append(task)
        #logger.info(f"Task {task.id} added to pending tasks of {self.vehicle_id}")


    def request_results(self):
        
        for task in self.pending_tasks[:]:
            
            if self.env.now > task.deadline:
                task.timeout_time = self.env.now  
                task.deadline_flag = 'F'
                self.pending_tasks.remove(task)
                #logger.warning(f"Deadline missed for Task {task.id}, Vehicle {self.vehicle_id} ... Marked as Failed")
            elif self.Current_RSU:
                if task.id in self.Current_RSU.cached_results:
                    task.delivered_time = self.env.now
                    task.deadline_flag = 'S'
                    task.delivered_RSU = self.Current_RSU
                    del self.Current_RSU.cached_results[task.id]
                    self.pending_tasks.remove(task)
                    #logger.info(f"Vehicle {self.vehicle_id} received result for task {task.id}")
                #else:
                    #logger.info(f"Vehicle {self.vehicle_id} not found Task {task.id} in {self.Current_RSU.rsu_id} at {self.env.now}")



    def set_current_rsu(self, writer):
        """Find the closest RSU to the vehicle using SUMO's real-time position."""
        
        self.position = traci.vehicle.getPosition(self.vehicle_id)
        x, y = self.position
        min_distance = float('inf')
        closest_rsu = None

        for rsu_id, rsu in self.graph_network.rsus.items():
            rsu_x, rsu_y = rsu["position"]
            distance = math.hypot(x - rsu_x, y - rsu_y)  

            if distance <= rsu["range"] and distance < min_distance:
                min_distance = distance
                closest_rsu = rsu_id

        self.Current_RSU = self.rsu_and_vehicle.get_rsu_by_id(closest_rsu)

        #if closest_rsu:
        #    logger.info(f"Vehicle {self.vehicle_id} is in range of {closest_rsu} at Simpytime {self.env.now} and SUMO time {traci.simulation.getTime()}")
        #else:
        #    logger.info(f"Vehicle {self.vehicle_id} is not in range of any RSU at Simpytime {self.env.now} and SUMO time {traci.simulation.getTime()}")
        
        writer.writerow([traci.simulation.getTime(), self.vehicle_id, closest_rsu])  #  Log vehicle-RSU association
        self.request_results()
        
    def reset(self, new_SimPy_env):
        """Reset the vehicle's state for a new simulation run."""
        self.env = new_SimPy_env
        self.Current_RSU = None
        self.pending_tasks.clear()
        #logger.info(f"Vehicle {self.vehicle_id} reset for new simulation run.")
