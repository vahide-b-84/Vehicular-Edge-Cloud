# server.py
import simpy

class Server:
    def __init__(self, env, server_type, server_id, RSU_id, processing_frequency, failure_rate, failure_model):
        self.env = env
        self.server_type = server_type
        self.server_id = server_id
        self.RSU_id = RSU_id
        self.queue = simpy.PriorityResource(env, capacity=1)

        self.processing_frequency = processing_frequency  # fn(t)
        self.failure_rate = failure_rate  # λ_n (t)
        self.failure_Model= failure_model # "Permanent"  or    "Transient"

    def reset_queue(self, env):
        """Reset the queue to be associated with the new environment."""
        self.env = env
        self.queue = simpy.PriorityResource(env, capacity=1)  # Reinitialize the queue with the new env
        

    def update_failure_params(self, failure_rate, failure_model):
        self.failure_rate = failure_rate
        self.failure_model = failure_model