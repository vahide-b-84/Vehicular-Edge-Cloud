# params.py
from configuration import parameters
from env_extractors import extract_max_speed_from_rou, estimate_max_e2e_delay
import pandas as pd
class params:
    
    
   
    model_summary = parameters.model_summary  # Options: "dqn", "original_only", "greedy"
    min_computation_demand = None
    SCENARIO_TYPE = parameters.SCENARIO_TYPE
    Permutation_Number = parameters.Permutation_Number
    
    additional_episodes=parameters.additional_episodes

    Alpha = parameters.compute_Alpha()
    alpha_edge = (None, None)
    alpha_cloud = (None, None)

    NUM_EDGE_SERVERS = parameters.NUM_EDGE_SERVERS
    NUM_CLOUD_SERVERS = parameters.NUM_CLOUD_SERVERS
    RSUs_EDGE_SERVERS = parameters.RSUs_EDGE_SERVERS
    RSU_radius = parameters.RSU_radius

    num_vehicles = parameters.num_vehicles
    task_result_size = parameters.task_result_size
    TASK_ARRIVAL_RATE_range = parameters.TASK_ARRIVAL_RATE_range
    taskno = parameters.taskno
    Vehicle_taskno = parameters.Vehicle_taskno
    total_episodes = parameters.total_episodes
    TASK_SIZE_RANGE = parameters.TASK_SIZE_RANGE
    Low_demand, High_demand = parameters.Low_demand, parameters.High_demand 
    EDGE_PROCESSING_FREQ_RANGE= parameters.EDGE_PROCESSING_FREQ_RANGE
    CLOUD_PROCESSING_FREQ_RANGE=parameters.CLOUD_PROCESSING_FREQ_RANGE

    MAX_VEHICLE_SPEED = extract_max_speed_from_rou()
    Max_e2e_Delay = estimate_max_e2e_delay()



    #network
    rsu_to_edge_profile = parameters.rsu_to_edge_profile
    rsu_to_cloud_profile = parameters.rsu_to_cloud_profile
    network_speed = parameters.network_speed
    RSU_LINK_BANDWIDTH_RANGE=parameters.RSU_LINK_BANDWIDTH_RANGE
    task_timeout_caching = parameters.task_timeout_caching
    link_failure_rate_range = parameters.link_failure_rate_range
    Q_alpha = parameters.Queuing_alpha
    beta = parameters.beta



    # Global DQN
    Global = {
        'activation': parameters.global_af,
        'hidden_layers': parameters.global_hidden_layers,
        'lr': parameters.global_lr,
        'gamma': parameters.global_gamma,
        'tau': parameters.global_tau,
        'buffer_capacity': parameters.global_buffer_capacity,
        'batch_size': parameters.global_batch_size,
        'epsilon_start': parameters.global_epsilon_start,
        'epsilon_end': parameters.global_epsilon_end,
        'epsilon_decay': parameters.global_epsilon_decay,
        'warmup_episodes': parameters.global_warmup_episodes,
        'load_penalty_weight' : parameters.global_load_penalty_weight,
        'epsilon':parameters.global_epsilon
    }

    # Local DQN
    Local = {
        'activation': parameters.local_af,
        'hidden_layers': parameters.local_hidden_layers,
        'lr': parameters.local_lr,
        'gamma': parameters.local_gamma,
        'tau': parameters.local_tau,
        'buffer_capacity': parameters.local_buffer_capacity,
        'batch_size': parameters.local_batch_size,
        'epsilon_start': parameters.local_epsilon_start,
        'epsilon_end': parameters.local_epsilon_end,
        'epsilon_decay': parameters.local_epsilon_decay,
        'warmup_episodes': parameters.local_warmup_episodes,
        'epsilon': parameters.local_epsilon
    }

