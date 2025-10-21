#configuration.py
import itertools
from scipy.stats import norm

class parameters:
    # Simulation settings
    SCENARIO_TYPE = "heterogeneous"  # Options: "homogeneous", "heterogeneous"
    model_summary= "greedy" # "dqn", "original_only", "greedy"
    Permutation_Number = 3  # Permutation number for the scenario

    
    # Simulation episodes
    total_episodes = 1000 # 1200
    additional_episodes=600

    # RSUs and servers
    NUM_EDGE_SERVERS = 0
    RSUs_EDGE_SERVERS = (6,7)
    RSU_radius = (850, 1000)
    NUM_CLOUD_SERVERS = 2
    serverNo = NUM_EDGE_SERVERS + NUM_CLOUD_SERVERS
    failure_model_weight = 1

    # Vehicle and task settings
    num_vehicles = 3
    Vehicle_taskno = 600
    taskno = Vehicle_taskno * num_vehicles
    TASK_ARRIVAL_RATE_range = (0.8,1) # task/s
    TASK_SIZE_RANGE = (100,1000) #Mb  
    task_result_size = 10 #Mb
    Low_demand, High_demand = 1, 100 # Normal(50,16) MIPS

    # Network
    network_speed = 2e8 #m/s for propagation delay
    RSU_LINK_BANDWIDTH_RANGE = (200, 500)  # Mb/s

    link_failure_rate_range = (0.1, 0.5)
    Queuing_alpha = 5 # (0,Q)s wait time in dest RSU according to its Load
    beta = 0.2
    task_timeout_caching = 100

    rsu_to_edge_profile = {
        "bandwidth": 10000,
        "propagation_delay": 0
    }

    rsu_to_cloud_profile = {
        "bandwidth": 80, #Mb/s
        "propagation_delay": 1
    }


    # Global DQN parameters
    global_hidden_layers = [128, 64]
    global_af = "relu"
    global_lr = 0.0003
    global_gamma = 0.85
    global_tau = 0.005
    global_buffer_capacity = 1000000
    global_batch_size = 256
    global_epsilon_start = 1.0
    global_epsilon_end = 0.01
    global_epsilon_decay = 400
    global_warmup_episodes=15
    global_load_penalty_weight = 5.0
    global_epsilon=0.3

    # Local DQN parameters
    local_hidden_layers = [128, 64]
    local_af = "relu"
    local_lr = 0.0005               # 
    local_gamma = 0.90
    local_tau = 0.005
    local_buffer_capacity = 200000 # 
    local_batch_size = 256         #
    local_epsilon_start = 1.0
    local_epsilon_end = 0.01
    local_epsilon_decay = 300      # 300 exploit                  500, 700 explore
    local_warmup_episodes = 20
    local_epsilon=0.2


    # Edge reliability
    INITIAL_FAILURE_PROB_LOW_EDGE = 0.0001
    INITIAL_FAILURE_PROB_HIGH_EDGE = 0.79
    INITIAL_FAILURE_PROB_MED_EDGE = 0.55
    HOMOGENEOUS_INTERVAL_EDGE = 0.10
    HETEROGENEOUS_INTERVAL_EDGE = 0.20
    EDGE_PROCESSING_FREQ_RANGE = (10, 15)#MIPS

    # Cloud reliability
    INITIAL_FAILURE_PROB_LOW_CLOUD = 1e-6
    INITIAL_FAILURE_PROB_HIGH_CLOUD = 7.9e-6
    INITIAL_FAILURE_PROB_MED_CLOUD = 5.5e-6
    HOMOGENEOUS_INTERVAL_CLOUD = 1e-6
    HETEROGENEOUS_INTERVAL_CLOUD = 2e-6
    CLOUD_PROCESSING_FREQ_RANGE = (30, 60)#MIPS

    STATES = {
        "S1": ("Low", 1 - failure_model_weight),
        "S2": ("High", 1 - failure_model_weight),
        "S3": ("Med", 1 - failure_model_weight)
    }

    @staticmethod
    def generate_state_permutations():
        return list(itertools.permutations(parameters.STATES.keys()))

    @staticmethod
    def compute_failure_probabilities():
        return {
            'edge': {
                'homogeneous': {
                    'low': (parameters.INITIAL_FAILURE_PROB_LOW_EDGE, parameters.INITIAL_FAILURE_PROB_LOW_EDGE + parameters.HOMOGENEOUS_INTERVAL_EDGE),
                    'high': (parameters.INITIAL_FAILURE_PROB_HIGH_EDGE, parameters.INITIAL_FAILURE_PROB_HIGH_EDGE + parameters.HOMOGENEOUS_INTERVAL_EDGE),
                    'med': (parameters.INITIAL_FAILURE_PROB_MED_EDGE, parameters.INITIAL_FAILURE_PROB_MED_EDGE + parameters.HOMOGENEOUS_INTERVAL_EDGE)
                },
                'heterogeneous': {
                    'low': (parameters.INITIAL_FAILURE_PROB_LOW_EDGE, parameters.INITIAL_FAILURE_PROB_LOW_EDGE + parameters.HETEROGENEOUS_INTERVAL_EDGE),
                    'high': (parameters.INITIAL_FAILURE_PROB_HIGH_EDGE, parameters.INITIAL_FAILURE_PROB_HIGH_EDGE + parameters.HETEROGENEOUS_INTERVAL_EDGE),
                    'med': (parameters.INITIAL_FAILURE_PROB_MED_EDGE, parameters.INITIAL_FAILURE_PROB_MED_EDGE + parameters.HETEROGENEOUS_INTERVAL_EDGE)
                }
            },
            'cloud': {
                'homogeneous': {
                    'low': (parameters.INITIAL_FAILURE_PROB_LOW_CLOUD, parameters.INITIAL_FAILURE_PROB_LOW_CLOUD + parameters.HOMOGENEOUS_INTERVAL_CLOUD),
                    'high': (parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD, parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD + parameters.HOMOGENEOUS_INTERVAL_CLOUD),
                    'med': (parameters.INITIAL_FAILURE_PROB_MED_CLOUD, parameters.INITIAL_FAILURE_PROB_MED_CLOUD + parameters.HOMOGENEOUS_INTERVAL_CLOUD)
                },
                'heterogeneous': {
                    'low': (parameters.INITIAL_FAILURE_PROB_LOW_CLOUD, parameters.INITIAL_FAILURE_PROB_LOW_CLOUD + parameters.HETEROGENEOUS_INTERVAL_CLOUD),
                    'high': (parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD, parameters.INITIAL_FAILURE_PROB_HIGH_CLOUD + parameters.HETEROGENEOUS_INTERVAL_CLOUD),
                    'med': (parameters.INITIAL_FAILURE_PROB_MED_CLOUD, parameters.INITIAL_FAILURE_PROB_MED_CLOUD + parameters.HETEROGENEOUS_INTERVAL_CLOUD)
                }
            }
        }

    @staticmethod
    def compute_failure_rates():
        failure_probs = parameters.compute_failure_probabilities()
        mean = (parameters.Low_demand + parameters.High_demand) / 2
        std = (parameters.High_demand - parameters.Low_demand) / 6

        def get_failure_rate_interval(prob_interval):
            lower_percentile_value = norm.ppf(1 - prob_interval[0], loc=mean, scale=std)
            upper_percentile_value = norm.ppf(1 - prob_interval[1], loc=mean, scale=std)
            return (1 / lower_percentile_value, 1 / upper_percentile_value)

        def compute_all(rtype):
            return {
                'homogeneous': {k: get_failure_rate_interval(v) for k, v in failure_probs[rtype]['homogeneous'].items()},
                'heterogeneous': {k: get_failure_rate_interval(v) for k, v in failure_probs[rtype]['heterogeneous'].items()}
            }

        return {
            'edge': compute_all('edge'),
            'cloud': compute_all('cloud')
        }

    @staticmethod
    def compute_Alpha():
        failure_rates = parameters.compute_failure_rates()

        def calc_alpha(rate_interval):
            return ((rate_interval[1] - rate_interval[0]) / parameters.taskno, rate_interval[1])

        def compute_all(rtype):
            return {
                'homogeneous': {k: calc_alpha(v) for k, v in failure_rates[rtype]['homogeneous'].items()},
                'heterogeneous': {k: calc_alpha(v) for k, v in failure_rates[rtype]['heterogeneous'].items()}
            }

        return {
            'edge': compute_all('edge'),
            'cloud': compute_all('cloud')
        }
