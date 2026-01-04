# Level-2 model
import numpy as np
import math
import torch
from DQN_template import DQNAgent
from params import params
import logging

logger = logging.getLogger(__name__)

class RSU:
    def __init__(self, rsu_and_vehicle, rsu_id, Edge_number, RSU_position, env):
        self.env = env
        self.env_state = None
        self.rsu_id = rsu_id
        self.rsu_position = RSU_position
        self.serverNo = Edge_number + params.NUM_CLOUD_SERVERS
        self.serverIDs = []
        self.rsu_and_vehicle = rsu_and_vehicle
        self.num_states = 4 * self.serverNo + 2
        self.num_actions = (self.serverNo * self.serverNo) + (self.serverNo * (self.serverNo - 1)) // 2

        self.epsilon_start = params.Local['epsilon_start']
        self.epsilon_end = params.Local['epsilon_end']
        self.epsilon_decay = params.Local['epsilon_decay']

        self.state = []
        self.action = None
        self.tempbuffer = {}
        self.taskCounter = 0
        self.pendingList = []
        self.rewardsAll = []
        self.ep_reward_list = []
        self.ep_delay_list = []
        self.avg_reward_list = []
        self.episodic_reward = 0
        self.episodic_delay = 0
        self.log_data = []
        self.task_Assignments_info = []
        self.index_of_actions = []

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if params.model_summary == "dqn":
            
            self.local_model = DQNAgent(
                num_states=self.num_states,
                num_actions=self.num_actions,
                hidden_layers=params.Local['hidden_layers'],
                device=device,
                gamma=params.Local['gamma'],
                lr=params.Local['lr'],
                tau=params.Local['tau'],
                buffer_size=params.Local['buffer_capacity'],
                batch_size=params.Local['batch_size'],
                activation=params.Local['activation'],
            )
            print(f"{self.rsu_id}: using DQNAgent as local model")


        self.cached_results = {}

    def get_epsilon(self, episode):
            # Linear decay instead of exponential
            epsilon = max(self.epsilon_end, self.epsilon_start - (episode / self.epsilon_decay))
            return epsilon
    
    def generate_combinations(self):
        for i in self.serverIDs:
            for j in self.serverIDs:
                self.index_of_actions.append((i, j, 0))
        for i in self.serverIDs:
            for j in self.serverIDs:
                if i < j:
                    self.index_of_actions.append((i, j, 1))

    def extract_parameters(self):
        index = self.action
        
        primary, backup, z = self.index_of_actions[index]
        return self.env_state.get_server_by_id(primary), self.env_state.get_server_by_id(backup), z

    def Recommend_XYZ(self, task, episode):
        self.taskCounter += 1
        
        self.state = self.env_state.get_state(task, self.rsu_id)

        if self.taskCounter > 1:
            temp = list(self.tempbuffer[self.taskCounter - 1])
            temp[3] = self.state
            self.tempbuffer[self.taskCounter - 1] = tuple(temp)
            self.add_train(episode)

        epsilon = self.get_epsilon(episode)
        self.action = self.local_model.select_action(self.state, epsilon)
        X, Y, Z = self.extract_parameters()
        self.tempbuffer[self.taskCounter] = (self.state, self.action, None, [])
        self.pendingList.append((task.id, self.taskCounter))
        return X, Y, Z

    def calcReward(self, taskID):
        task = self.env_state.get_task_by_id(taskID)
        z = task.z
        primaryStat = task.primaryStat
        backupStat = task.backupStat
        primaryFinished = task.primaryFinished
        primaryStarted = task.primaryStarted
        backupFinished = task.backupFinished
        backupStarted = task.backupStarted

        flag = "s"  
        delay = None

        if z == 0:
            if primaryStat == 'success' and backupStat is None and primaryFinished is not None:
                delay = primaryFinished - primaryStarted
            elif primaryStat == 'failure' and backupStat == 'success' and backupFinished is not None:
                delay = backupFinished - primaryStarted
            elif primaryStat == 'failure' and backupStat == 'failure':
                delay = backupFinished - primaryStarted
                flag = "f"
            else:
                flag = "n"
        else:
            if primaryStat == 'success' and backupStat == 'success' and primaryFinished is not None and backupFinished is not None:
                delay = min(primaryFinished, backupFinished) - primaryStarted
            elif primaryStat == 'success' and backupStat == 'failure' and primaryFinished is not None:
                delay = primaryFinished - primaryStarted
            elif primaryStat == 'failure' and backupStat == 'success' and backupFinished is not None:
                delay = backupFinished - backupStarted
            elif primaryStat == 'failure' and backupStat == 'failure':
                delay = max(backupFinished - backupStarted, primaryFinished - primaryStarted)
                flag = "f"
            elif primaryStat == 'success' and backupStat is None and primaryFinished is not None:
                delay = primaryFinished - primaryStarted
            elif primaryStat is None and backupStat == 'success' and backupFinished is not None:
                delay = backupFinished - backupStarted
            else:
                flag = "n"

        # reward constants DQN1
        success_reward_weight = 1.0
        failure_penalty_weight = 20.0
        max_success_reward = 30.0
        min_failure_penalty = -3.0
        max_failure_penalty = -3 * max_success_reward  # i.e., -90.0


        if flag == "f":
            reward = -failure_penalty_weight * delay
            reward = max(min(reward, min_failure_penalty), max_failure_penalty)


        elif flag == "s":
            reward = success_reward_weight * (
                math.log(1 - (1 / math.exp(math.sqrt(delay)))) / math.log(0.995)
            )
            reward = min(reward, max_success_reward)        

        else:
            reward = None
        
        return reward, delay

    def add_train(self, episode):
        if len(self.local_model.replay_buffer) > 0:
            self.local_model.train_step()

        removeList = []
        for taskid, task_counter in self.pendingList:
            reward, delay = self.calcReward(taskid)
            if reward is not None:
                self.episodic_reward += reward
                self.episodic_delay += delay
                self.rewardsAll.append(reward)

                temp = list(self.tempbuffer[task_counter])
                temp[2] = reward
                self.tempbuffer[task_counter] = tuple(temp)
                s, a, r, s_ = self.tempbuffer[task_counter]
                self.local_model.store_transition((s, a, r, s_))
                self.local_model.train_step()
                removeList.append((taskid, task_counter))

        for t in removeList:
            self.pendingList.remove(t)
            task = self.env_state.get_task_by_id(t[0])
            self.task_Assignments_info.append((
                episode, task.id, task.vehicle_id,
                task.primaryNode.server_id, task.primaryStarted, task.primaryFinished,
                task.primaryStat, task.backupNode.server_id, task.backupStarted,
                task.backupFinished, task.backupStat, task.z, task.execution_status_flag
            ))

    def forward_result(self, task):
        for rsuid in task.vehicle.rsu_subgraph:
            dist_rsu = self.rsu_and_vehicle.get_rsu_by_id(rsuid)
            self.env.process(dist_rsu.receive_result(task))

    def receive_task(self, task):
        delay = 0 if task.original_RSU.rsu_id == self.rsu_id else self.env_state.calculate_e2e_delay(task.original_RSU.rsu_id, self.rsu_id, task.task_size)
        yield self.env.timeout(delay)
        task.selected_rsu_start_time = self.env.now 
    
    def receive_result(self, task):
        #delay = 0 if task.selected_RSU.rsu_id == self.rsu_id else self.env_state.calculate_e2e_delay(task.selected_RSU.rsu_id, self.rsu_id, params.task_result_size)
        delay = 0 if task.selected_RSU.rsu_id == self.rsu_id else 1
        yield self.env.timeout(delay)
        self.cached_results[task.id] = task
        #logger.info(f"                  {self.rsu_id} cached result of Task {task.id} at {self.env.now}")
        #self.env.process(self.remove_cached_result(task.id, params.task_timeout_caching))

    def remove_cached_result(self, task_id, timeout):
        yield self.env.timeout(timeout)
        self.cached_results.pop(task_id, None)
        #logger.info(f"RSU {self.rsu_id} pop cached result of Task {task_id} at {self.env.now}")

    def process_pendingList_and_log_result(self, episode):
        if self.taskCounter > 0:
            temp = list(self.tempbuffer[self.taskCounter])
            temp[3] = self.state
            self.tempbuffer[self.taskCounter] = tuple(temp)

        while self.pendingList:
            yield self.env.timeout(params.min_computation_demand)
            self.add_train(episode)

        self.ep_reward_list.append(self.episodic_reward)
        self.ep_delay_list.append(self.episodic_delay)
        avg_r = np.mean(self.ep_reward_list[-40:])
        avg_d = np.mean(self.ep_delay_list[-40:])
        self.avg_reward_list.append(avg_r)
        self.log_data.append((episode, avg_r, self.episodic_reward, avg_d, self.episodic_delay))
        print(f"{self.rsu_id}: Episode {episode} | Avg R: {avg_r:.2f} | This Episode R: {self.episodic_reward:.2f}")
  
    def reset(self, new_env):
        self.env = new_env
        self.episodic_reward = 0
        self.episodic_delay = 0
        self.tempbuffer.clear()
        self.taskCounter = 0
        self.pendingList.clear()
        self.cached_results.clear()

        
    @property
    def load(self):
        return len(self.pendingList)

