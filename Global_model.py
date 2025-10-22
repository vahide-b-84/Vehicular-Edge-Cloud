import torch
import numpy as np
import math
from params import params
from DQN_template import DQNAgent

class global_model:
    def __init__(self, env, env_state, num_rsus):
        self.env = env
        self.env_state = env_state
        self.num_states = 5 * num_rsus + num_rsus * (num_rsus - 1) + 3
        self.num_actions = num_rsus

        self.gamma = params.Global['gamma']
        self.tau = params.Global['tau']
        self.buffer_capacity = params.Global['buffer_capacity']
        self.batch_size = params.Global['batch_size']
        self.lr = params.Global['lr']
        self.activation = params.Global['activation']
        self.epsilon_start = params.Global['epsilon_start']
        self.epsilon_end = params.Global['epsilon_end']
        self.epsilon_decay = params.Global['epsilon_decay']
        self.hidden_layers = params.Global['hidden_layers']

        self.current_epsilon=self.epsilon_start

        self.rsu_selection_history = []
        self.G_state = []
        self.G_action = []
        self.tempbuffer = {}
        self.taskCounter = 1
        self.pendingList = []
        self.rewardsAll = []
        self.ep_reward_list = []
        self.ep_delay_list = []
        self.avg_reward_list = []
        self.episodic_reward = 0
        self.episodic_delay = 0
        self.log_data = []
        self.task_Assignments_info = []
        self.total_episodes = params.total_episodes

        self.agent = DQNAgent(
            num_states=self.num_states,
            num_actions=self.num_actions,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            gamma=self.gamma,
            lr=self.lr,
            tau=self.tau,
            buffer_size=self.buffer_capacity,
            batch_size=self.batch_size,
            activation=self.activation,
            hidden_layers=self.hidden_layers
        )

    
    def update_episode_epsilon(self,episode):
        epsilon = max(self.epsilon_end, self.epsilon_start - (episode / self.epsilon_decay))

        if len(self.ep_reward_list) >= 40:
            recent_rewards = self.ep_reward_list[-40:]
            avg_recent = np.mean(recent_rewards)
            last_reward = self.ep_reward_list[-1]

            drop_ratio = max(0.0, (avg_recent - last_reward) / avg_recent)

            if drop_ratio > 0.4:
                epsilon = max(epsilon, 0.4)
            elif drop_ratio > 0.2:
                epsilon = max(epsilon, 0.3)
            elif drop_ratio > 0.05:
                epsilon = max(epsilon, 0.2)

        self.current_epsilon = epsilon

    def add_train(self, this_episode):
        
        #if this_episode > self.warmup_episodes and len(self.agent.replay_buffer) > 0:
        if len(self.agent.replay_buffer) > 0:
            self.agent.train_step()

        removeList = []

        for taskid, task_counter in self.pendingList:
            reward, delay = self.calcReward(taskid)
            if reward is not None:
                self.episodic_reward += reward
                self.episodic_delay += delay
                self.rewardsAll.append(reward)

                (s, a, _, _) = self.tempbuffer[task_counter]
                task = self.env_state.get_task_by_id(taskid)
                s_next = self.env_state.get_state(task)

                self.tempbuffer[task_counter] = (s, a, reward, s_next)
                self.agent.store_transition((s, a, reward, s_next))

                self.agent.train_step()
                removeList.append((taskid, task_counter))
                
        for t in removeList:
            #print("task:", t[0] , "removed frome pending list of RSU:", task.selected_RSU.rsu_id)

            self.pendingList.remove(t)
            task = self.env_state.get_task_by_id(t[0])
            self.task_Assignments_info.append((
                this_episode,
                task.id,
                task.vehicle_id,
                task.original_RSU.rsu_id,
                task.submitted_time,
                task.selected_RSU.rsu_id,
                task.selected_rsu_start_time,
                task.delivered_RSU.rsu_id if task.delivered_RSU else "None",
                task.delivered_time if task.delivered_time else task.timeout_time,
                task.execution_status_flag,
                task.deadline_flag,
                task.final_status_flag
            ))
            
    def calcReward(self, taskID):
        task = self.env_state.get_task_by_id(taskID)
        reward=None
        delay=None

        if task is None:
            print("None task!")
            input("press Enter!")
            return None, None

        submitted_time = task.submitted_time
        deadline_flag = task.deadline_flag  # 'S' یا 'F'
        execution_flag = task.execution_status_flag  # 's' یا 'f'

        if submitted_time is None:
            print("task not submited yet in global model processing")
            input("press Enter!")
            return None, None

        # delay:
        if deadline_flag == 'S':
            delay = task.delivered_time - submitted_time
        elif deadline_flag == 'F':
            delay = task.timeout_time - submitted_time
        else: # deadline_flag == 'N': not reached deadline and not receive result yet
            #print("not reached deadline and not receive result yet")
            #input("press Enter!")
            return None, None  # not reached deadline and not receive result yet
        
        # reward and penalty weight:
        max_success_reward = 25.0
        min_success_reward = 5 # 1>10
        reward_decay_scale = 100.0  # key parameter for reward decay

        failure_penalty_weight = 10.0 # 
        min_failure_penalty = -3.0
        max_failure_penalty = -150.0 #

            
        # sucsessful task:
        if execution_flag == 's' and deadline_flag == 'S':
            task.final_status_flag = "s"
            reward = min_success_reward + (max_success_reward - min_success_reward) * math.exp(-delay / reward_decay_scale)

        # failed task:
        elif execution_flag == 'f' or deadline_flag == 'F':
            task.final_status_flag = "f"

            reward = -failure_penalty_weight * delay
            reward = max(min(reward, min_failure_penalty), max_failure_penalty)

        else:
            print("execution_flag:", execution_flag, "deadline_flag:", deadline_flag)
            input("press Enter!: None--------------------------------------------------")
        # print("taskID:", taskID, "reward:", reward, "delay:", delay)
        # input("press Enter to continue!")        
        return reward, delay

    def Recommend_RSU(self, task, rsus, this_episode):
        
        self.G_state = self.env_state.get_state(task)
       
        #RSU_index = self.agent.select_action(self.G_state, self.current_epsilon)
        RSU_index = self.agent.select_action(self.G_state, self.current_epsilon, use_softmax=True, temperature=1.5)
        self.rsu_selection_history.append(RSU_index)

        RSU_ID = f"RSU_{RSU_index}"

        task.selected_RSU = rsus.get(RSU_ID, None)

        self.tempbuffer[self.taskCounter] = (self.G_state, RSU_index, None, None)
        
        self.pendingList.append((task.id, self.taskCounter))
        self.taskCounter += 1
        
        return task.selected_RSU
        
    def process_pendingList_and_log_result(self, this_episode):
        if self.taskCounter > 1:
            print("Global model: process pending List! the last taskCounter:", self.taskCounter - 1)
            s, a, r, s_next = self.tempbuffer[self.taskCounter - 1]
            updated_s_next = self.env_state.get_state(self.env_state.get_task_by_id(self.taskCounter - 1))
            self.tempbuffer[self.taskCounter - 1] = (s, a, r, updated_s_next)

        while self.pendingList:
            yield self.env.timeout(params.min_computation_demand)
            self.add_train(this_episode)

        self.ep_reward_list.append(self.episodic_reward)
        self.ep_delay_list.append(self.episodic_delay)
        avg_reward = np.mean(self.ep_reward_list[-40:])
        avg_delay = np.mean(self.ep_delay_list[-40:])
        self.log_data.append((this_episode, avg_reward, self.episodic_reward, avg_delay,self.episodic_delay))

        print("DQN Global Model: Episode * {} * Avg Reward is ==> {}".format(this_episode, avg_reward), "This episode:", self.episodic_reward)
        self.avg_reward_list.append(avg_reward)
        #self.log_rsu_selection_stats(this_episode)

    def simple_Recommend_RSU(self, task):
        self.pendingList.append((task.id, self.taskCounter))
        self.taskCounter += 1
        
        task.selected_RSU = task.original_RSU # the approache use the source rsu as executed rsu
        RSU_index=int(task.selected_RSU.rsu_id.split("_")[1])
        self.rsu_selection_history.append(RSU_index)

        return task.selected_RSU

    def greedy_Recommend_RSU(self, task):
        self.pendingList.append((task.id, self.taskCounter))
        self.taskCounter += 1

        candidate_rsus = task.vehicle.rsu_subgraph
        best_rsu = None
        best_score = float('inf')

        max_counter = max(
            [self.env_state.RSU_and_Vehicle.get_rsu_by_id(r).taskCounter for r in candidate_rsus]
        ) or 1
        max_delay = 1  # 

        # 
        delays = {}
        for rsu_id in candidate_rsus:
            if rsu_id == task.original_RSU.rsu_id:
                delay = 0
            else:
                delay = self.env_state.calculate_e2e_delay(
                    task.original_RSU.rsu_id, rsu_id, task.task_size
                )
            delays[rsu_id] = delay
            if delay > max_delay:
                max_delay = delay

        # 
        a = 0.6  # 
        b = 0.4  # 

        for rsu_id in candidate_rsus:
            rsu = self.env_state.RSU_and_Vehicle.get_rsu_by_id(rsu_id)
            counter = rsu.taskCounter
            norm_counter = counter / max_counter
            norm_delay = delays[rsu_id] / max_delay

            score = a * norm_counter + b * norm_delay
            if score < best_score:
                best_score = score
                best_rsu = rsu

        task.selected_RSU = best_rsu
        RSU_index = int(task.selected_RSU.rsu_id.split("_")[1])
        self.rsu_selection_history.append(RSU_index)
        return best_rsu
        
    def simple_process_pendingList_and_log_result(self, this_episode):
               
            while self.pendingList:
                
                yield self.env.timeout(params.min_computation_demand)
                self.simple_add_train(this_episode)

            self.ep_reward_list.append(self.episodic_reward)
            self.ep_delay_list.append(self.episodic_delay)
            avg_reward = np.mean(self.ep_reward_list[-40:])
            avg_delay = np.mean(self.ep_delay_list[-40:])
            self.log_data.append((this_episode, avg_reward, self.episodic_reward, avg_delay,self.episodic_delay))

            print("No Global Model: Episode * {} * Avg Reward is ==> {}".format(this_episode, avg_reward), "This episode:", self.episodic_reward)
            self.avg_reward_list.append(avg_reward)
            #self.log_rsu_selection_stats(this_episode)
    
    def simple_add_train(self, this_episode):
                
        removeList = []

        for taskid, task_counter in self.pendingList:
            reward, delay = self.calcReward(taskid)
            if reward is not None:
                self.episodic_reward += reward
                self.episodic_delay += delay
                self.rewardsAll.append(reward)
                removeList.append((taskid, task_counter))

        for t in removeList:
            self.pendingList.remove(t)
            task = self.env_state.get_task_by_id(t[0])
            self.task_Assignments_info.append((
                this_episode,
                task.id,
                task.vehicle_id,
                task.original_RSU.rsu_id,
                task.submitted_time,
                task.selected_RSU.rsu_id,
                task.selected_rsu_start_time,
                task.delivered_RSU.rsu_id if task.delivered_RSU else "None",
                task.delivered_time if task.delivered_time else task.timeout_time,
                task.execution_status_flag,
                task.deadline_flag,
                task.final_status_flag
            ))
            #self.env_state.remove_task(t[0])

    def reset(self, new_SimPy_env,episode):
        self.env = new_SimPy_env
        self.episodic_reward = 0
        self.episodic_delay = 0
        self.tempbuffer = {}
        self.taskCounter = 1
        self.pendingList.clear()
        self.update_episode_epsilon(episode)
        self.rsu_selection_history = []

    def log_rsu_selection_stats(self, episode):
        from collections import Counter
        import pandas as pd
        import os
     

        rsu_counter = Counter(self.rsu_selection_history)
        total = sum(rsu_counter.values())

        rsu_usage_log = {f"RSU_{i}": rsu_counter.get(i, 0) / total for i in range(self.num_actions)}
        rsu_usage_log['episode'] = episode
        rsu_usage_log['epsilon'] = self.current_epsilon
        rsu_usage_log['avg_reward'] = np.mean(self.ep_reward_list[-40:])

        df = pd.DataFrame([rsu_usage_log])
        df.to_csv("rsu_behavior_log.csv", mode='a', header=not os.path.exists("rsu_behavior_log.csv"), index=False)
