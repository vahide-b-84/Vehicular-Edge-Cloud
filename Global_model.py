# Level-1 model
import torch
import numpy as np
import math
from params import params
import random
from DQN_template import DQNAgent
import os
import csv


class global_model:
    def __init__(self, env, env_state, num_rsus):
        self.env = env
        self.env_state = env_state
        self.num_states = 5 * num_rsus + num_rsus * (num_rsus - 1) + 3
        self.num_actions = num_rsus
        self.use_softmax=True
        self.epsilon_start = params.Global['epsilon_start']
        self.epsilon_end = params.Global['epsilon_end']
        self.epsilon_decay = params.Global['epsilon_decay']
        self.current_epsilon=self.epsilon_start
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
        # NEW: per-episode reproducible RNG (NOT recreated per task)
        self.net_rng = random.Random(self.total_episodes)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if params.model_summary == "dqn":
            # Global DQN
            self.agent = DQNAgent(
                num_states=self.num_states,
                num_actions=self.num_actions,
                device=device,
                gamma=params.Global['gamma'],
                lr=params.Global['lr'],
                tau=params.Global['tau'],
                buffer_size=params.Global['buffer_capacity'],
                batch_size=params.Global['batch_size'],
                activation=params.Global['activation'],
                hidden_layers=params.Global['hidden_layers'],
            )
            print("Global model: using DQNAgent")

    def update_episode_epsilon(self,episode):
        epsilon = max(self.epsilon_end, self.epsilon_start - (episode / self.epsilon_decay))

        if len(self.ep_reward_list) >= 40:
            recent_rewards = self.ep_reward_list[-40:]
            avg_recent = np.mean(recent_rewards)
            last_reward = self.ep_reward_list[-1]

            drop_ratio = max(0.0, (avg_recent - last_reward) / avg_recent)

            if drop_ratio > 0.4:
                epsilon = max(epsilon, 0.25)
            elif drop_ratio > 0.2:
                epsilon = max(epsilon, 0.20)
            elif drop_ratio > 0.05:
                epsilon = max(epsilon, 0.15)

        self.current_epsilon = epsilon

    def add_train(self, this_episode):
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
                if params.scenario=="missing_data":
                    if self.net_rng.random() > params.missing_data_p:
                        self.sent_cnt += 1
                        self.agent.store_transition((s, a, r, s_))
                        self.agent.train_step()
                    else:
                        self.drop_cnt += 1
                else:
                    self.agent.store_transition((s, a, r, s_))
                    self.agent.train_step()
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
        

    def calcReward(self, taskID):
        task = self.env_state.get_task_by_id(taskID)
        reward=None
        delay=None

        if task is None:
            print("None task!")
            input("press Enter!")
            return None, None

        submitted_time = task.submitted_time
        deadline_flag = task.deadline_flag  # 'S' /'F'
        execution_flag = task.execution_status_flag  # 's' /'f'

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
        #add this state to the last pending tuple in tempBuffer
        if self.taskCounter>1:
            tempx=list(self.tempbuffer[self.taskCounter-1])
            tempx[3]=self.G_state
            self.tempbuffer[self.taskCounter-1]=tuple(tempx)
            self.add_train(this_episode) 

        RSU_index = self.agent.select_action(self.G_state, self.current_epsilon, self.use_softmax, temperature=1.5)

        RSU_ID = f"RSU_{RSU_index}"

        task.selected_RSU = rsus.get(RSU_ID, None)

        self.tempbuffer[self.taskCounter] = (self.G_state, RSU_index, None, None)
        
        self.pendingList.append((task.id, self.taskCounter))
        self.taskCounter += 1
        
        return task.selected_RSU
        
    def process_pendingList_and_log_result(self, this_episode):
        
        print("Global model: process pending List! the last taskCounter:", self.taskCounter - 1)
        s, a, r, s_next = self.tempbuffer[self.taskCounter - 1]
        s_next = self.G_state
        self.tempbuffer[self.taskCounter - 1] = (s, a, r, s_next)

        while self.pendingList:
            yield self.env.timeout(params.min_computation_demand)
            self.add_train(this_episode)

        self.ep_reward_list.append(self.episodic_reward)
        self.ep_delay_list.append(self.episodic_delay)
        avg_reward = np.mean(self.ep_reward_list[-40:])
        avg_delay = np.mean(self.ep_delay_list[-40:])
        self.log_data.append((this_episode, avg_reward, self.episodic_reward, avg_delay,self.episodic_delay))

        print("Global Model: Episode * {} * Avg Reward is ==> {}".format(this_episode, avg_reward), "This episode:", self.episodic_reward)
        self.avg_reward_list.append(avg_reward)
        if params.scenario=="missing_data":
            # ================= DROP RATIO LOGGING =================
            total = self.drop_cnt + self.sent_cnt
            drop_ratio = (self.drop_cnt / total) if total else 0.0

            # scenario folder (فعلاً فقط heterogeneous)
            base_folder = "heterogeneous_results"

            # subfolder name: dqn_0_30  (مثال)
            scenario = getattr(params, "scenario", "base")

            if scenario == "missing_data":
                p = getattr(params, "missing_data_p", 0.0)
                scenario_folder = "missing_data"
            else:
                # base (یا هر مقدار ناشناخته)
                p = 0.0
                scenario_folder = "base"

            # subfolder name: dqn_0_30  (مثال)
            subfolder_name = f"{params.model_summary}_{p:.2f}".replace(".", "_")

            # now: heterogeneous_results/missing_data/dqn_0_30  یا heterogeneous_results/base/dqn_0_00
            save_dir = os.path.join(base_folder, scenario_folder, subfolder_name)
            os.makedirs(save_dir, exist_ok=True)

            # csv file name
            csv_path = os.path.join(
                save_dir,
                f"DROP_Ratio_{params.model_summary}.csv"
            )

            # write header only once
            write_header = not os.path.exists(csv_path)

            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "episode",
                        "sent_transitions",
                        "dropped_transitions",
                        "drop_ratio"
                    ])

                writer.writerow([
                    this_episode,
                    self.sent_cnt,
                    self.drop_cnt,
                    round(drop_ratio, 4)                
                ])


    def reset(self, new_SimPy_env,episode):
        self.env = new_SimPy_env
        self.episodic_reward = 0
        self.episodic_delay = 0
        self.tempbuffer = {}
        self.taskCounter = 1
        self.pendingList.clear()
        self.update_episode_epsilon(episode)
        # drop counters

        self.drop_cnt = 0
        self.sent_cnt = 0
                
