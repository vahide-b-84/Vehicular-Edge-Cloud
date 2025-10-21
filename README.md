# Vehicular Edge-Cloud Computing (VECC) - Fault-Tolerant Task Offloading

## Description
This project proposes a **mobility-aware and fault-tolerant task offloading framework** for Vehicular Edge-Cloud Computing (VECC) environments. The system integrates **mobility awareness** and **fault tolerance** through a **bi-level Deep Q-Network (DQN)** approach, which optimizes task offloading, network latency, and task execution reliability under dynamic vehicular environments.

The framework adapts to **vehicular movement**, **link variations**, and **node failures**, ensuring efficient task execution despite unpredictable network conditions.

## Major Contributions
- **Proposed VECC architecture**: Integrates mobility awareness and fault-tolerant mechanisms through hierarchical learning. The system dynamically adapts to vehicular movement, link variations, and node failures, ensuring reliable task execution under changing network conditions.
- **Bi-level DQN structure**: A level-1 DQN agent handles RSU selection using network-wide insights, while level-2 agents at RSUs manage task allocation and recovery. The model selects the best recovery strategy based on task characteristics, RSU conditions, and failure probabilities, achieving a balanced trade-off between latency and reliability.
- **Analytical system model**: Captures both computational and communication delays in multi-RSU environments, incorporating mobility-induced latency and link-failure probabilities.
- **Python-based simulation framework**: Integrates **SUMO** and **SimPy** for evaluating the proposed method under various mobility and traffic conditions.

## Getting Started

### Prerequisites
Before running this project, ensure that the following tools and libraries are installed:

- **Python 3.10.18** (or newer)
- **Anaconda** (for managing the Python environment)
- **SUMO** (for vehicular mobility simulation)
- **SimPy** (for discrete event simulation)
- **PyTorch** (for implementing the DQN models)
- **NumPy**, **SciPy**, **Pandas**, **Matplotlib**, **Openpyxl**, **Subprocess**, and **OS** (for various Python tasks)

You can create an environment with these dependencies using the `requirements.txt` file.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/Vehicular-Edge-Cloud.git
   cd Vehicular-Edge-Cloud
2.	Install the required dependencies:
   pip install -r requirements.txt
3.	SUMO Installation:
   o	Follow the instructions on the SUMO website to install SUMO.
   o	Ensure that TraCI is properly set up to interact with Python.


