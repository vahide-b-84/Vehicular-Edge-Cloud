# Vehicular Edge-Cloud Computing (VECC) - Fault-Tolerant Task Offloading

## Description
This project proposes a mobility-aware and fault-tolerant task offloading framework for Vehicular Edge-Cloud Computing (VECC) environments. The system integrates mobility awareness and fault tolerance through a bi-level Deep Q-Network (DQN) approach, which optimizes task offloading, network latency, and task execution reliability under dynamic vehicular environments. The framework adapts to vehicular movement, link variations, and node failures, ensuring efficient task execution despite unpredictable network conditions.

## Major Contributions
- **Proposed VECC architecture**: Integrates mobility awareness and fault-tolerant mechanisms through hierarchical learning. The system dynamically adapts to vehicular movement, link variations, and node failures, ensuring reliable task execution under changing network conditions.
- **Bi-level DQN structure**: A level-1 DQN agent handles RSU selection using network-wide insights, while level-2 agents at RSUs manage task allocation and recovery. The model selects the best recovery strategy based on task characteristics, RSU conditions, and failure probabilities, achieving a balanced trade-off between latency and reliability.
- **Analytical system model**: Captures both computational and communication delays in multi-RSU environments, incorporating mobility-induced latency and link-failure probabilities.
- **Python-based simulation framework**: Integrates SUMO and SimPy for evaluating the proposed method under various mobility and traffic conditions.

## Getting Started

### Prerequisites
Before running this project, ensure that the following tools and libraries are installed:
- Python 3.10.18 (or newer)
- Anaconda (for managing the Python environment)
- SUMO (for vehicular mobility simulation)
- SimPy (for discrete event simulation)
- PyTorch (for implementing the DQN models)
- NumPy, SciPy, Pandas, Matplotlib, Openpyxl, Subprocess, and OS (for various Python tasks)

You can create an environment with these dependencies using the `requirements.txt` file.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/username/Vehicular-Edge-Cloud.git
    cd Vehicular-Edge-Cloud
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. **SUMO Installation**:
   - Follow the instructions on the [SUMO website](https://www.eclipse.org/sumo/) to install SUMO.
   - Ensure that TraCI is properly set up to interact with Python.

## Running the Simulation

### Step 1: Configuration
Before running the simulation, ensure that the configuration file is updated with the correct parameters based on your desired simulation scenario.

### Step 2: Pre-Simulation Setup
Run the `before_Simulation.py` script to set up the simulation environment and generate the required data:

```bash
python before_Simulation.py

### Step 3: Running the Main Simulation
Once the setup is complete, run the project_main.py script to start the task offloading simulation:

```bash
python project_main.py

## Example usage
To run the full simulation pipeline, use the following commands:
```bash
python before_Simulation.py  # Prepare the simulation environment
python project_main.py  # Start the task offloading simulation
The simulation results, including task completion times and failure rates, will be saved in the results/ directory.

## Notes:

- **The before_Simulation.py script sets up the simulation environment, so it must be run before starting the main simulation.

- **Ensure that SUMO and TraCI are installed and configured correctly before running the simulation.

