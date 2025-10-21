#ddpg_main.py
import sys
import os
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from params import params  # Import from current directory
from mainLoop import MainLoop  # Import from current directory

# Function to run your simulation
def run_simulation():
    ml=MainLoop()
    ml.EP() 

def clear_results_folders():
    
    if os.path.exists("rsu_behavior_log.csv"):
        print("Removing rsu_behavior_log.csv")
        os.remove("rsu_behavior_log.csv")

    folders = ["heterogeneous_results", "homogeneous_results"]
    for folder in folders:
        subfolder = os.path.join(folder, params.model_summary)
        if os.path.exists(subfolder) and os.path.isdir(subfolder):
            print(f"Removing model-specific folder: {subfolder}")
            shutil.rmtree(subfolder)
            print(f"Folder {subfolder} removed.")
        else:
            print(f"Folder {subfolder} does not exist, skipping.")

def main():
    
    clear_results_folders()  # 
    scenario_types = ["heterogeneous"] #, "homogeneous", "heterogeneous"
    permutation_numbers = [3] #  for range run: 1 to 24 (1,25) /////  [1,7,13,19] seprate permutation
    for scenario_type in scenario_types:
        for permutation_number in permutation_numbers:
            print(f"##################   Running simulation for {scenario_type} scenario, permutation {permutation_number}  ####################")
            
            # Set the scenario type and permutation number
            params.SCENARIO_TYPE = scenario_type
            params.Permutation_Number = permutation_number
            if permutation_number==1:
                failure="low" 
            elif permutation_number==3:
                failure="high"
            else:
                failure="med"
            params.alpha_edge=params.Alpha['edge'][scenario_type][failure]
            params.alpha_cloud=params.Alpha['cloud'][scenario_type][failure]

            # Run the simulation for the current scenario and permutation
            run_simulation()

if __name__ == "__main__":
    main()