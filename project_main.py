# main.py
import sys
import os
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from params import params
from mainLoop import MainLoop


# Function to run your simulation
def run_simulation():
    ml = MainLoop()
    ml.EP()


def _get_scenario_and_p():
    """
    Supported scenarios:
      - base            => p = 0.0
      - trajectory_noise => p = params.trajectory_noise_p
      - missing_data    => p = params.missing_data_p
    """
    scenario = getattr(params, "scenario", "base")

    if scenario == "trajectory_noise":
        p = getattr(params, "trajectory_noise_p", 0.0)
        scenario_folder = "trajectory_noise"
    elif scenario == "missing_data":
        p = getattr(params, "missing_data_p", 0.0)
        scenario_folder = "missing_data"
    else:
        # base 
        p = 0.0
        scenario_folder = "base"

    return scenario_folder, p


def clear_results_folders():

    if os.path.exists("rsu_behavior_log.csv"):
        print("Removing rsu_behavior_log.csv")
        os.remove("rsu_behavior_log.csv")

    scenario_folder, p = _get_scenario_and_p()

    # subfolder name: e.g., dqn_0_30
    model_tag = f"{params.model_summary}_{p:.2f}".replace(".", "_")

    folders = ["heterogeneous_results", "homogeneous_results"]

    for folder in folders:
        # NEW PATH: <folder>/<scenario>/<model_tag>
        subfolder = os.path.join(folder, scenario_folder, model_tag)

        if os.path.exists(subfolder) and os.path.isdir(subfolder):
            print(f"Folder detected: {subfolder}")
            ans = input(f"Are you sure you want to DELETE this folder and all contents? (y/N): ")

            if ans.lower() == "y":
                shutil.rmtree(subfolder)
                print(f"Folder {subfolder} removed.")
            else:
                print(f"Skipped deleting {subfolder}.")
        else:
            print(f"Folder {subfolder} does not exist, skipping.")


def main():

    clear_results_folders()

    scenario_types = ["heterogeneous"]  # , "homogeneous"
    permutation_numbers = [3]  # example

    for scenario_type in scenario_types:
        for permutation_number in permutation_numbers:
            print(
                f"##################   Running simulation for {scenario_type} scenario, permutation {permutation_number}  ####################"
            )

            # Set the scenario type and permutation number
            params.SCENARIO_TYPE = scenario_type
            params.Permutation_Number = permutation_number

            if permutation_number == 1:
                failure = "low"
            elif permutation_number == 3:
                failure = "high"
            else:
                failure = "med"

            params.alpha_edge = params.Alpha['edge'][scenario_type][failure]
            params.alpha_cloud = params.Alpha['cloud'][scenario_type][failure]

            # Run the simulation for the current scenario and permutation
            run_simulation()


if __name__ == "__main__":
    main()
