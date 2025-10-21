import xml.etree.ElementTree as ET
import numpy as np
from configuration import parameters



def extract_max_speed_from_rou(xml_file="SUMO/test.rou.xml", vehicle_type="Car"):
    """Extract maxSpeed from vType definition in .rou file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for vtype in root.findall("vType"):
        if vtype.get("id") == vehicle_type:
            return float(vtype.get("maxSpeed"))
    return 30.0  # fallback default


def extract_rsu_positions_from_additional(additional_file="SUMO/test.add.xml"):
    """Parse RSU positions from the .add.xml file and return as list of coordinates."""
    tree = ET.parse(additional_file)
    root = tree.getroot()
    positions = []
    for poi in root.findall("poi"):
        x = float(poi.get("x"))
        y = float(poi.get("y"))
        positions.append(np.array([x, y]))
    return positions


def compute_max_rsu_distance(additional_file="SUMO/test.add.xml"):
    """Compute the max Euclidean distance between any two RSUs."""
    positions = extract_rsu_positions_from_additional(additional_file)
    max_dist = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            max_dist = max(max_dist, dist)
    return max_dist



def estimate_max_e2e_delay():
    max_task_size = parameters.TASK_SIZE_RANGE[1]
    min_bandwidth = parameters.RSU_LINK_BANDWIDTH_RANGE[0]
    max_dist = compute_max_rsu_distance()
    speed = parameters.network_speed
    alpha = parameters.Queuing_alpha
    beta = parameters.beta
    max_L_ratio = 1
    P = min(parameters.link_failure_rate_range[1] + beta * max_L_ratio, 0.7)
    E_N = (1 / (1 - P)) - 1

    D_trans = max_task_size / min_bandwidth
    D_prop = max_dist / speed
    D_queue = alpha * max_L_ratio
    D_retrans = E_N * (D_trans + D_prop)
    return D_trans + D_prop + D_queue + D_retrans
