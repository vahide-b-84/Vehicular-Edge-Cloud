import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import numpy as np
import sumolib
import json
import networkx as nx
from params import params
import math
from heapq import heappush, heappop
from collections import defaultdict

class GraphNetwork:
    def __init__(self):
        self.num_nodes = None #load_sumo_data
        self.num_rsus =  None #load_sumo_data
        self.G = nx.Graph()
        self.positions = {}  # Junction positions , load_sumo_data
        self.rsus = {}  # RSUs dictionary (position, range) , extract_rsus_from_additional
        self.vehicles = {}  # Vehicle data dictionary , extract_vehicle_data
        # --- NEW ---
        self.RSU_Pairs_failure_rate = {}
        self.RSU_distances = {}
        self.RSU_link_bandwidths = {}

    # Method to load data from SUMO network
    def load_sumo_data(self, net_file, additional_file, route_file):
        # Load network from SUMO using sumolib
        net = sumolib.net.readNet(net_file)

        # Extract junctions (nodes) and their coordinates
        for junction in net.getNodes():
            
            node_id = junction.getID()
            node_pos = junction.getCoord()
            self.positions[node_id] = node_pos
            self.G.add_node(node_id, pos=node_pos)  # Add position attribute to the node

        # Extract edges and add them to the graph
        for edge in net.getEdges():
            edge_id = edge.getID()  # Get edge ID as stored in SUMO
            from_junction = edge.getFromNode().getID()
            to_junction = edge.getToNode().getID()
            edge_length = edge.getLength()

            self.G.add_edge(from_junction, to_junction, length=edge_length, id=edge_id)

        # Extract RSU data from the additional file
        self.extract_rsus_from_additional(additional_file)

        # Extract vehicle data from the route file
        self.extract_vehicle_data(route_file)

    # Method to extract RSUs from the additional file (test.add.xml)
    def extract_rsus_from_additional(self, additional_file):
        tree = ET.parse(additional_file)
        root = tree.getroot()

        rsus = {}  # Store RSUs in the desired format

        # Sort POIs based on their default SUMO ID (e.g., "poi_0", "poi_1", ...)
        sorted_pois = sorted(root.findall("poi"), key=lambda poi: int(poi.get("id").split("_")[1]))

        for idx, poi in enumerate(sorted_pois):  # Ensure order remains consistent
            x = float(poi.get("x"))
            y = float(poi.get("y"))

            # Generate a random range for RSU coverage
            range_radius=random.uniform(params.RSU_radius[0],params.RSU_radius[1])

            # Generate the number of edge servers for the RSU based on params
            num_edge_servers = random.randint(params.RSUs_EDGE_SERVERS[0], params.RSUs_EDGE_SERVERS[1])

            # Assign new RSU ID format based on sorted order
            rsu_id = f"RSU_{idx}"  

            # Store in the requested JSON format
            rsus[rsu_id] = {
                "position": [x, y],
                "range": range_radius,
                "edge_server_numbers": num_edge_servers
            }

        self.rsus = rsus  # Save RSU data in the class

    # Method to extract vehicle data from the routes file (test.rou.xml)
    def extract_vehicle_data(self, route_file):
        tree = ET.parse(route_file)
        root = tree.getroot()

        # Build a dictionary of routes (mapping route ID to list of edges)
        routes_dict = {route.get("id"): route.get("edges").split() for route in root.findall("route")}

        vehicle_data = {}

        for vehicle in root.findall("vehicle"):
            vehicle_id = vehicle.get("id")
            route_id = vehicle.get("route")
            speed = float(vehicle.get("departSpeed", 0))

            # Ensure the route exists
            if route_id not in routes_dict:
                print(f"⚠ Warning: Route {route_id} not found for vehicle {vehicle_id}")
                continue

            path = routes_dict[route_id]  
            
            # Generate task times
            task_arrival_rate = np.random.uniform(params.TASK_ARRIVAL_RATE_range[0], params.TASK_ARRIVAL_RATE_range[1])
            task_times = []
            t = 0
            for _ in range(params.Vehicle_taskno):
                inter_arrival_time = np.random.poisson(1 / task_arrival_rate)
                t += inter_arrival_time
                task_times.append(t)

            # Compute RSU subgraph for this vehicle
            rsu_subgraph = self.get_rsu_subgraph(path)

            # Store vehicle data
            vehicle_data[vehicle_id] = {
                "path": path,
                "speed": speed,
                "task_times": task_times,
                "rsu_subgraph": rsu_subgraph  # Add the RSU subgraph information
            }

        self.vehicles = vehicle_data

    # New helper method to compute RSU subgraph for a vehicle
    def get_rsu_subgraph(self, path):
        rsu_subgraph = set()  # Set to store RSUs covering the vehicle's path

        # Iterate over each edge in the vehicle's path
        for edge_id in path:
            for rsu_id, rsu in self.rsus.items():
                rsu_position = rsu["position"]
                rsu_range = rsu["range"]
                if self.check_edge_rsu_coverage(edge_id, rsu_position, rsu_range):
                    rsu_subgraph.add(rsu_id)

        return list(rsu_subgraph)

    # Method to check if an edge is within the RSU's coverage range
    def check_edge_rsu_coverage(self, edge_id, rsu_position, rsu_range):
        # Extract edge data from XML (assuming you have a method to get this data from your XML)
        edge_data = self.get_edge_data_from_xml(edge_id)
        start_node = edge_data['from']
        end_node = edge_data['to']

        # Extract positions of the start and end nodes
        start_pos = self.G.nodes[start_node]['pos']
        end_pos = self.G.nodes[end_node]['pos']

        # RSU position
        rx, ry = rsu_position

        # Check if both start and end points are inside the RSU's range
        if (
            (start_pos[0] - rx)**2 + (start_pos[1] - ry)**2 <= rsu_range**2 and
            (end_pos[0] - rx)**2 + (end_pos[1] - ry)**2 <= rsu_range**2
        ):
            return True  # Both points are inside the circle; fully covered

        # Vector for the edge (line segment)
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]

        # Coefficients for the quadratic equation
        A = dx**2 + dy**2
        B = 2 * (dx * (start_pos[0] - rx) + dy * (start_pos[1] - ry))
        C = (start_pos[0] - rx)**2 + (start_pos[1] - ry)**2 - rsu_range**2

        # Discriminant to check for intersection
        discriminant = B**2 - 4 * A * C

        if discriminant < 0:
            return False  # No intersection

        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-B - sqrt_discriminant) / (2 * A)
        t2 = (-B + sqrt_discriminant) / (2 * A)

        # Check if intersection points are within the edge's range
        if 0 <= t1 <= 1 or 0 <= t2 <= 1:
            return True

        return False

    def get_edge_data_from_xml(self, edge_id):
        tree = ET.parse("SUMO/test.net.xml")
        root = tree.getroot()

        for edge in root.findall("edge"):
            if edge.get("id") == edge_id:
                from_node = edge.get("from")
                to_node = edge.get("to")
                return {"from": from_node, "to": to_node}

        raise ValueError(f"Edge {edge_id} not found in the network file.")

    def set_RSU_failure_rate_bandwidths_distances(self):
        self.RSU_Pairs_failure_rate = {}
        self.RSU_distances = defaultdict(dict)
        self.RSU_link_bandwidths = {}

        rsu_ids = list(self.rsus.keys())
        for i, rsu_i in enumerate(rsu_ids):
            pos_i = self.rsus[rsu_i]["position"]
            for j, rsu_j in enumerate(rsu_ids):
                if i == j:
                    continue
                # failure rate (symmetric)
                fr = np.random.uniform(params.link_failure_rate_range[0],
                                    params.link_failure_rate_range[1])
                self.RSU_Pairs_failure_rate[(rsu_i, rsu_j)] = fr
                self.RSU_Pairs_failure_rate[(rsu_j, rsu_i)] = fr

                # distance (symmetric)
                pos_j = self.rsus[rsu_j]["position"]
                d = math.hypot(pos_i[0]-pos_j[0], pos_i[1]-pos_j[1])
                self.RSU_distances[rsu_i][rsu_j] = d
                self.RSU_distances[rsu_j][rsu_i] = d

                # bandwidth (symmetric)
                bw = np.random.uniform(params.RSU_LINK_BANDWIDTH_RANGE[0],
                                    params.RSU_LINK_BANDWIDTH_RANGE[1])
                self.RSU_link_bandwidths[(rsu_i, rsu_j)] = bw
                self.RSU_link_bandwidths[(rsu_j, rsu_i)] = bw
    # Method to save the graph and data to JSON start_pos = self.graph_network.G.nodes[start_node]['pos']
    def save_graph(self):
        """Save the graph in JSON format, including vehicle data if provided."""
        graph_data = {
            "nodes": {node: {"pos": self.G.nodes[node]["pos"]} for node in self.G.nodes},
            "edges": [(u, v, self.G[u][v]["length"], self.G[u][v].get("id", "")) for u, v in self.G.edges],  # ذخیره id یال‌ها

            "rsus": self.rsus,  # RSUs information saved separately
            # --- NEW: serialize tuple-keys as strings: "src->dst"
            "RSU_Pairs_failure_rate": {f"{a}->{b}": v for (a, b), v in self.RSU_Pairs_failure_rate.items()},
            "RSU_distances": {src: dsts for src, dsts in self.RSU_distances.items()},
            "RSU_link_bandwidths": {f"{a}->{b}": v for (a, b), v in self.RSU_link_bandwidths.items()},
        }

        # Add vehicle data if available
        if self.vehicles:
            graph_data["vehicles"] = self.vehicles

        # Save to file
        filename = "graph_data.json"
        with open(filename, "w") as f:
            json.dump(graph_data, f, indent=4)
        print(f"Graph saved to {filename}")
    
    def load_graph(self, filename="graph_data.json"):
        """Load graph, RSUs, and vehicle data from JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)

        self.G.clear()
        self.positions.clear()
        self.rsus.clear()
        self.vehicles.clear()
        self.RSU_Pairs_failure_rate.clear()
        self.RSU_link_bandwidths.clear()
        self.RSU_distances.clear

        for node, attrs in data["nodes"].items():
            self.G.add_node(node, pos=attrs["pos"])

            self.positions[node] = attrs["pos"] 

        for u, v, length, edge_id in data["edges"]:
            self.G.add_edge(u, v, length=length, edge_id=edge_id )

        self.rsus = data["rsus"]

        self.vehicles = data.get("vehicles", {})  
        # Set number of nodes and RSUs based on the loaded data
        self.num_nodes = len(self.positions)
        self.num_rsus = len(self.rsus)

        # --- NEW: de-serialize back to tuples
        self.RSU_Pairs_failure_rate = {
            tuple(k.split("->")): v for k, v in data.get("RSU_Pairs_failure_rate", {}).items()
        }
        self.RSU_distances = data.get("RSU_distances", {})
        self.RSU_link_bandwidths = {
            tuple(k.split("->")): v for k, v in data.get("RSU_link_bandwidths", {}).items()
        }
        print("Graph successfully loaded.")

    def plot_graph(self):
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        pos = {node: (data["pos"][0], data["pos"][1]) for node, data in self.G.nodes(data=True)}
        nx.draw(self.G, pos, with_labels=True, node_color='orange', edge_color='gray', node_size=50, font_size=8)
        
        for rsu_id, rsu in self.rsus.items():
            x, y = rsu["position"]
            range_radius = rsu["range"]

            circle = plt.Circle((x, y), range_radius, color='blue', alpha=0.1, zorder=0)
            ax.add_patch(circle)

            
            plt.scatter(x, y, color='red', marker='^', s=100, zorder=1)

           
            plt.text(x, y + 10, rsu_id, color='black', fontsize=5, ha='center')

        legend_elements = [
            Line2D([0], [0], marker='^', color='w', label='RSU Node', markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Regular Node', markerfacecolor='orange', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc="upper right")
        
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Graph with RSU Coverage and Nodes")
        plt.grid(True)
        plt.show()
    # Method to generate a task queue from the vehicle data
    def generate_Task_Queue(self):
        # Reading the JSON file
        with open('graph_data.json', 'r') as file:
            data = json.load(file)

        # Creating a priority queue (heap) to merge the sorted task lists
        heap = []
        for vehicle_id, vehicle_info in data["vehicles"].items():
            task_times = vehicle_info["task_times"]
            if task_times:  # Check if the list is not empty
                # Add the first task of each vehicle to the priority queue
                heappush(heap, (task_times[0], vehicle_id, 0))  # (task time, vehicle ID, index in the list)

        # Sorting and calculating interarrival_time
        interarrival_list = []
        previous_time = 0
        while heap:
            current_time, vehicle_id, index = heappop(heap)
            interarrival_time = current_time - previous_time
            interarrival_list.append({
                "vehicle_id": vehicle_id,
                "time": current_time,
                "interarrival_time": interarrival_time
            })
            previous_time = current_time

            # Add the next task from the same vehicle to the priority queue
            next_index = index + 1
            task_times = data["vehicles"][vehicle_id]["task_times"]
            if next_index < len(task_times):
                heappush(heap, (task_times[next_index], vehicle_id, next_index))

        # Displaying the final list
        '''for item in interarrival_list:
            print(item)'''

        # Saving the data to a JSON file
        output_file = 'taskQueue.json'
        with open(output_file, 'w') as outfile:
            json.dump(interarrival_list, outfile, indent=4)

        print(f"The task scheduling list has been saved in the file {output_file}.")

