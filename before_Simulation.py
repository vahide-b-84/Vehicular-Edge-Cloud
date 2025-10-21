
from Sumo_Graph import GraphNetwork
import generate_server_and_task_parameters


network = GraphNetwork()
network.load_sumo_data("SUMO/test.net.xml", "SUMO/test.add.xml", "SUMO/test.rou.xml")
network.set_RSU_failure_rate_bandwidths_distances()
# Save the graph 
network.save_graph()


network.generate_Task_Queue()
network.plot_graph()          

generate_server_and_task_parameters.main()