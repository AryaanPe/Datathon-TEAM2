import csv
import networkx as nx
import matplotlib.pyplot as plt
# 1. Load the CSV data
def load_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip the header row
        nodes = headers[1:]   # Extract city names from headers

        graph_data = {}
        for row in reader:
            origin_city = row[0]
            for i in range(1, len(row)):
                destination_city = headers[i]
                distance = int(row[i])  # Convert distance to integer
                graph_data[(origin_city, destination_city)] = distance
    return nodes, graph_data

# 2. Create the graph
def create_graph(nodes, graph_data):
    G = nx.Graph()
    G.add_nodes_from(nodes)

    # Adjust for the format of graph_data:
    G.add_weighted_edges_from((key[0], key[1], value) for key, value in graph_data.items())

    return G

# 3. Find the optimized path with alternate options
def find_optimized_path(G, start, end, unavailable_nodes=[]):
    try:
        primary_path = nx.dijkstra_path(G, start, end, weight='weight')
        primary_time = nx.dijkstra_path_length(G, start, end, weight='weight')
    except nx.NetworkXNoPath:
        print("No path exists between the specified nodes.")
        return None, None

    # Check for unavailable nodes and find alternate paths
    if unavailable_nodes:
        for unavailable_node in unavailable_nodes:
            if unavailable_node in primary_path:
                G.remove_node(unavailable_node)
                try:
                    alternate_path = nx.dijkstra_path(G, start, end, weight='weight')
                    alternate_time = nx.dijkstra_path_length(G, start, end, weight='weight')
                    return primary_path, primary_time, alternate_path, alternate_time
                except nx.NetworkXNoPath:
                    pass
                G.add_node(unavailable_node)  # Restore the removed node

    return primary_path, primary_time, None, None

def plot(G,path):
  pos = nx.spring_layout(G)  # You can use different layout algorithms
  nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8)
  nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange', node_size=700)
  nx.draw_networkx_edges(G, pos, width=2, edge_color='gray')
  plt.show()


filename = 'Cities_FlightDuration_Mins.csv'
nodes, graph_data = load_data(filename)
G = create_graph(nodes, graph_data)

# start_node = 'Kanpur'
# end_node = 'Nagpur'
# unavailable_nodes = ['Delhi']  # Example of unavailable nodes

# primary_path, primary_time, alternate_path, alternate_time = find_optimized_path(G, start_node, end_node, unavailable_nodes)

# print("Primary path:", primary_path, "(Time:", primary_time, ")")
# if alternate_path:
#     print("Alternate path:", alternate_path, "(Time:", alternate_time, ")")
#     plot(G,alternate_path)
