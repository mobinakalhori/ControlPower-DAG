import random
import time
import csv
import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd  #Create and save an Excel file

def get_desktop_path():
    home = os.path.expanduser('~')
    onedrive_desktop = os.path.join(home, 'OneDrive', 'Desktop')
    normal_desktop = os.path.join(home, 'Desktop')
    
    if os.path.exists(onedrive_desktop):
        return onedrive_desktop
    else:
        return normal_desktop

def generate_dag(num_nodes, avg_degree):
    if num_nodes <= 1:
        return {i: [] for i in range(num_nodes)}
    
    p = avg_degree / (num_nodes - 1)
    if p > 1:
        p = 1.0
        
    adj_list = {i: [] for i in range(num_nodes)}
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < p:
                adj_list[i].append(j)
                
    return adj_list

def save_to_csv(adj_list, filename="dag_output.csv"):
    desktop_path = get_desktop_path()
    full_path = os.path.join(desktop_path, filename)
    
    with open(full_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Target"])
        for source, targets in adj_list.items():
            for target in targets:
                writer.writerow([source, target])
    print(f"\n The adjacency list file was successfully saved to the desktop: \n{full_path}")

def save_adjacency_matrix_to_excel(adj_list, filename="dag_adjacency_matrix.xlsx"):
    desktop_path = get_desktop_path()
    full_path = os.path.join(desktop_path, filename)
    
    num_nodes = len(adj_list)
    
    #Initialize the matrix with all entries set to 0
    matrix = [[0] * num_nodes for _ in range(num_nodes)]
    
    #Set the entries to 1 based on the adjacency list
    for source, targets in adj_list.items():
        for target in targets:
            matrix[source][target] = 1
            
    #Convert to a DataFrame and label the rows and columns
    node_labels = [f"Node_{i}" for i in range(num_nodes)]
    df = pd.DataFrame(matrix, index=node_labels, columns=node_labels)
    
    #Save to an Excel file
    df.to_excel(full_path, engine='openpyxl')
    print(f"The adjacency matrix (0/1) file was successfully saved to the desktop:\n{full_path}\n")

def visualize_and_verify_dag(adj_list):
    G = nx.DiGraph()
    for source, targets in adj_list.items():
        G.add_node(source)
        for target in targets:
            G.add_edge(source, target)
            
    is_dag = nx.is_directed_acyclic_graph(G)
    print("\n" + "="*40)
    if is_dag:
        print("✅ Scientific validation: The generated network is 100% DAG (acyclic).")
    else:
        print("❌ Error: The network contains a cycle!")
    print("="*40)
    
    plt.figure(figsize=(10, 8))
    plt.title("Generated Directed Acyclic Graph (DAG)", fontsize=14, fontweight='bold')
    
    pos = nx.spring_layout(G, seed=42) 
    
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color='skyblue', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif', font_weight='bold')
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color='gray', width=1.5)
    
    plt.axis('off')
    plt.tight_layout()
    print("Opening the graph display window...")
    plt.show()

#Get user input and generate the network
try:
    nodes = int(input("Enter the number of nodes: "))
    deg = float(input("Enter the average degree: "))

    user_dag = generate_dag(nodes, deg)
    
    #Save the adjacency list file (CSV)
    save_to_csv(user_dag)
    
    #Save the 0/1 adjacency matrix file (Excel)
    save_adjacency_matrix_to_excel(user_dag)
    
    #Plot and validate the user's network
    visualize_and_verify_dag(user_dag)
    
except ValueError as e:
    print("Please enter valid numbers.", e)

#Benchmark and calculate the average execution time (100 runs on a 50-node network)
print("\nCalculating the average execution time on a 50-node network...")
benchmark_nodes = 50
benchmark_deg = 4.0
iterations = 100

start_time = time.perf_counter()
for _ in range(iterations):
    _ = generate_dag(benchmark_nodes, benchmark_deg)
end_time = time.perf_counter()

total_time = end_time - start_time
avg_time = total_time / iterations

print(f"Benchmark results (100 runs on {benchmark_nodes} nodes):")
print(f"  Total time elapsed: {total_time:.6f} seconds")
print(f"  Average execution time: {avg_time:.6f} seconds")
