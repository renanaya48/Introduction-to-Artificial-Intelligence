import matplotlib.pyplot as plt
from ways import tools
from ways import load_map_from_csv
import MyPriorityQueue
import Node
import csv
from algo import algo_interface
import queue

flag = "NOT_FOUND"

def h_func(node):
    goal = node.target
    goal_lat = goal.lat
    goal_lon = goal.lon

    compute = tools.compute_distance(node.lat, node.lon, goal_lat, goal_lon)
    total_compue = (compute / 110)

    return total_compue

def ida_star1(source, target):
    graph = algo_interface.getGraph()
    with open('short_problems.csv', mode='r') as csv_file:
        readCSV = csv.reader(csv_file, delimiter=',')
        for row in readCSV:
            source = int(row[0])
            target = int(row[1])
            if ((source not in graph) | (target not in graph)):
                raise IndexError
            path = []
            node = Node.Node(source)
            closed_list = []
            lat = graph[source].lat
            lon = graph[source].lon
            node.lat = lat
            node.lon = lon
            goal = Node.Node(target)
            lat = graph[target].lat
            lon = graph[target].lon
            goal.lon = lon
            goal.lat = lat
            node.target = goal
            node.h = h_func(node)
            bound = h_func(node)
            path.append(node)
            g = node.g
            closed_list.append(source)

            while True:
                t = search(path, 0, bound, graph)
                if t== "FOUND":
                    txt_file = open("results/IDARuns.txt", "a")
                    txt_file.write(str(path[len(path)-1].g)+ "\n")
                    txt_file.close()
                    break
                    #list = []
                   # for node in path:
                    #    list.append(node.state)
                if t == float("inf"):
                    return "NOT_FOUND"
                bound = t
#@tools.timed
def ida_star(source, target):
    #path = queue.Queue()
    path = []
    graph = algo_interface.getGraph()
    if ((source not in graph) | (target not in graph)):
        raise IndexError
    node = Node.Node(source)
    closed_list = []
    lat = graph[source].lat
    lon = graph[source].lon
    node.lat = lat
    node.lon = lon
    goal = Node.Node(target)
    lat = graph[target].lat
    lon = graph[target].lon
    goal.lon = lon
    goal.lat = lat
    node.target = goal
    node.h = h_func(node)
    bound = h_func(node)
    path.append(node)
    g = node.g
    closed_list.append(source)

    while True:
        t = search(path, 0, bound, graph)
        if t== "FOUND":

            list = []
            for node in path:
                list.append(node.state)
            return list
        if t == float("inf"):
            return "NOT_FOUND"
        bound = t




def search(path, g, bound, graph):
    temp_node = path[len(path)-1]
    lat = graph[temp_node.state].lat
    lon = graph[temp_node.state].lon
    temp_node.lat = lat
    temp_node.lon = lon
    f = g + h_func(temp_node)
    if(f > bound):
        return f
    if (temp_node.state == temp_node.target.state):
        return "FOUND"
    min = float("inf")
    for link in graph[temp_node.state].links:
        current_node = Node.Node(link.target)
        current_node.target = temp_node.target
        new_g = algo_interface.price_function(link)
        g_to_send = g + new_g
        current_node.g = g_to_send
        lat1 = graph[current_node.state].lat
        lon1 = graph[current_node.state].lon
        current_node.lat = lat1
        current_node.lon = lon1
        path.append(current_node)
        t = search(path, g_to_send, bound, graph)
        if t== "FOUND":
            return "FOUND"
        if t < min:
            min = t
        path.pop()
    return min

