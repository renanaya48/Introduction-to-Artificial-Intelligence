import matplotlib.pyplot as plt
from ways import tools
from ways import load_map_from_csv
import MyPriorityQueue
import Node
import csv
from algo import algo_interface


def h_func(node):
    goal = node.target
    goal_lat = goal.lat
    goal_lon = goal.lon

    compute = tools.compute_distance(node.lat, node.lon, goal_lat, goal_lon)
    total_compue = (compute / 110)

    return total_compue

def astar_cost_func(node):
    g = algo_interface.price_function(node)
    h = h_func(node)
    total = g+h
    return (total)


def astar_search(source, target):
    x = []
    y = []
    graph = algo_interface.getGraph()
    with open('problems.csv', mode='r') as csv_file:
        readCSV = csv.reader(csv_file, delimiter=',')
        for row in readCSV:
            source = int(row[0])
            target = int(row[1])
            if ((source not in graph) | (target not in graph) ):
                raise IndexError
            node = Node.Node(source)
            frontier = MyPriorityQueue.PriorityQueue(astar_cost_func)  # Priority Queue
            closed_list = set()
            lat = graph[source].lat
            lon = graph[source].lon
            node.lat = lat
            node.lon = lon
            goal = Node.Node(target)
            lat = graph[target].lat
            lon = graph[target].lon
            goal.lon = lon
            goal.lat =lat
            node.target = goal
            node.h = h_func(node)
            frontier.append(node)

            while True:

                next = frontier.pop()
                closed_list.add(next.state)
                if (target == next.state):
                    txt_file = open("results/AStarRuns.txt", "a")
                    txt_file.write(str(node.h) + " " + str(next.g) + "\n")
                    txt_file.close()
                    algo_interface.path(source, next)

                    x.append(node.h)
                    y.append(next.g)
                    break

                currentJuncIndex = next.state

                for link in graph[currentJuncIndex].links:
                    if (link.target not in closed_list):
                        highway_type = link.highway_type
                        speed = algo_interface.find_speed(highway_type)
                        dista = link.distance
                        price_for_now = algo_interface.price_function(link)
                        y_cost = next.g + price_for_now
                        parent = graph[currentJuncIndex]
                        child = graph[link.target]
                        tempNode = Node.Node(link.target)
                        tempNode.lat = child.lat
                        tempNode.lon = child.lon
                        tempNode.parent = next
                        tempNode.target = goal
                        tempNode.highway_type = highway_type
                        tempNode.distance = dista
                        x_cost = next.total_h + h_func(tempNode)
                        h_cost = astar_cost_func(tempNode)
                        only_h = h_func(tempNode)
                        h_new_cost = h_cost + next.path_cost
                        lat = child.lat
                        lon = child.lon
                        new_node = Node.Node(link.target, highway_type, dista, next, h_new_cost, only_h, y_cost, x_cost, lat, lon, goal)
                        frontier.append(new_node)
    plt.clf()
    # plotting the points
    plt.plot(x, y, 'o', color = 'blue')

    # naming the x axis
    plt.xlabel('x - h')
    # naming the y axis
    plt.ylabel('y - g')

    # giving a title to my graph
    plt.title('astar graph')

    # function to show the plot
    plt.show()

#@tools.timed
def astar_search1(source, target):
    x = []
    y = []
    graph = algo_interface.getGraph()
    if ((source not in graph) | (target not in graph) ):
        raise IndexError
    node = Node.Node(source)
    frontier = MyPriorityQueue.PriorityQueue(astar_cost_func)  # Priority Queue
    closed_list = set()
    lat = graph[source].lat
    lon = graph[source].lon
    node.lat = lat
    node.lon = lon
    goal = Node.Node(target)
    lat = graph[target].lat
    lon = graph[target].lon
    goal.lon = lon
    goal.lat =lat
    node.target = goal
    node.h = h_func(node)
    frontier.append(node)

    while True:

        next = frontier.pop()
        closed_list.add(next.state)
        if (target == next.state):
            path = (algo_interface.path(source, next))
            return path

        currentJuncIndex = next.state

        for link in graph[currentJuncIndex].links:
            if (link.target not in closed_list):
                highway_type = link.highway_type
                speed = algo_interface.find_speed(highway_type)
                dista = link.distance
                price_for_now = algo_interface.price_function(link)
                y_cost = next.g + price_for_now
                parent = graph[currentJuncIndex]
                child = graph[link.target]
                tempNode = Node.Node(link.target)
                tempNode.lat = child.lat
                tempNode.lon = child.lon
                tempNode.parent = next
                tempNode.target = goal
                tempNode.highway_type = highway_type
                tempNode.distance = dista
                x_cost = next.total_h + h_func(tempNode)
                h_cost = astar_cost_func(tempNode)
                h_new_cost = h_cost + next.path_cost
                lat = child.lat
                lon = child.lon
                new_node = Node.Node(link.target, highway_type, dista, next, h_new_cost, h_cost, y_cost, x_cost, lat, lon, goal)
                if (frontier.__contains__(new_node)):
                    old_cost = frontier.__getitem__(new_node)
                    if old_cost > y_cost:
                        frontier.__delitem__(new_node)
                        frontier.append(new_node)
                else:
                    frontier.append(new_node)

