from ways import tools
import MyPriorityQueue
import Node
import csv
from algo import algo_interface, ucs

import utility


def g (node):
    price = algo_interface.cost_function(node)
    if (node.parent != None):
        parent_cost = node.parent.path_cost
        total_return = price + parent_cost
        return total_return
    return price

def ucs_search_for_file(source, target):

    graph = algo_interface.getGraph()
    with open('problems.csv', mode='r') as csv_file:
        readCSV = csv.reader(csv_file, delimiter=',')
        for row in readCSV:
            source = int(row[0])
            target = int(row[1])
            if ((source not in graph) | (target not in graph)):
                raise IndexError
            path = ucs.ucs(source, target)
            print (path)



#@tools.timed
def ucs_search(source, target):
    graph = algo_interface.getGraph()
    if ((source not in graph) | (target not in graph)):
        raise IndexError
    node = Node.Node(source)
    frontier = MyPriorityQueue.PriorityQueue(algo_interface.price_function)  # Priority Queue
    closed_list = set()
    node.path_cost = 0
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
                ###################################
                price_for_now = algo_interface.price_function(link)
                #price_for_now = algo_interface.cost_function(link)
                new_cost = next.path_cost + price_for_now
                new_node = Node.Node(link.target, highway_type, dista, next, new_cost)
                if(frontier.__contains__(new_node)):
                    old_cost = frontier.__getitem__(new_node)
                    if old_cost > new_cost:
                        frontier.__delitem__(new_node)
                        frontier.append(new_node)
                else:
                    frontier.append(new_node)



def ucs_search(source, target):

    graph = algo_interface.getGraph()
    with open('problems.csv', mode='r') as csv_file:
        readCSV = csv.reader(csv_file, delimiter=',')
        for row in readCSV:
            source = int(row[0])
            target = int(row[1])
            if ((source not in graph) | (target not in graph)):
                raise IndexError
            node = Node.Node(source)
            frontier = MyPriorityQueue.PriorityQueue(algo_interface.price_function)  # Priority Queue
            closed_list = set()
            frontier.append(node)


            while True:
                next = frontier.pop()
                closed_list.add(next.state)
                if(target == next.state):
                    txt_file = open("results/UCSRuns.txt", "a")
                    txt_file.write(str(next.path_cost)+ "\n")
                    txt_file.close()
                    break

                currentJuncIndex = next.state

                for link in graph[currentJuncIndex].links:
                    if(link.target not in closed_list):
                        highway_type = link.highway_type
                        speed = algo_interface.find_speed(highway_type)
                        dista = link.distance
                        price_for_now = algo_interface.price_function(link)
                        new_cost = next.path_cost + price_for_now
                        new_node = Node.Node(link.target, highway_type, dista, next, new_cost)
                        frontier.append(new_node)

