import MyPriorityQueue
import Node
import utility
from algo import algo_interface

def get_neighbors(node, graph):
    neighbors = []
    for n in graph[node.state].links:
        neighbor = Node.Node(n.target)
        neighbor.distance = n.distance
        neighbor.highway_type = n.highway_type
        neighbor.parent = node
        neighbors.append(neighbor)
    return neighbors


def cost_function(node):
    hi_type = node.highway_type
    hi_speed = algo_interface.find_speed(hi_type)
    if(hi_speed == 0):
        return 0
    dist = node.distance
    time = ((dist / hi_speed)/1000)
    return time

def g(node):
    return node.path_cost

def ucs(source, target):
    graph = algo_interface.getGraph()
    node = Node.Node(source)
    node.path_cost = 0
    open = MyPriorityQueue.PriorityQueue(cost_function)
    open.append(node)
    close = []


    while True:
        neighbors=[]
        next = open.pop()
        close.append(next)
        if(next.state == target):
            path = algo_interface.path(source, next)
            return path
        neighbors = get_neighbors(next, graph)
        for n in neighbors:
            if (n.state not in close):
                new_cost = g(next) + cost_function(n)
                n.path_cost = new_cost
                open.append(n)


