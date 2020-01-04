from ways import load_map_from_csv
import MyPriorityQueue
import Node
import csv

def getGraph():
    graph = load_map_from_csv()
    map_dic = dict()
    for junc in graph.junctions():
        map_dic[junc.index] = junc
    return map_dic

def find_speed(speed):
    if ((speed == 0) | (speed == 2)):
        return 110
    elif (speed == 1):
        return 100
    elif ((speed == 3) | (speed == 4) | (speed == 12)):
        return 90
    elif ((speed == 5) | (speed == 6) | (speed == 8)):
        return 80
    elif (speed == 7):
        return 70
    elif (speed == 9):
        return 60
    elif (speed == 10):
        return 50
    elif (speed == 11):
        return 30
    else:
        return 0


def cost_function(link):
    price = cost_function(link)
    if (link.parent != None):
        parent_cost = link.parent.path_cost
        total_return = price + parent_cost
        return total_return
    return price


def price_function(link):
    type_hi = link.highway_type
    high_speed = find_speed(type_hi)
    if(high_speed == 0):
        return 0
    dist = link.distance
    return_price = ((dist / high_speed) / 1000)
    return return_price



def path(source, node):

    path = []
    rev_path=[]
    while(node.state != source):
        path.append(node.state)
        node = node.parent
    path.append(source)
    for i in reversed(path):
        rev_path.append(i)
    return (rev_path)