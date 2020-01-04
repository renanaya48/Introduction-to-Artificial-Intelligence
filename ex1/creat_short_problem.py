from ways import graph
from random import randrange
import csv
import sys


def create_csv(roads):
    counter = 0
    with open('short_problems.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(["32", "35"])
        filewriter.writerow(["1", "3"])
        filewriter.writerow(["74", "77"])


def adj(roads, v):
    succ = []
    links = roads.get(v).links
    for k in links:
        succ.append(k[1])
    return succ


def BFS(roads, source_id):
    visited = [False] * (len(roads))
    queue = []
    queue.append(source_id)
    visited[source_id] = True
    while queue:
        u = queue.pop(0)
        succ = adj(roads, u)
        for v in succ:
            if not visited[v]:
                queue.append(v)
                visited[v] = True
    return visited


def if_accessible(roads, source_id, target_id):
    visited = BFS(roads, source_id)
    if visited[target_id]:
        return True
    else:
        return False


r = graph.load_map_from_csv()
create_csv(r)