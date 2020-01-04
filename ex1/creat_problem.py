from ways import graph
from random import randrange
import csv
import sys


def create_csv(roads):
    counter = 0
    with open('problems.csv', 'w', newline='') as csvfile:
        wtiter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(sys.maxsize):
            if counter == 100:
                break
            source = randrange(len(roads))
            target = randrange(len(roads))
            if there_is_way(roads, source, target):
                wtiter.writerow([str(source), str(target)])
                counter += 1


def adj(roads, v):
    neighb = []
    links = roads.get(v).links
    for link in links:
        neighb.append(link[1])
    return neighb


def BFS(roads, source_id):
    close = [False] * (len(roads))
    queue = []
    queue.append(source_id)
    close[source_id] = True
    while queue:
        u = queue.pop(0)
        neighb = adj(roads, u)
        for n in neighb:
            if not close[n]:
                queue.append(n)
                close[n] = True
    return close


def there_is_way(roads, source_id, target_id):
    visited = BFS(roads, source_id)
    if visited[target_id]:
        return True
    else:
        return False


r = graph.load_map_from_csv()
create_csv(r)