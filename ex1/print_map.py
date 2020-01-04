from algo import aStar_algorithm, algo_interface
from ways import draw
import matplotlib.pyplot as plt
import csv

def print_map(path, count, name):
    graph = algo_interface.getGraph()
    draw.plot_path(graph, path, color='g')
    #plt.show()
    plt.savefig("sulotions_img/" + name + "_graph" + str(count) + ".png")
    plt.clf()


count = 1

with open('problems.csv', mode='r') as csv_file:
    readCSV = csv.reader(csv_file, delimiter=',')
    for row in readCSV:
        if count > 10:
            break
        source = int(row[0])
        target = int(row[1])
        path = aStar_algorithm.astar_search(source, target)
        print_map(path, count, "astar")
        count = count + 1
    csv_file.close()
