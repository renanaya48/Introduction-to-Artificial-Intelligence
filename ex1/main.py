from algo import ucs_algorithm, aStar_algorithm, ida_star, ucs
from ways import *
import MyPriorityQueue
import Node
'''
Parse input and run appropriate code.
Don't use this file for the actual work; only minimal code should be here.
We just parse input and call methods from other modules.
'''


# do NOT import ways. This should be done from other files
# simply import your modules and call the appropriate functions


def find_ucs_rout(source, target):

    return (ucs.ucs(source, target))
    # return(ucs_algorithm.ucs_search(source, target))
    #return (ucs_algorithm.ucs_search_for_file(source, target))


def find_astar_route(source, target):
    return(aStar_algorithm.astar_search(source, target))



def find_idastar_route(source, target):
    return(ida_star.ida_star(source, target))


def dispatch(argv):
    from sys import argv
    source, target = int(argv[2]), int(argv[3])
    if argv[1] == 'ucs':
        path = find_ucs_rout(source, target)
    elif argv[1] == 'astar':
        path = find_astar_route(source, target)
    elif argv[1] == 'idastar':
        path = find_idastar_route(source, target)
    print(' '.join(str(j) for j in path))


if __name__ == '__main__':
    from sys import argv

    #p = MyPriorityQueue.PriorityQueue()
    #n= Node.Node("a")
    #print(n)

    dispatch(argv)
