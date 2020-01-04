'''
This file should be runnable to print map_statistics using
$ python stats.py
'''

from collections import namedtuple
from ways import load_map_from_csv
import collections


def map_statistics(roads):
    '''return a dictionary containing the desired information
    You can edit this function as you wish'''
    Stat = namedtuple('Stat', ['max', 'min', 'avg'])
    Junctions = roads.junctions()
    numOfLinks = 0
    maxOut = 0
    minOut = len(Junctions[0].links)
    maxDis = 0
    minDis = Junctions[0].links[0].distance
    sumDis = 0
    histogram = collections.Counter(getattr(link, 'highway_type') for link in roads.iterlinks())
    for junc in Junctions:
        numCurrentLinks = (len(junc.links))
        numOfLinks=numOfLinks + numCurrentLinks
        if numCurrentLinks > maxOut:
            maxOut = numCurrentLinks
        if numCurrentLinks < minOut:
            minOut = numCurrentLinks
        for link in junc.links:
            currentDis = link.distance
            sumDis = sumDis + currentDis
            if currentDis > maxDis:
                maxDis = currentDis
            if currentDis < minDis:
                minDis = currentDis



    return {
        'Number of junctions': len(roads.junctions()),
        'Number of links': numOfLinks,
        'Outgoing branching factor': Stat(max=maxOut, min=minOut, avg=(numOfLinks/len(roads.junctions()))),
        'Link distance': Stat(max=maxDis, min=minDis, avg=sumDis / numOfLinks),
        # value should be a dictionary
        # mapping each road_info.TYPE to the no' of links of this type
        'Link type histogram': histogram,  # tip: use collections.Counter
    }


def print_stats():
    for k, v in map_statistics(load_map_from_csv()).items():
        print('{}: {}'.format(k, v))


if __name__ == '__main__':
    from sys import argv

    assert len(argv) == 1
    print_stats()
