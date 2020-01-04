import MyPriorityQueue
from algo import algo_interface
import Node

def g(node):
    return node.path_cost

p = MyPriorityQueue.PriorityQueue(g)
n1 = Node.Node(1)
n2 = Node.Node(2)
n3 = Node.Node(3)
n2.target = 5
n2.path_cost = 80
n2.distance = 42
n1.path_cost = 21
p.append(n1)
p.append(n2)
p.append(n3)
n4 = Node.Node(2)
n4.path_cost = 8
p.append(n4)
l=[]
n5 = Node.Node(5)
l.append(n5)
l = [3, 4, 5]
p.extend(l)
y = p.__getitem__(n4)
if(p.__contains__(n4)):
    print("ok")
p.__delitem__(n4)
if (type(n4)== Node.Node):
    print("node")
if (type(p)==Node.Node):
    print("bla")
print (type(n4))
print(y)