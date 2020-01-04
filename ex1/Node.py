import utility
#@total_ordering
class Node:
    def __init__(self, state, highway_type = 20, distance=0, parent=None, path_cost=0, h=0, g=0, total_h = 0, lat=0, lon=0, target = None):
        self.state = state
        self.parent = parent
        self.h = h
        self.path_cost = path_cost
        self.depth = 0
        self.highway_type = highway_type
        self.distance = distance
        self.g = g
        self.total_h = total_h
        self.lat = lat
        self.lon = lon
        self.target = target
        if parent:
            self.depth = parent.depth + 1

    def expand(self, problem):
        return utility.ordered_set([self.child_node(problem, action)
                            for action in problem.actions(self.state)])

    def child_node(self, problem, action):
        next_state = problem.succ(self.state, action)
        next_node = Node(next_state, self, action,
                         self.path_cost + problem.step_cost(self.state, action))
        return next_node

    def solution(self):
        return [node.action for node in self.path()[1:]]

    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __repr__(self):
        return f"<{self.state}>"

    def __lt__(self, node):
        return self.state < node.state

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.state)