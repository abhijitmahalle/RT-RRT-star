import math 
import random
import numpy as np 
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        
        
    def path_vertical(self):
        if self.parent:
            return [self.y, self.parent.y]
        else:
            return []

    def path_horizontal(self):
        if self.parent:
            return [self.x, self.parent.x]
        else:
            return []
        
class Base_Class:
    class Edge:

        def __init__(self, fnode, tnode):
            self.fromx = fnode.x
            self.fromy = fnode.y
            self.towards_x = tnode.x
            self.towards_y = tnode.y

    def __init__(self, start, goal, circular_obstacles, line_obstacles, world_map,
                 expand_radius=0.2, goal_sample_rate=5, max_iter=500, node_list = []):
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_randx = world_map[0]
        self.max_randx = world_map[1]
        self.min_randy = world_map[2]
        self.max_randy = world_map[3]
        self.expand_radius = expand_radius
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.circular_obstacles = circular_obstacles
        self.line_obstacles = line_obstacles
        self.node_list = node_list

    def rewire_random_node(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.cost_to_come(new_node, to_node)

        if extend_length > d:
            extend_length = d
        
        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)

        new_node.parent = from_node
        
        return new_node

    def backtracking(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        
        return path

    def get_random_node(self):
        if random.randint(0, 100) >= self.goal_sample_rate:
            rnd = self.Node(random.uniform(self.min_randx, self.max_randx),
                            random.uniform(self.min_randy, self.max_randy))
        else:
            rnd = self.Node(self.end.x, self.end.y)
        return rnd
    
    def cost_to_go(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    @staticmethod
    def generate_circle(x, y, size, color="-k"):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    def compute_angle(self, a, b, c):
        
        angle = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        if angle < 0:
            angle = angle + 180

        if angle < 0:
            angle = angle + 180

        if angle > 180:
            angle = angle - 180

        if angle > 180:
            angle = angle - 180
            
        return angle

    def neighbour_node_index(self, neighbour_node):
        dlist = [math.sqrt((node.x - neighbour_node.x) ** 2 + (node.y - neighbour_node.y)
                 ** 2) for node in self.node_list]
        minind = dlist.index(min(dlist))

        return minind    

    def get_edge_dist(self, edge, node):
        a = edge.fromy - edge.towards_y
        b = -(edge.fromx - edge.towards_x)
        c = edge.fromx*edge.towards_y + edge.towards_x*edge.fromy
        if self.compute_angle((edge.fromx,edge.fromy),(edge.towards_x,edge.towards_y),(node.x,node.y)) < 90 and self.compute_angle((edge.towards_x,edge.towards_y),(edge.fromx,edge.fromy),(node.x,node.y)) < 90:
            return abs(a*node.x+b*node.y+c)/math.hypot(a,b)
        return min(math.hypot(node.x-edge.fromx,node.y-edge.fromy), math.hypot(node.x-edge.towards_x,node.y-edge.towards_y))
    
    def get_nearest_point_on_edge(self, edge, node):
        a = edge.fromy - edge.towards_y
        b = -(edge.fromx - edge.towards_x)
        c = edge.fromx*edge.towards_y - edge.towards_x*edge.fromy
        if self.compute_angle((edge.fromx,edge.fromy),(edge.towards_x,edge.towards_y),(node.x,node.y)) < 90 and self.compute_angle((node.x,node.y),(edge.fromx,edge.fromy),(edge.towards_x,edge.towards_y)) < 90:
            temp = -1 * (a * node.x + b * node.y + c) / (a * a + b * b)
            x = temp * a + node.x
            y = temp * b + node.y
            return x, y
        elif math.hypot((edge.fromx - node.x),(edge.fromy - node.y)) > math.hypot((edge.towards_x - node.x),(edge.towards_y - node.y)):
            return edge.towards_x, edge.towards_y
        return edge.fromx, edge.fromy



    def is_in_obstacle(self, node, obstacleList, linearObstacleList):

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_horizontal()]
            dy_list = [oy - y for y in node.path_vertical()]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= size ** 2:
                return False
            
            firstx , firsty = node.path_horizontal()[0] , node.path_vertical()[0]
            secondx , secondy = node.path_horizontal()[1] ,  node.path_vertical()[1]
            
           
            edge = self.Edge(self.Node(firstx, firsty), self.Node(secondx, secondy))

            footx, footy = self.get_nearest_point_on_edge(edge, self.Node(ox, oy))
            dist = (footx - ox)*(footx - ox) + (footy - oy)*(footy - oy)

            if dist <= size ** 2:
                return False
                
        for (sx,sy,ex,ey) in linearObstacleList:
            firstx  , firsty= node.path_horizontal()[0] , node.path_vertical()[0]
            secondx , secondy = node.path_horizontal()[1] , node.path_vertical()[1]
             
            A = [firstx, firsty]
            B = [secondx, secondy]

            C = [sx, sy]
            D = [ex, ey]

            if self.line_intersection((A, B), (C, D)):
                return False

        return True

    @staticmethod
    def line_intersection(line1, line2):
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            
           return False
        
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div

        if (x > max(line1[0][0], line1[1][0]) or x < min(line1[0][0], line1[1][0]) or
            x > max(line2[0][0], line2[1][0]) or x < min(line2[0][0], line2[1][0]) or
            y > max(line2[0][1], line2[1][1]) or y <min(line2[0][1], line2[1][1]) or
            y > max(line1[0][1], line1[1][1]) or y <min(line1[0][1], line1[1][1])):
            
            return False
        
        return True

    @staticmethod
    def cost_to_come(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta
