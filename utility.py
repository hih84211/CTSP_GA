import math
from functools import cmp_to_key
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import json
import sys
import optparse


class MotherBoardInput:
    def __init__(self, photo_name, json_name):
        img = mpimg.imread(photo_name)
        self.json_name = json_name
        self.gray = img[:, :, 2]

    # Return a tuple(regions, gluewidth),
    #   where regions is a list of rectangles,
    #   where rectangle is represented as a list of it's four vertices,
    #   where each vertex is a 2D coordinate.
    def info_extraction(self):
        with open(self.json_name, 'r') as file:
            data = json.load(file)
        shapes = data['shapes']
        regions = []
        gluewidth = -1.
        for item in shapes:
            if item['label'] == 'gluewidth':
                gluewidth = item['points'][1][1] - item['points'][0][1]
            else:
                #for a in item['points']:
                    #a = sorted(a)
                    #print(tuple(map(int, a)))
                #regions.append([tuple(map(int, a)) for a in item['points']])
                regions.append(sort_verices([tuple(map(int, a)) for a in item['points']]))
        return regions, gluewidth, self.gray

    # Return the rectangle region based on the index of mother board image
    def target_area(self, x, y, x2, y2):
        ta = self.gray[int(y): int(y2), int(x): int(x2)]
        return ta

    # process the regions info and label info
    # for path constructing
    def shapes_extrat(self, item_in_shapes):
        if item_in_shapes['label'] == "gluewidth":
            gluewidth = item_in_shapes['points'][1][1] - item_in_shapes['points'][0][1]
            start_point = (-1, -1)
            return (gluewidth, gluewidth)
        else:
            # print(item_in_shapes['points'])
            return item_in_shapes['points']


# Turns out that PathTools doesn't have to be a dedicated class:)
class PathToolBox:
    def __init__(self, target_regions, gluewidth, img):
        self.target_regions = target_regions
        self.gluewidth = gluewidth
        self.gray = img

    def path_plot(self, path=None):
        plt.imshow(self.gray, cmap=plt.get_cmap('gray'))
        if path is None:
            for rect in self.target_regions:
                # vertices = np.concatenate([rect, [rect[0]]], axis=0)
                vertices = np.concatenate([rect])
                print(vertices)
                plt.plot(vertices[:, 0], vertices[:, 1], color='red')
                if self.get_shortest_side(rect)[0]:
                    plt.plot(vertices[1:3, 0], vertices[1:3, 1], color='blue')
                if not self.get_shortest_side(rect)[0]:
                    plt.plot(vertices[0:2, 0], vertices[0:2, 1], color='green')
        if path.any():
            corners = [self.target_regions[path[0].rect][path[0].i]]
            for rect in path:
                vertices = np.concatenate([self.target_regions[rect.rect],
                                           [self.target_regions[rect.rect][0]]], axis=0)
                plt.plot(vertices[:, 0], vertices[:, 1], color='red')

                # print(vertices)
                corners = np.concatenate([corners, [self.target_regions[rect.rect][rect.i]]], axis=0)
                corners = np.concatenate([corners, [self.target_regions[rect.rect][rect.o]]], axis=0)
                # corners.extend(self.target_regions[rect.rect][rect.i], self.target_regions[rect.rect][rect.o])
                plt.plot(corners[:, 0], corners[:, 1], color='blue')
                plt.plot(corners[:, 0][0], corners[:, 1][0],
                         corners[:, 0][1] - corners[:, 0][0],
                         corners[:, 1][1] - corners[:, 1][0], )
        plt.show()

    def angle(self, v1):
        dx1 = v1[0]
        dy1 = v1[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180 / math.pi)
        if angle1 < 0:
            angle1 = 360 + angle1
        return angle1

    def barrier_detect(self, vector_barrier, euler_barrier, vector, euler):
        # size of all inputs : num_rec_corners x B x [value]
        self.barrier_path = []
        rewards = [0] * len(vector_barrier[0])
        vector = vector.tolist()
        euler = euler.tolist()
        for batch_num in range(len(vector_barrier[0])):
            single_map_barrier = vector_barrier[:, batch_num]
            single_map_barrier = single_map_barrier.tolist()
            single_map_euler = euler_barrier[:, batch_num]
            single_map_euler = single_map_euler.tolist()
            for i in range(0, len(vector_barrier), 4):
                temp_vector = single_map_barrier[i:i + 4]
                temp_euler = single_map_euler[i:i + 4]  # load four points of an object
                temp = []
                idxs = [j for j in range(i, i + 4)]
                for vec, eul, k in zip(temp_vector, temp_euler, idxs):
                    temp.append([self.angle(vec), vec, eul, k])  # k for latter operation of index
                temp = sorted(temp, key=lambda temp: temp[0])  # small -> big
                temp_euler.sort()
                next_point_vec = self.angle(vector[batch_num])
                if next_point_vec >= temp[0][0] and next_point_vec <= temp[-1][0]:
                    if euler[batch_num] > temp_euler[0]:
                        # collision detected
                        if (abs(next_point_vec - temp[0][0])) < abs((next_point_vec - temp[-1][0])):
                            if batch_num == 0:
                                if temp[0][2] < temp[1][2]:
                                    self.barrier_path.append(temp[0][3])
                                    self.barrier_path.append(temp[1][3])
                                else:
                                    self.barrier_path.append(temp[1][3])
                                    self.barrier_path.append(temp[0][3])

                            vec_dis = [temp[0][1][0] - temp[1][1][0], temp[0][1][1] - temp[1][1][1]]
                            vec_dis = self.vec_euler(vec_dis)
                            vec_dis += self.vec_euler(
                                [vector[batch_num][0] - temp[1][1][0], vector[batch_num][1] - temp[1][1][1]])
                            vec_dis += self.vec_euler(temp[0][1])
                        else:
                            if batch_num == 0:
                                if temp[-1][2] < temp[-2][2]:
                                    self.barrier_path.append(temp[-1][3])
                                    self.barrier_path.append(temp[-2][3])
                                else:
                                    self.barrier_path.append(temp[-2][3])
                                    self.barrier_path.append(temp[-1][3])
                            vec_dis = [temp[-1][1][0] - temp[-2][1][0], temp[-1][1][1] - temp[-2][1][1]]
                            vec_dis = self.vec_euler(vec_dis)
                            vec_dis += self.vec_euler(
                                [vector[batch_num][0] - temp[-2][1][0], vector[batch_num][1] - temp[-2][1][1]])
                            vec_dis += self.vec_euler(temp[-1][1])
                        rewards[batch_num] += vec_dis - euler[batch_num]
                    else:
                        pass
                else:
                    pass
                rewards = torch.tensor(rewards)
                return rewards


    def get_shortest_side(self, rect):
        side1 = self.dist_euler(rect[0], rect[1])
        side2 = self.dist_euler(rect[1], rect[2])
        if side1 < side2:
            return 0, side1
        else:
            return 1, side2

    def get_outcorner(self, rect_index, incorner):
        # precomputed with high precision super algorithm:
        #
        # 0 3
        # 1 2
        #
        # 0 even:1 odd:2 | even:3 odd:2
        # 1 even:0 odd:3 | even:2 odd:3
        # 2 even:3 odd:0 | even:1 odd:0
        # 3 even:2 odd:1 | even:0 odd:1
        look_up = [[(1, 2), (0, 3), (3, 0), (2, 1)], [(3, 2), (2, 3), (1, 0), (0, 1)]]
        rect = self.target_regions[rect_index]
        short_side = self.get_shortest_side(rect)
        turns = int(short_side[1] / self.gluewidth)
        return look_up[short_side[0]][incorner][turns % 2]

    def dist_euler(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def dist_max(self, a, b):
        edge1 = abs(a[0] - b[0])
        edge2 = abs(a[1] - b[1])
        if edge1 > edge2:
            return edge1
        else:
            return edge2

def sort_verices(v):
    r = sorted(v)
    r2 = r[2:4]
    r2.reverse()
    return r[0:2] + r2


if __name__ == "__main__":
    mb_info = MotherBoardInput('mother_board.png', 'rectangles.json').info_extraction()
    rect_list = mb_info[0]
    glue_width = mb_info[1]
    path_tool = PathToolBox(rect_list, glue_width, mb_info[2])
    # path_tool.target_regions = path_tool.target_regions[2:4]
    path_tool.path_plot()
    #print(path_tool.target_regions)
    #print(sorted(path_tool.target_regions))


