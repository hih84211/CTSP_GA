import math

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
        plt.imshow(self.gray, cmap=plt.get_cmap('gray'))

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
                regions.append([tuple(map(int, a)) for a in item['points']])
        return regions, gluewidth

    # Return the rectangle region based on the index of mother board image
    def target_area(self, x, y, x2, y2):
        ta = self.gray[int(y): int(y2), int(x): int(x2)]
        return ta

    # process the regions info and label info
    # for path constructing
    def __shapes_extrat(self, item_in_shapes):
        if item_in_shapes['label'] == "gluewidth":
            gluewidth = item_in_shapes['points'][1][1] - item_in_shapes['points'][0][1]
            start_point = (-1, -1)
            return (gluewidth, gluewidth)
        else:
            print(item_in_shapes['points'])
            return item_in_shapes['points']



# Turns out that PathTools doesn't have to be a dedicated class:)
class PathToolBox:
    def __init__(self, target_areas, gluewidth):
        self.target_areas = target_areas
        self.gluewidth = gluewidth

    def path_plot(self, path):
        path_corners = [[], []]
        for rect in path:
            vertices = np.concatenate((rect.vertices, [rect.vertices[0]]), axis=0)
            plt.plot(vertices[:, 0], vertices[:, 1], color='red')
            path_corners[0].append(rect.i)
            path_corners[1].append(rect.o)
        plt.plot(path_corners[0], path_corners[1])
        plt.show()

    def __get_shortest_side(self, rect):
        side1 = self.dist_euler(rect[0], rect[1])
        side2 = self.dist_euler(rect[0], rect[2])
        if side1 < side2:
            return side1
        else:
            return side2

    def get_outcorner(self, rect_index, incorner):
        # precomputed with high precision super algorithm:
        #
        # 0 1
        # 2 3
        #
        # 3 even:2 odd:0
        # 2 even:3 odd:1
        # 1 even:0 odd:2
        # 0 even:1 odd:3
        look_up = [(1, 3), (0, 2), (3, 1), (2, 0)]
        rect = self.target_areas[rect_index]
        short_side = self.__get_shortest_side(rect)
        turns = int(short_side / self.gluewidth)
        return look_up[incorner][turns % 2]

    def dist_euler(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def dist_max(self, a, b):
        edge1 = abs(a[0] - b[0])
        edge2 = abs(a[1] - b[1])
        if edge1 > edge2:
            return edge1
        else:
            return edge2

    '''
    def zig_zag_path(self, path_corners_index):
        # path corners index = [[start_corner_index, end_corner_index], ....] = array 2d (path_lengh,2)
        path_gazebo = []
        path_corners = np.array([])
        self.X_all = self.every_point()
        for index in path_corners_index:
            np.append(path_corners, [self.X_all[index[0]], self.X_all[index[1]]])

        plt.plot(path_corners[:, 0], path_corners[:, 1])
        data_1 = np.array(self.X_all)
        data_1 = np.reshape(data_1, (34, 4, 2))
        for rec in data_1:
            rec = np.concatenate((rec, [rec[0]]), axis=0)
            plt.plot(rec[:, 0], rec[:, 1], color='red')
        plt.show()
        for index in path_corners_index:  # find the longer side => zig-zag to end point
            corner_num = index[0] % 4
            if (abs(self.X_all[index[0]][0] - self.X_all[index[1]][0])) > (
            abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])):  # if longer side = horizon side = row side
                if corner_num == 0:
                    y_way = range(0, int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])),
                                  int(self.target_metrices[1]))
                    x_way_left = range(0, len(self.target_metrices[0][int(index[0] / 4)][0][0]),
                                       int(self.target_metrices[1]))
                    x_way_right = range(len(self.target_metrices[0][int(index[0] / 4)][0][0]), 0,
                                        -int(self.target_metrices[1]))
                elif corner_num == 3:  # start = left down ,out = left up
                    y_way = range(0, -int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])),
                                  -int(self.target_metrices[1]))
                    x_way_left = range(0, len(self.target_metrices[0][int(index[0] / 4)][0][0]),
                                       int(self.target_metrices[1]))
                    x_way_right = range(len(self.target_metrices[0][int(index[0] / 4)][0][0]), 0,
                                        -int(self.target_metrices[1]))
                elif corner_num == 1:  # start = right up, out = right down
                    y_way = range(0, int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])),
                                  int(self.target_metrices[1]))
                    x_way_left = range(0, -len(self.target_metrices[0][int(index[0] / 4)][0][0]),
                                       -int(self.target_metrices[1]))
                    x_way_right = range(-len(self.target_metrices[0][int(index[0] / 4)][0][0]), 0,
                                        int(self.target_metrices[1]))
                else:  # corner_num == 2: #start = right down, out = right up
                    y_way = range(0, -int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])),
                                  -int(self.target_metrices[1]))
                    x_way_left = range(0, -len(self.target_metrices[0][int(index[0] / 4)][0][0]),
                                       -int(self.target_metrices[1]))
                    x_way_right = range(-len(self.target_metrices[0][int(index[0] / 4)][0][0]), 0,
                                        int(self.target_metrices[1]))
                # x_way_left = when the agent is on the left side , then it should move to the right side
                way_2 = x_way_left
                way_1 = y_way
                turn = 0
                for i in way_1:
                    turn += 1
                    for j in way_2:
                        path_gazebo.append([self.X_all[index[0]][0] + j, self.X_all[index[0]][1] + i])
                    if (turn % 2) == 1:
                        way_2 = x_way_right
                    else:
                        way_2 = x_way_left
            else:  # long side = straight side = column
                if corner_num == 0:  # lu -> ?
                    y_way = range(0, len(self.target_metrices[0][int(index[0] / 4)][0]), int(self.target_metrices[1]))
                    x_way_left = range(0, int(abs(self.X_all[index[0]][0] - self.X_all[index[1]][0])),
                                       int(self.target_metrices[1]))
                    x_way_right = range(int(abs(self.X_all[index[0]][0] - self.X_all[index[1]][0])), 0,
                                        -int(self.target_metrices[1]))
                elif corner_num == 3:  # start = left down ,out = ?
                    y_way = range(len(self.target_metrices[0][int(index[0] / 4)][0]), 0, -int(self.target_metrices[1]))
                    x_way_left = range(0, -int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])),
                                       -int(self.target_metrices[1]))
                    x_way_right = range(-int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])), 0,
                                        int(self.target_metrices[1]))
                elif corner_num == 1:  # start = right up, out = right down
                    y_way = range(0, -len(self.target_metrices[0][int(index[0] / 4)][0]), -int(self.target_metrices[1]))
                    x_way_left = range(0, int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])),
                                       int(self.target_metrices[1]))
                    x_way_right = range(int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])), 0,
                                        -int(self.target_metrices[1]))
                else:  # corner_num == 2: # start = right down, out = right up
                    y_way = range(0, -len(self.target_metrices[0][int(index[0] / 4)][0]), -int(self.target_metrices[1]))
                    x_way_left = range(0, -int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])),
                                       -int(self.target_metrices[1]))
                    x_way_right = range(-int(abs(self.X_all[index[0]][1] - self.X_all[index[1]][1])), 0,
                                        int(self.target_metrices[1]))
                # x_way_left = when the agent is on the left side , then it should move to the right side
                way_2 = x_way_left
                way_1 = y_way
                turn = 0
                for i in way_1:
                    turn += 1
                    for j in way_2:
                        path_gazebo.append([self.X_all[index[0]][0] + j, self.X_all[index[0]][1] + i])
                    if (turn % 2) == 1:
                        way_2 = x_way_right
                    else:
                        way_2 = x_way_left
        return path_gazebo
    '''


if __name__ == "__main__":
    test = MotherBoardInput('mother_board.png', 'rectangles.json')
    test.info_extraction()


