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
            print(item_in_shapes['points'])
            return item_in_shapes['points']


# Turns out that PathTools doesn't have to be a dedicated class:)
class PathToolBox:
    def __init__(self, target_regions, gluewidth, img):
        self.target_regions = target_regions
        self.gluewidth = gluewidth
        self.gray = img


    def path_plot(self, path):
        plt.imshow(self.gray, cmap=plt.get_cmap('gray'))
        corners = [self.target_regions[path[0].rect][path[0].i]]
        for rect in path:
            vertices = np.concatenate([self.target_regions[rect.rect],
                                       [self.target_regions[rect.rect][0]]], axis=0)
            plt.plot(vertices[:, 0], vertices[:, 1], color='red')
            # print(vertices)
            corners = np.concatenate([corners, [self.target_regions[rect.rect][rect.i]]], axis=0)
            corners = np.concatenate([corners, [self.target_regions[rect.rect][rect.o]]], axis=0)
            #corners.extend(self.target_regions[rect.rect][rect.i], self.target_regions[rect.rect][rect.o])

        # print('corners: ', path_corners)
        plt.plot(corners[:, 0], corners[:, 1], color='blue')
        plt.plot(corners[:, 0][0], corners[:, 1][0],
                 corners[:, 0][1]-corners[:, 0][0],
                 corners[:, 1][1] - corners[:, 1][0],)

        plt.show()

    def get_shortest_side(self, rect):
        side1 = self.dist_euler(rect[0], rect[1])
        side2 = self.dist_euler(rect[0], rect[3])
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
        look_up = [[(1, 2), (0, 3), (3, 0), (2, 1)], [(3, 1), (2, 3), (1, 0), (0, 1)]]
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


if __name__ == "__main__":
    mb_info = MotherBoardInput('mother_board.png', 'rectangles.json').info_extraction()
    rect_list = mb_info[0]
    glue_width = mb_info[1]
    path_tool = PathToolBox(rect_list, glue_width)
    path_tool.target_regions = path_tool.target_regions[0:10]
    print(len(path_tool.target_regions))


