"""Utility functions."""

import json
from copy import deepcopy
from typing import List


class TimeInterval:
    def __init__(self, interval: List[int]):
        self.min, self.max = interval
        self.prev_num = self.min

    def to_list(self):
        return [self.min, self.max]

    def __str__(self):
        return str([self.min, self.max])

    def num_frames(self):
        return self.max - self.min + 1


def read_annotations(ann_path: str):
    with open(ann_path, "r") as f:
        ann_dict = json.load(f)
    ann_dict_conv = {}
    for key in ann_dict:
        ann_dict_conv[key] = [TimeInterval(x) for x in ann_dict[key]]
    return ann_dict_conv


def read_polygon(polygon_path: str):
    with open(polygon_path, "r") as f:
        pol_dict = json.load(f)
    return pol_dict


class Bbox:
    def __init__(self, points: List[float], frame_dim: List[int]):
        self.x_min = max(0, int(points[0]))
        self.y_min = max(0, int(points[1]))
        self.x_max = min(frame_dim[1], int(points[2]))
        self.y_max = min(frame_dim[0], int(points[3]))
        self.center_x = (self.x_max + self.x_min) // 2
        self.center_y = (self.y_max + self.y_min) // 2
        self.point_min = (self.x_min, self.y_min)
        self.point_max = (self.x_max, self.y_max)
        self.center = (self.center_x, self.center_y)
        self.list = [self.x_min, self.y_min, self.x_max, self.y_max]


def bounding_rectangle(points):
    x_coordinates, y_coordinates = zip(*points)
    return [
        min(x_coordinates),
        min(y_coordinates),
        max(x_coordinates),
        max(y_coordinates),
    ]


def transform_polygon(polygon: List[List[int]], b_rect: Bbox):
    polygon_ = deepcopy(polygon)
    for i_point in range(len(polygon_)):
        polygon_[i_point][0] -= b_rect.x_min
        polygon_[i_point][1] -= b_rect.y_min
    return polygon_
