"""
Basic usage:
```

v_wall = VerticalWall(width, x_pos, bottom_y, top_y)
ball = Ball(radius=width)

init_xy = ...
new_xy = init_xy + velocity

if v_wall.collides_with(init_xy, new_xy):
    new_xy = v_wall.handle_collision(init_xy, new_xy)
```
"""
import operator

import numpy as np
import abc


class Wall(object, metaclass=abc.ABCMeta):
    def __init__(self, min_x, max_x, min_y, max_y, min_dist, thickness):
        self.top_segment = Segment(
            min_x,
            max_y,
            max_x,
            max_y,
        )
        self.bottom_segment = Segment(
            min_x,
            min_y,
            max_x,
            min_y,
        )
        self.left_segment = Segment(
            min_x,
            min_y,
            min_x,
            max_y,
        )
        self.right_segment = Segment(
            max_x,
            min_y,
            max_x,
            max_y,
        )
        self.segments = [
            self.top_segment,
            self.bottom_segment,
            self.right_segment,
            self.left_segment,
        ]
        self.min_dist = min_dist
        self.thickness = thickness
        self.max_x = max_x
        self.min_x = min_x
        self.max_y = max_y
        self.min_y = min_y

    def contains_point(self, points, inclusive=False):
        points = np.array(points)

        op = (operator.le if inclusive else operator.lt)

        if points.ndim == 1:
            points = points[None]

        result = np.logical_and(
            np.logical_and(
                op(self.min_x, points[:, 0]),
                op(points[:, 0], self.max_x)),
            np.logical_and(
                op(self.min_y, points[:, 1]),
                op(points[:, 1], self.max_y)))

        return result

    def contains_segment(self, segment):
        start, end = segment

        top = self.top_segment
        bottom = self.bottom_segment
        left = self.left_segment
        right = self.right_segment

        assert top.y0 == top.y1
        assert bottom.y0 == bottom.y1
        assert left.x0 == left.x1
        assert right.x0 == right.x1

        is_diagonal = end[0] != start[0] and end[1] != start[1]

        if (top.intersects_with(segment)
            or (top.y0 == max(end[1], start[1])
                and right.x0 == max(end[0], start[0])
                and is_diagonal)):
            return True
        if (right.intersects_with(segment)
            or (right.x0 == max(end[0], start[0])
                and bottom.y0 == min(end[1], start[1])
                and is_diagonal)):
            return True
        if (bottom.intersects_with(segment)
            or (bottom.y0 == min(end[1], start[1])
                and left.x0 == min(end[0], start[0])
                and is_diagonal)):
            return True
        if (left.intersects_with(segment)
            or (left.x0 == min(end[0], start[0])
                and top.y0 == max(end[1], start[1])
                and is_diagonal)):
            return True

        return False

    def handle_collision(self, start_point, end_point):
        trajectory_segment = (
            (start_point[0], start_point[1]),
            (end_point[0], end_point[1]),
        )
        if (self.top_segment.intersects_with(trajectory_segment) and
                end_point[1] <= start_point[1] >= self.max_y):
            end_point[1] = self.max_y
        if (self.bottom_segment.intersects_with(trajectory_segment) and
                end_point[1] >= start_point[1] <= self.min_y):
            end_point[1] = self.min_y
        if (self.right_segment.intersects_with(trajectory_segment) and
                end_point[0] <= start_point[0] >= self.max_x):
            end_point[0] = self.max_x
        if (self.left_segment.intersects_with(trajectory_segment) and
                end_point[0] >= start_point[0] <= self.min_x):
            end_point[0] = self.min_x
        return end_point


def intersects_with(s1, s2):
    A, B = s1
    C, D = s2

    cases = [
        [
            [A[0]-C[0], B[0]-C[0]],
            [A[1]-C[1], B[1]-C[1]],
        ],

        [
            [A[0]-D[0], B[0]-D[0]],
            [A[1]-D[1], B[1]-D[1]],
        ],
        [
            [C[0]-A[0], D[0]-A[0]],
            [C[1]-A[1], D[1]-A[1]],
        ],
        [
            [C[0]-B[0], D[0]-B[0]],
            [C[1]-B[1], D[1]-B[1]],
        ]
    ]

    determinants = [np.linalg.det(case) for case in cases]
    signs = np.sign(determinants)

    return (
        signs[0] * signs[1] != 0 and signs[0] == -signs[1]
        and signs[2] * signs[3] != 0 and signs[2] == -signs[3])


class Segment(object):
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def intersects_with(self, s2):
        s1 = ((self.x0, self.y0), (self.x1, self.y1))
        return intersects_with(s1, s2)


class VerticalWall(Wall):
    def __init__(self, min_dist, x_pos, bottom_y, top_y, thickness=0.0):
        assert bottom_y < top_y
        min_y = bottom_y - min_dist - thickness
        max_y = top_y + min_dist + thickness
        min_x = x_pos - min_dist - thickness
        max_x = x_pos + min_dist + thickness
        super().__init__(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_dist=min_dist,
            thickness=thickness,
        )
        self.endpoint1 = (x_pos+thickness, top_y+thickness)
        self.endpoint2 = (x_pos+thickness, bottom_y-thickness)
        self.endpoint3 = (x_pos-thickness, bottom_y-thickness)
        self.endpoint4 = (x_pos-thickness, top_y+thickness)


class HorizontalWall(Wall):
    def __init__(self, min_dist, y_pos, left_x, right_x, thickness=0.0):
        assert left_x < right_x
        min_y = y_pos - min_dist - thickness
        max_y = y_pos + min_dist + thickness
        min_x = left_x - min_dist - thickness
        max_x = right_x + min_dist + thickness
        super().__init__(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_dist=min_dist,
            thickness=thickness,
        )
        self.endpoint1 = (right_x+thickness, y_pos+thickness)
        self.endpoint2 = (right_x+thickness, y_pos-thickness)
        self.endpoint3 = (left_x-thickness, y_pos-thickness)
        self.endpoint4 = (left_x-thickness, y_pos+thickness)
