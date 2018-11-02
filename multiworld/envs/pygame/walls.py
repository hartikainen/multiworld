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
            side='top'
        )
        self.bottom_segment = Segment(
            min_x,
            min_y,
            max_x,
            min_y,
            side='bottom'
        )
        self.left_segment = Segment(
            min_x,
            min_y,
            min_x,
            max_y,
            side='left'
        )
        self.right_segment = Segment(
            max_x,
            min_y,
            max_x,
            max_y,
            side='right'
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

        sides = (
            self.top_segment,
            self.right_segment,
            self.bottom_segment,
            self.left_segment,
        )

        if any(side.intersects_with(segment) for side in sides):
            return True

        for side in sides:
            if side.intersects_with(segment):
                return True
            if ((np.all(start == (side.x0, side.y0))
                 or np.all(start == (side.x1, side.y1)))
                and self.contains_point(end)):
                return True

        return False

    def handle_collision(self, start_point, end_point):
        trajectory_segment = (start_point, end_point)
        old_end_point = end_point

        sides = (
            self.top_segment,
            self.right_segment,
            self.bottom_segment,
            self.left_segment,
        )

        for side in sides:
            if ((np.all(start_point == (side.x0, side.y0))
                 or np.all(start_point == (side.x1, side.y1)))
                and self.contains_point(end_point)):
                end_point = start_point
                return end_point

        if self.top_segment.intersects_with(trajectory_segment):
            end_point[1] = self.max_y
        if self.bottom_segment.intersects_with(trajectory_segment):
            end_point[1] = self.min_y
        if self.right_segment.intersects_with(trajectory_segment):
            end_point[0] = self.max_x
        if self.left_segment.intersects_with(trajectory_segment):
            end_point[0] = self.min_x

        assert not self.contains_point(end_point), (
            start_point, old_end_point)

        return end_point


class Segment(object):
    def __init__(self, x0, y0, x1, y1, side):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.side = side

    def intersects_with(self,
                        s2,
                        include_end_points=False,
                        exclude_overlapping=False,
                        stop=False):
        s1 = ((self.x0, self.y0), (self.x1, self.y1))

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

        determinants = np.array([np.linalg.det(case) for case in cases])
        signs = np.sign(determinants)

        if np.all(determinants[2:] == 0):
            return False

        if ((signs[0] == signs[1] or signs[1] == 0)
            or (signs[2] == signs[3] or np.any(signs[2:] == 0))):
            return False

        expected_sign = {
            'top': 1,
            'right': -1,
            'bottom': -1,
            'left': 1,
        }[self.side]

        begins_from_side = determinants[0] == 0
        if (begins_from_side and (signs[1] in (expected_sign, 0))):
            return False

        return True


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
