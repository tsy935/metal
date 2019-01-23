# Create points (config -> point list)
# random or grid?
# Assign classes
# Create lfs
# Identify slices

import random

import numpy as np


def flip(y):
    if y == 0:
        raise Exception("Cannot flip the label of an abstain vote")
    else:
        return 1 if y == 2 else 2


class Region(object):
    def contains(self, p):
        """Returns a boolean expressing whether p is in the region"""
        return NotImplementedError(
            "Abstract class: children must implement contains()"
        )

    def center(self):
        """Returns the center of the given region"""
        return NotImplementedError(
            "Abstract class: children must implement center()"
        )

    def __contains__(self, p):
        return self.contains(p)


class Rectangle(Region):
    """A rectangle defined by (t)op, (b)ottom, (l)eft, and (r)ight edges."""

    def __init__(self, b, t, l, r):
        assert b < t
        assert l < r
        self.b = b
        self.t = t
        self.l = l
        self.r = r

    def contains(self, p):
        x, y = p
        return x > self.b and x < self.t and y > self.l and y < self.r

    def center(self):
        x = (self.l + self.r) / 2
        y = (self.b + self.t) / 2
        return (x, y)


class Square(Rectangle):
    def __init__(self, *args):
        super().__init__(*args)
        assert self.t - self.b == self.r - self.l


class Ellipse(Region):
    """An ellipse defined by (h,k), a, and b

    (x - h)^2/a^2 + (y - k)^2/b^2 = 1
    """

    def __init__(self, h, k, a, b):
        self.h = h
        self.k = k
        self.a = a
        self.b = b

    def contains(self, p):
        x, y = p
        return (x - self.h) ** 2 / self.a ** 2 + (
            y - self.k
        ) ** 2 / self.b ** 2 < 1

    def center(self):
        return (self.h, self.k)


class Circle(Ellipse):
    """A circle defined by (h,k) and r"""

    def __init__(self, h, k, r):
        self.h = h
        self.k = k
        self.a = r
        self.b = r


def create_points(rect, n, random=True):
    """Creates n points randomly distributed throughout a rectangular region"""
    assert isinstance(rect, Rectangle)
    if random:
        x = np.random.uniform(rect.l, rect.r, size=(n, 1))
        y = np.random.uniform(rect.b, rect.t, size=(n, 1))
        X = np.hstack((x, y))
    else:
        X = []
        s = np.ceil(np.sqrt(n))
        for x in np.linspace(rect.l, rect.r, num=s):
            for y in np.linspace(rect.b, rect.t, num=s):
                X.append([x, y])
        X = np.array(X)[:n]
    return X


def get_points(X, region):
    """Returns the members of a region and the corresponding indicator vector"""
    indices = [x in region for x in X]
    return X[indices, :], indices


def assign_y(X, Y, region, label):
    """Sets y=label for every point in X that falls within the region"""
    for i, x in enumerate(X):
        if x in region:
            Y[i] = label
    return Y.astype(int)


def assign_l(X, Y, region, props, accs):
    """Returns a label vector for an lf in the given region, accs, and props

    prop[l] = P(lambda_i != 0 | y_i = l)
    acc[i]: P(lambda_i = y | y_i = y, lambda_i != 0)
    """
    L_j = np.zeros_like(Y)
    for i, (x, y) in enumerate(zip(X, Y)):
        if x in region:
            if random.random() < props[y - 1]:
                if random.random() < accs[y - 1]:
                    L_j[i] = int(y)
                else:
                    L_j[i] = int(flip(y))
    return L_j.astype(int)
