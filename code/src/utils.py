import numpy as np

def distribute_line_points(A, B, K):
    '''
    Distribute K points on the line segment between A and B.
    Input: A, B (numpy arrays), K (int)
    Output: points (numpy array)
    '''
    A, B = np.array(A), np.array(B)
    t = np.linspace(0, 1, K)
    return np.array([(1 - ti) * A + ti * B for ti in t])


def distribute_triangle_points(A, B, C, K):
    '''
    Distribute K points on the triangle defined by A, B, and C.
    Input: A, B, C (numpy arrays), K (int)
    Output: points (numpy array)
    '''
    A, B, C = np.array(A), np.array(B), np.array(C)
    points = []
    for i in range(K):
        for j in range(K - i):
            a = i / (K - 1)
            b = j / (K - 1)
            c = 1 - a - b
            point = a * A + b * B + c * C
            points.append(point)
    return np.array(points)