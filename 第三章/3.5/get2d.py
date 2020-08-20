import numpy as np

def GetProjectivePoint_2D(point, line):
    a = point[0]
    b = point[1]
    k = line[0]
    t = line[1]

    if k == 0:
        return[a, t]
    elif k == np.inf:
        return[0, b]
    x = (a+k*b-k*t)/(k*k+1)
    y = k*x+t
    return [x, y]
