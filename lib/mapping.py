import numpy as np
import itertools
import math


ZERO = 0.01 # meters ~(1 centimeter)

    
def circle_intersection(p1, r1, p2, r2):
    

    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    d = ((x2-x1)**2 + (y2-y1)**2)**0.5  # distance between circle centers
    
    # check if circles intersect
    if d <= r1 + r2 and d >= abs(r1 - r2):
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h = (r1**2 - a**2)**0.5
        xm = x1 + a * (x2 - x1) / d
        ym = y1 + a * (y2 - y1) / d
        x1_i = xm + h * (y2 - y1) / d
        y1_i = ym - h * (x2 - x1) / d
        x2_i = xm - h * (y2 - y1) / d
        y2_i = ym + h * (x2 - x1) / d
        return (x1_i, y1_i), (x2_i, y2_i)
    else:
        return None, None

def equalZero(value):
    return value <= ZERO and value >= -ZERO

def intrsLineCircl(p1, p2, p3, r3):
    
    # extracting coordinates
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]

    # calculating line y = mx + c

    if not equalZero(x2-x1):
        m = (y2-y1)/(x2-x1) # slope 
    else:
        # vertical line
        if equalZero(x2-x1):
            X = (x2+x1)/2 # correction
            Y1 = np.sqrt(r3**2-(X-x3)**2) + y3
            Y2 = - np.sqrt(r3**2-(X-x3)**2) + y3
            return [(X, Y1), (X, Y2)]
        # horizontal line
        if equalZero(y2-y1):
            Y = (y2+y1)/2
            X1 = np.sqrt(r3**2-(Y-y3)**2) + x3
            X2 = - np.sqrt(r3**2-(Y-y3)**2) + x3
            return [(X1, Y), (X2, Y)]
    
    c = (x2-x1) * y1 - (y2-y1) * x1

    # Calculate discriminant
    disc = (2*m*(c-y3) - 2*x3)**2 - 4*(1+m**2)*((c-y3)**2-x3**2-r3**2)
    print(disc)

    if disc < -ZERO:
        # no intersections
        return None
    elif equalZero(disc):
        # one interection
        X = ((2*m*(c-y3) - 2*x3))/(2*(1+m**2))
        Y = m*X + c
        return (X, Y)
    else:
        # two interesection
        X1 = ((2*m*(c-y3) - 2*x3) + np.sqrt(disc))/(2*(1+m**2))
        X2 = ((2*m*(c-y3) - 2*x3) - np.sqrt(disc))/(2*(1+m**2))
        Y1 = m*X1 + c
        Y2 = m*X2 + c  
    
    return [(X1, Y1), (X2, Y2)]

def euclid_dist(p1, p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def midpoint(p1, p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    x = (x1+x2)/2
    y = (y1+y2)/2

    return (round(x,1), round(y,1))

def mapping_process(p1, p2, p3):
    comb = itertools.permutations([p1, p2, p3], 3)

    min_dist = 9999
    min_point = (None, None)
    
    for c in comb:
        print('------------')
        P1, P2 = circle_intersection(c[0][:2], c[0][2], c[1][:2], c[1][2])
        if P1 == None: continue

        # print(f'P1: {P1}, P2: {P2}')

        points = intrsLineCircl(P1, P2, c[2][:2], c[2][2])

        if points != None: 
            for p in [P1, P2]:
                if len(points) > 1: 
                    for point in points:
                        if min_dist > euclid_dist(point, p):
                            min_dist = euclid_dist(point, p)
                            min_point = midpoint(point, p)
                else: 
                    if min_dist > euclid_dist(points, p):
                        min_dist = euclid_dist(points, p)
                        min_point = midpoint(points, p)
        
    return min_point, min_dist
