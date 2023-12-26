import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .stable_region_analytical import StableRegion
import yaml
from matplotlib import collections  as mc

def fibonacci_sphere(samples=2000):

    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples//2):
        x = 1 - (i / float(samples - 1)) * 2  # x goes from 1 to -1
        radius = np.sqrt(1 - x * x)  # radius at x

        theta = phi * i  # golden angle increment

        y = np.cos(theta) *     radius
        z = np.sin(theta) * radius

        points.append((x, y, z))
    velocity = np.array(points)
    return velocity
    
def velocity2icr(velocity):
    """
    Calculate ICR (Instantaneous Center of Rotation) for each velocity.
    """
    vx, vy, w = velocity[:,0], velocity[:,1], velocity[:,2]
    ICRs = []
    for i in range(len(vx)):
        if w[i] == 0:
            w[i] = 1e-6
        icr= np.array([-vy[i] / w[i], vx[i] / w[i]])
        ICRs.append(icr)
        
    return np.array(ICRs)