#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__      = "Emerson Andrade"
__version__   = "1.0.0"
__date__   = "May 2024"

import csv
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from maxpol import Optimization, ObjectiveFunction

def read_csv(filename):
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        result = np.asarray(list(reader)[:])
        result = np.char.replace(result, ',', '.')
        result = result.astype('float64')
    return result

def get_given_polygon(filename):
    data = read_csv(filename)
    xp = data[:,0]
    yp = data[:,1]
    given_polygon = Polygon([(xi, yi) for xi,yi in zip(xp.tolist(), yp.tolist())])

    return [xp,yp,given_polygon]

def plot_result(xp, yp, xo, yo):
    fig, ax = plt.subplots(1, 1, figsize=(12,4), sharey=True, sharex=True)
    ax.plot(xp, yp, label='given polygon')
    ax.plot(xo, yo, label='optimal polygon')
    ax.legend()
    plt.show()
    #plt.savefig("result.png", dpi=300)
    plt.close()

def main():

    xp,yp,given_polygon = get_given_polygon('given_polygon.csv')

    shape = 'trapezoid'
    #shape = 'rectangle'

    problem = Optimization(xp, yp, given_polygon, shape)
    result = problem.optimize()

    if shape=='trapezoid':
        xo, yo = ObjectiveFunction.trapezoid(1, result[0], result[1], result[2], result[3], result[4]).exterior.xy
    if shape=='rectangle':
        xo, yo = ObjectiveFunction.trapezoid(1, result[0], result[1], result[2], result[3], result[3]).exterior.xy
    
    plot_result(xp, yp, xo, yo)

main()
    
