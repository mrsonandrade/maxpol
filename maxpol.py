#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__      = "Emerson Andrade"
__version__   = "1.0.0"
__date__   = "May 2024"

import csv
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from pymoo.optimize import minimize
from pymoode.algorithms import DE
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.termination import Termination

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


class ObjectiveFunction(ElementwiseProblem):
        
    def __init__(self, given_polygon, lower_limit, upper_limit, dimensions):
        self.given_polygon = given_polygon
        dimensions = dimensions
        xl = lower_limit
        xu = upper_limit
        super().__init__(n_var=dimensions, n_obj=1, n_ieq_constr=0, xl=xl, xu=xu)

    def trapezoid(self, x0, y0, length, base, top):
        polygon = Polygon([(x0,        y0+base/2.),
                          (x0+length, y0+top/2.),
                          (x0+length, y0-top/2.0),
                          (x0,        y0-base/2.0)])
        return polygon

    def objective_function(self, x, given_polygon):

        x0 =     x[0]
        y0 =     x[1]
        length = x[2]
        base =   x[3]
        top =    x[4]

        trapezoid_polygon = self.trapezoid(x0, y0, length, base, top)

        score = 10000.0
        if given_polygon.contains(trapezoid_polygon):
            score = 1.0/trapezoid_polygon.area

        return score
            
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.objective_function(x, self.given_polygon)

class FunctionCallTermination(Termination):

    def __init__(self, ref, tol, n_max_evals=float("inf")) -> None:
        super().__init__()

        self.tol = tol
        self.ref = ref
        self.n_max_evals = n_max_evals

    def _update(self, algorithm):

        # the object from the current iteration
        current = self._data(algorithm)

        if self.n_max_evals is None:
            return 0.0

        elif (current-self.ref) <= self.tol:
            return 1.0

        else:
            return algorithm.evaluator.n_eval / self.n_max_evals

    def _data(self, algorithm):
        opt = algorithm.opt
        f = opt.get("f")

        if len(f) > 0:
            return f.min()
        else:
            return np.inf

class Optimization():

    def __init__(self, xp, yp, given_polygon, iterations, population_size, F, CR, seed, max_evals):
        self.xp = xp
        self.yp = yp
        self.given_polygon = given_polygon
        self.iterations = iterations
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.seed = seed
        self.max_evals = max_evals
        
        self.dimensions = 5
        self.variant = "DE/rand/1/bin"

        # trapezio limits: x0, y0, length, base, top
        self.lower_limit = np.array([np.min(self.xp),
                                     np.min(self.yp),
                                     0.1,
                                     0.1,
                                     0.1],
                                    dtype=float)
        self.upper_limit = np.array([np.max(self.xp),
                                     np.max(self.yp),
                                     np.max(self.xp)-np.min(self.xp),
                                     np.max(self.xp)-np.min(self.xp),
                                     np.max(self.xp)-np.min(self.xp)],
                                    dtype=float)

    def trapezoid(self, x0, y0, length, base, top):
        polygon = Polygon([(x0,        y0+base/2.),
                          (x0+length, y0+top/2.),
                          (x0+length, y0-top/2.0),
                          (x0,        y0-base/2.0)])
        return polygon

    def optimize(self):

        fx = DE(pop_size=self.population_size,
                variant=self.variant,
                CR=self.CR,
                F=self.F,
                gamma=1e-4,
                de_repair="to-bounds")

        termination = FunctionCallTermination(ref = 0.0,
                                              tol = 1e-8,
                                              n_max_evals = self.max_evals)

        result = minimize(ObjectiveFunction(self.given_polygon,
                                            self.lower_limit,
                                            self.upper_limit,
                                            self.dimensions),
                          fx,
                          termination,
                          seed=self.seed,
                          save_history=True,
                          verbose=False)

        xo, yo = self.trapezoid(result.X[0], result.X[1], result.X[2], result.X[3], result.X[4]).exterior.xy

        return result, xo, yo

def main():

    xp,yp,given_polygon = get_given_polygon('given_polygon.csv')

    seed = 1
    population_size = 10
    F = 1.0
    CR = 0.5
    max_evals = 9000
    iterations = int(max_evals/population_size)

    problem = Optimization(xp, yp, given_polygon, iterations, population_size, F, CR, seed, max_evals)
    result, xo, yo = problem.optimize()
    
    plot_result(xp, yp, xo, yo)

main()
    
