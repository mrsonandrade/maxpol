#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__      = "Emerson Andrade"
__version__   = "1.0.0"
__date__   = "May 2024"

import numpy as np
from shapely.geometry import Polygon
from pymoo.optimize import minimize
from pymoode.algorithms import DE
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.termination import Termination


class ObjectiveFunction(ElementwiseProblem):
        
    def __init__(self, given_polygon, shape, lower_limit, upper_limit, dimensions):
        self.given_polygon = given_polygon
        self.shape = shape
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

        if self.shape=='trapezoid':
            trapezoid_polygon = self.trapezoid(x0, y0, length, base, top)
        if self.shape=='rectangle':
            trapezoid_polygon = self.trapezoid(x0, y0, length, base, base)

        score = 10000.0
        if given_polygon.contains(trapezoid_polygon):
            score = 1.0/np.sqrt(trapezoid_polygon.area)

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

    def __init__(self,
                 xp,
                 yp,
                 given_polygon,
                 shape='trapezoid',
                 iterations=2000,
                 population_size=10,
                 F=0.7,
                 CR=0.3,
                 seed=1,
                 max_evals=20000):
        self.xp = xp
        self.yp = yp
        self.given_polygon = given_polygon
        self.shape = shape
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
                                            self.shape,
                                            self.lower_limit,
                                            self.upper_limit,
                                            self.dimensions),
                          fx,
                          termination,
                          seed=self.seed,
                          save_history=True,
                          verbose=False)

        return result.X
    
