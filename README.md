# maxpol
 Find the maximally inscribed polygon for another given polygon.

An optimization procedure is implemented, where:
 * To optimize the largest *optimized polygon* inside our *given polygon*, the package [`pymoode`](https://github.com/mooscaliaproject/pymoode) is used, which is a Python framework for Differential Evolution.
     * Please, note that this implementation has only the trapezoid polygon as an *optimized polygon* option. However, other polygons are easy to implement by following the same structure.
 * To check if the *given polygon* contains the *optimized polygon* (and compute their areas) the package [`shapely`](https://shapely.readthedocs.io/en/stable/manual.html) is used, which is a Python package for set-theoretic analysis and manipulation of planar features using functions from the well known and widely deployed GEOS library.
 * The solution also depends on the libraries: `csv`, `numpy`, and `matplotlib`.

 For details, please, see the figure *result.png* and check the code *maxpol.py*.