# maxpol
 Find the maximally inscribed polygon for another given polygon.

 Basically, an optimization procedure is implemented, where:
 * To optimize the largest *optimized polygon* inside our *given polygon*, the package [`pymoode`](https://github.com/mooscaliaproject/pymoode) is used, which is a Python framework for Differential Evolution.
 * To check if the *given polygon* contains the *optimized polygon* (and compute their areas) the package [`shapely`](https://shapely.readthedocs.io/en/stable/manual.html) is used, which is a Python package for set-theoretic analysis and manipulation of planar features using functions from the well known and widely deployed GEOS library.
 * The solution also depends on the libraries: `csv`, `numpy`, and `matplotlib`.

 For details, please, see the figure *result.png* and check the code *maxpol.py*.