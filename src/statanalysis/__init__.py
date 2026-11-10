"""
Statistical analysis for the partitioning-strategy study (protocol section 13).

NAMED `statanalysis`, NOT `statistics`, DELIBERATELY. `src/` is on `sys.path`,
so a package here named `statistics` shadows the standard-library module of that
name for every importer in the process. That is not hypothetical: it broke
`geopandas.explore`, which does `from statistics import mean`, and with it
`libpysal` and the whole spatial-dependence analysis. Any name added directly
under `src/` must not collide with the standard library.
"""
