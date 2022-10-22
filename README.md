# MCMC-Topology-Robustness

Experiments demonstrating that MCMC methods for detecting gerrymandering are sensitive to changes in the underlying graph topology in previous unexplored ways.

To reproduce our exact results, run `conda env create -f environment_linux.yml` to create a conda environment identical to ours. Additionally, ensure that the environment variable `PYTHONHASHSEED` is set to `0`. You may follow the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#saving-environment-variables) for setting environment variables for setting environment variables in your conda environment.

## TODOs
Most of the code here started as something which was intended to be quickly hacked together, and then grew and evolved over time. As a result, despite being a small repo, it has several short comings. Some todos:
* `special_edge_graph.py` has many comments. This is to some degree justified in order to avoid geometric intuition, but really it needs less comments and more documentation.
* `SpecialEdgeGraphs` are modified in place. This was intended as an optimization to avoid the deep copying that making the class immutable would require. However, the intended purpose of this class is to be one step of a pipeline of running a MCMC recom chain, and deep copying graphs seem to be orders of magnitude faster than recom chains. As a result, this class should be made immutable. This would allow the class to inherit from `gerrychain.Graph` and have a simpler interface, among other benefits. Premature optimization truly is the root of all evil!
* The tests are really bad (but good enough for our purposes)! They should be rewritten if development on this repo continues.