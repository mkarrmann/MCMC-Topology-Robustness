import itertools
import argparse
import logging
import os
import json
from copy import deepcopy
from typing import Tuple
from statistics import mean, median, stdev
from typing import List, Callable
import multiprocessing as mp
from networkx import connected_components
from config import CONFIG
from gerrychain import Graph
from gerrychain.partition import Partition
from special_edge_graph import SpecialEdgeGraph
from gerrychain.tree import recursive_seed_part
from utils import create_directory, load_graph, create_chain, ELECTION_MAP

# For reproducibility, uses gerrychain's built-in random module
# to ensure a consistent seed.
from gerrychain.random import random

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gerry_chain_steps", help="Length of gerrychain steps", type=int)

# Default in macOS is spawn. However, this breaks the logic for using a single, shared
# config module that is dynamically modified. Override to use fork in all environments.
mp.set_start_method('fork')

def run_chain_score(special_graph: SpecialEdgeGraph, assign: dict, id: str) -> float:
    """Runs Recom on graph and returns average of election statistics.
    """
    logging.info(f'{id}: Creating chain')
    chain = create_chain(special_graph, CONFIG['SCORE_CHAIN_STEPS'], assign)
    logging.info(f'{id}: Running chain')
    # Run chain, save stats of chosen statistic. Also saves the most gerrymandered
    # partition.
    stats = []
    for i, part in enumerate(chain):
        stat = ELECTION_MAP[CONFIG['ELECTION_STATISTIC']](part)
        stats.append(stat)
        if i % 500 == 0:
            logging.debug(f'{id}: {i} steps completed')

    return mean(stats)

def run_chain_partition(
    special_graph: SpecialEdgeGraph,
    num_parts: int,
) -> Tuple[List[Partition], float]:
    """Runs Recom on graph and returns most gerrymandered partitions.

    Args:
        special_graph: SpecialEdgeGraph object
        num_parts: number of partitions to return

    Returns:
        List of Partition objects
        Average of election statistics
    """
    chain = create_chain(special_graph, CONFIG['GERRY_CHAIN_STEPS'])

    # Run the chain, saving the most gerrymandered partitions.
    stats, max_parts = [], []
    for i, part in enumerate(chain):
        stat = ELECTION_MAP[CONFIG['ELECTION_STATISTIC']](part)
        stats.append(stat)
        # Keep only top num_parts partitions
        max_parts.append((stat, part))
        max_parts.sort(key=lambda x: x[0], reverse=True)
        max_parts = max_parts[:num_parts]
        if i % 500 == 0:
            logging.debug(f'{id}: {i} steps completed')

    return [part for stat, part in max_parts], mean(stats)

def _add_dual_scores(special_graph: SpecialEdgeGraph, part: Partition) -> SpecialEdgeGraph:
    # TODO figure out what to do with this
    """Adds the dual score of the given partition to the given special graph.\
        The dual score of a face is an indication of how close a node is to
        one of the boundaries of the partition classes. A high score indicates
        being close to the boundary (or on the boundary), while a low score
        being far away.

        Note that we just add to the dual score- this way we are only adding
        on top of a dual score that was already there.

    Args:
        special_graph (SpecialEdgeGraph): special graph to set distance on the
        dual graph of
        part (Partition): partition to set the distance around

    Returns:
        SpecialEdgeGraph: special graph being modified. Note is modified in
        place
    """
    crossing_faces = []
    # Note: this is a bit slow for large graphs.
    for face in special_graph.dual.nodes():
        for e in part['cut_edges']:
            if e[0] in face.nodes and e[1] in face.nodes:
                crossing_faces.append(face)
                special_graph.dual.nodes[face]['distance'] = 0

    score = 100 # Starting score assigned to faces on boundary
    visited, prev_nodes = set(crossing_faces), crossing_faces
    while len(visited) < special_graph.dual.number_of_nodes():
        next_nodes = []
        for node in prev_nodes:
            for neighbor in special_graph.dual.neighbors(node):
                if neighbor not in visited:
                    next_nodes.append(neighbor)
                    visited.add(neighbor)
                    special_graph.dual.nodes[neighbor]['score'] += score
        prev_nodes = next_nodes
        score //= CONFIG['DUAL_SCORE_DIVISOR']

    return special_graph

def score_by_party(special_graph: SpecialEdgeGraph, party='A') -> SpecialEdgeGraph:
    """Provides score to face based upon average score of nodes in face.

    Args:
        special_graph (SpecialEdgeGraph): special graph to set score on
        party (str, optional): party to get score from. Defaults to 'A'.

    Returns:
        SpecialEdgeGraph: special graph being modified. Note is modified in
        place
    """
    def _score(node) -> float:
        if special_graph.graph.nodes[node][CONFIG['PARTY_A_COL']] + special_graph.graph.nodes[node][CONFIG['PARTY_B_COL']] > 0:
            return (special_graph.graph.nodes[node][CONFIG['PARTY_A_COL']] - special_graph.graph.nodes[node][CONFIG['PARTY_B_COL']]) \
                / (special_graph.graph.nodes[node][CONFIG['PARTY_A_COL']] + special_graph.graph.nodes[node][CONFIG['PARTY_B_COL']]) \
                * (1 if party == 'A' else -1)
        return 0
    faces = list(special_graph.dual.nodes())
    for face in faces:
        face.score = mean([_score(node) for node in face.nodes])
    return special_graph

def init_special_graphs(graph: Graph, score_func: Callable) -> Tuple[List[SpecialEdgeGraph], float]:
    """Initializes the list of SpecialEdgeGraphs to be used in the experiment.
    Args:
        graph: Gerrychain Graph object
        score_func: Function mapping from Face to float. Used to determine which
        faces to metamander.
    Returns:
        List of SpecialEdgeGraph objects
        Original graph's average election statistic
    """

    logging.info("Initializing special graphs")
    special_graph = score_func(SpecialEdgeGraph(graph))
    graphs = [deepcopy(special_graph)
              for _ in range(1)]  # TODO config number graphs
    logging.info("Copied graphs")

    for graph in graphs:
        faces = list(graph.dual.nodes())
        for face in faces:
            # TODO recursively modify faces
            if face.score > 0: # TODO config
                graph.add_random_edge(face)
    logging.info("Added edges")
    return graphs


def init_parts(graph: Graph) -> List[Partition]:
    """Initializes the list of partitions to be used in the experiment.
    Creates one partition for each element of CONFIG['NUM_PARTS'].

    Args:
        graph (Graph): graph to partition

    Returns:
        List[Partition]: list of partitions
    """
    parts = []
    total_pop = sum([graph.nodes[node][CONFIG['POP_COL']]
                    for node in graph.nodes()])
    for num_parts in CONFIG['NUM_PARTS']:
        parts.append(recursive_seed_part(
            graph,
            parts=range(num_parts),
            pop_target=total_pop / num_parts,
            pop_col=CONFIG['POP_COL'],
            epsilon=CONFIG['EPSILON'],
        ))
    return parts

    #blue_nodes = [
    #    n for n in graph.nodes if (graph.nodes[n]['PRES16D'] + graph.nodes[n]['PRES16R'] > 0 and \
    #    graph.nodes[n]['PRES16D'] - graph.nodes[n]['PRES16R']) \
    #    / (graph.nodes[n]['PRES16D'] + graph.nodes[n]['PRES16R'] + 1) > 0.15
    #]
    #blue_graph = graph.subgraph(blue_nodes)
    #from collections import Counter
    #logging.info(Counter([len(comp) for comp in connected_components(blue_graph)]))
    #for comp in connected_components(blue_graph):
    #    if len(comp) > 200:
    #        try:
    #            g = SpecialEdgeGraph(graph.subgraph(comp))
    #            logging.info(g.unbounded_face.nodes)
    #        except AssertionError:
    #            # TODO remove
    #            logging.warning("uh oh")
    #            logging.warning([n for n in comp])
    #            g = SpecialEdgeGraph(graph.subgraph(comp))
    #            for face in g.dual.nodes():
    #                logging.warning(face.nodes)
    #            ns = graph.subgraph(comp)
    #            import networkx as nx
    #            nx.draw(ns, pos={n: (ns.nodes[n]['x'], ns.nodes[n]['y']) for n in ns.nodes}, with_labels=True)
    #            import matplotlib.pyplot as plt
    #            plt.show()
    #logging.info(f'Connected components: {len(list(connected_components(blue_graph)))}')
    ## Number of graphs per generation is equal to the number of sexual offspring
    ## plus number of asexual offspring
    #num_graphs = CONFIG['NUM_ASEXUAL'] + CONFIG['NUM_SEXUAL']
    #gerrys, orig_score = run_chain_partition(SpecialEdgeGraph(graph), num_graphs)
    #logging.info(f"Generated gerrymaders. Avg score: {orig_score}.")
    #special_graphs = [
    #    SpecialEdgeGraph(graph) for _ in range(CONFIG['NUM_ASEXUAL'] + CONFIG['NUM_SEXUAL'])
    #]

    ## Metamander around gerrys.
    #for graph, gerry in zip(special_graphs, gerrys):
    #    _set_dual_distance(graph, gerry)
    #    faces = list(graph.dual.nodes(data=True))
    #    for face in faces:
    #        # Add a random edge to every face sufficiently far away from boundary
    #        # of corresponding gerrymandered partition.
    #        if face[1]['distance'] > CONFIG['DISTANCE_TO_METAMANDER']:
    #            graph.add_random_edge(face[0])

    #return special_graphs, orig_score

def main():
    """Runs "manual" algorithm for metamandering graph
    """
    # Note: there's lot of inefficiencies in this code (e.g. duplicate creation
    # of identical SpecialEdgeGraphs, doing mutations serially instead of in
    # parallel, etc.). However, because the overwhelming majority of the time
    # is spent on the Recom chains, it makes sense to keep things simple for
    # this logic.
    logging.info(f'Experiment: {CONFIG["EXPERIMENT_NAME"]}')
    graph = load_graph()
    logging.info(f'Loaded graph')
    logging.info("Creating init parts")
    parts = init_parts(graph)
    logging.info("Created init parts")
    special_graphs = init_special_graphs(graph, globals()[CONFIG['SCORE_FUNC']]) + \
        [SpecialEdgeGraph(graph)]
    graph_part_pairs = list(itertools.product(parts, special_graphs))
    with mp.Pool(processes=CONFIG['NUM_PROCESSES']) as pool:
        # Run Recom chain on each SpecialEdgeGraph
        results = pool.starmap(
            run_chain_score,
            zip(
                [graph_part_pair[1] for graph_part_pair in graph_part_pairs],
                [graph_part_pair[0] for graph_part_pair in graph_part_pairs],
                (f'{"Orig" if i % 2 == 1 else "Special"}{len(set(pair[0].values()))}' for i, pair in enumerate(graph_part_pairs)),
            ),
        )
        logging.info(results)
        with open(os.path.join(CONFIG['EXPERIMENT_NAME'], 'results.json'), 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    # Override CONFIG with command line arguments
    args = parser.parse_args()
    CONFIG['SCORE_CHAIN_STEPS'] = args.gerry_chain_steps or CONFIG['SCORE_CHAIN_STEPS']
    CONFIG['EXPERIMENT_NAME'] = create_directory()
    logging.info(CONFIG)
    main()