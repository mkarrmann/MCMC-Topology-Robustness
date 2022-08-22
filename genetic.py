from copy import deepcopy
import json
import os
import argparse
import logging
from typing import Tuple
from functools import partial
from statistics import mean, median, stdev
from typing import List
import multiprocessing as mp
from config import CONFIG
from gerrychain.constraints.contiguity import affected_parts
from gerrychain import Graph, MarkovChain
from gerrychain.constraints import (Validator, within_percent_of_ideal_population)
from gerrychain.updaters import Election, Tally, cut_edges
from gerrychain.partition import Partition
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from special_edge_graph import SpecialEdgeGraph

# For reproducibility, uses gerrychain's built-in random module
# to ensure a consistent seed.
from gerrychain.random import random

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gerry_chain_steps", help="Length of gerrychain steps", type=int)

# Default in macOS is spawn. However, this breaks the logic for using a single, shared
# config module that is dynamically modified. Override to use fork in all environments.
mp.set_start_method('fork')

# Maps from name of election statistics to the appropriate function
ELECTION_MAP = {
    'seats' : (lambda x: x[CONFIG['ELECTION_NAME']].seats('PartyA')),
    'won' : (lambda x: x[CONFIG['ELECTION_NAME']].seats('PartyA')),
    'efficiency_gap' : (lambda x: x[CONFIG['ELECTION_NAME']].efficiency_gap()),
    'mean_median' : (lambda x: x[CONFIG['ELECTION_NAME']].mean_median()),
    'mean_thirdian' : (lambda x: x[CONFIG['ELECTION_NAME']].mean_thirdian()),
    'partisan_bias' : (lambda x: x[CONFIG['ELECTION_NAME']].partisan_bias()),
    'partisan_gini' : (lambda x: x[CONFIG['ELECTION_NAME']].partisan_gini()),
}


def create_directory() -> str:
    """Creates experiment directory to track experiment configuration information and output.
    Args:
        CONFIG file for experiment

    Returns:
        String: name of experiment directory
    """
    num = 0
    suffix = lambda x: f'-{x}' if x != 0 else ''
    while os.path.exists(CONFIG['EXPERIMENT_NAME'] + suffix(num)):
        num += 1
    os.makedirs(CONFIG['EXPERIMENT_NAME'] + suffix(num))
    config_file = os.path.join(CONFIG['EXPERIMENT_NAME'] + suffix(num), 'config.json')
    with open(config_file, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    return CONFIG['EXPERIMENT_NAME'] + suffix(num)

def load_graph():
    """Loads graph from json file, along with additional preprocessing steps.
    Returns:
        GerryChain.Graph object
    """
    graph = Graph.from_json(CONFIG['INPUT_GRAPH_FILE'])
    for node in graph.nodes():
        graph.nodes[node]['x'] = graph.nodes[node][CONFIG['X_POSITION']]
        graph.nodes[node]['y'] = graph.nodes[node][CONFIG['Y_POSITION']]
    return graph

def is_original_contiguous(partition: Partition, special_graph: SpecialEdgeGraph) -> bool:
    """ Checks if the subgraphs of the Partition, created by an MCMC step,
    are contiguous when only considering the original graph.
    Args:
        partition: Partition object
    Returns:
        Boolean
    """
    return all(
        special_graph.is_original_subgraph_connected([n for n in partition.subgraphs[part]]) \
            for part in affected_parts(partition)
    )

def _init_chain(special_graph: SpecialEdgeGraph, num_steps: int) -> MarkovChain:
    """Runs Recom on graph
    Args:
        special_graph: SpecialEdgeGraph object
        num_steps: number of steps in the chain. If None, uses value from CONFIG

    Returns:
        MarkovChain object
    """
    graph = special_graph.graph
    # List of districts in original graph
    parts = list(set([graph.nodes[node][CONFIG['ASSIGN_COL']] for node in graph.nodes()]))
    # Ideal population of districts
    ideal_pop = sum([graph.nodes[node][CONFIG['POP_COL']] for node in graph.nodes()]) / len(parts)
    election = Election(
        CONFIG['ELECTION_NAME'],
        {
            'PartyA': CONFIG['PARTY_A_COL'],
            'PartyB': CONFIG['PARTY_B_COL'],
        }
    )
    updaters = {
        'population': Tally(CONFIG['POP_COL']),
        'cut_edges': cut_edges,
        CONFIG['ELECTION_NAME'] : election,
    }
    init_part = Partition(graph=graph, assignment=CONFIG['ASSIGN_COL'], updaters=updaters)
    popbound = within_percent_of_ideal_population(init_part, CONFIG['EPSILON'])
    # We check if the subgraphs are contiguous under the original graph, as
    # opposed to the modified special graph. This is because the set of valid
    # districting plans can conceivably be considered "politically relevant",
    # while the purpose of this experiment is to demonstrate that modifying
    # purely politically irrelevant features can have an impact of MCMC
    # analysis.
    is_contiguous = partial(is_original_contiguous, special_graph=special_graph)

    proposal = partial(
        recom,
        pop_col=CONFIG['POP_COL'],
        pop_target=ideal_pop,
        epsilon=CONFIG['EPSILON'],
    )

    return MarkovChain(
        proposal=proposal,
        constraints=Validator([popbound, is_contiguous]),
        accept=always_accept,
        initial_state=init_part,
        total_steps= num_steps,
    )


def run_chain_score(special_graph: SpecialEdgeGraph, id: str) -> float:
    """Runs Recom on graph and returns average of election statistics.
    """
    chain = _init_chain(special_graph, CONFIG['SCORE_CHAIN_STEPS'])
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
    chain = _init_chain(special_graph, CONFIG['GERRY_CHAIN_STEPS'])

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

def mutate_graph(special_graph: SpecialEdgeGraph) -> SpecialEdgeGraph:
    """Mutates the graph in place by adding and removing random edges"""
    # Sample both the number of edges to remove and to add from the same normal distribution
    num_remove = int(
        random.gauss(CONFIG['AVG_EDGE_CHANGES'], CONFIG['STD_EDGE_CHANGES'])
    )

    special_graph.remove_random_special_edges(num_remove)

    num_add = int(
        random.gauss(CONFIG['AVG_EDGE_CHANGES'], CONFIG['STD_EDGE_CHANGES'])
    )

    special_graph.add_random_special_edges(num_add)

    return special_graph

def init_special_graphs(graph: Graph) -> Tuple[List[SpecialEdgeGraph], float]:
    """Initializes the list of SpecialEdgeGraphs to be used in the experiment.
    Args:
        graph: Graph object
    Returns:
        List of SpecialEdgeGraph objects
        Original graph's average election statistic
    """
    def _set_dual_distance(special_graph: SpecialEdgeGraph, part: Partition) -> SpecialEdgeGraph:
        """Sets the distance of each node of the dual graph from the crossing
        edges of the partition

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

        visited, prev_nodes = set(crossing_faces), crossing_faces
        while len(visited) < special_graph.dual.number_of_nodes():
            next_nodes = []
            for node in prev_nodes:
                for neighbor in special_graph.dual.neighbors(node):
                    if neighbor not in visited:
                        next_nodes.append(neighbor)
                        visited.add(neighbor)
                        special_graph.dual.nodes[neighbor]['distance'] = \
                            special_graph.dual.nodes[node]['distance'] + 1
            prev_nodes = next_nodes

        return special_graph

    logging.info("Initializing special graphs")
    # Number of graphs per generation is equal to the number of sexual offspring
    # plus number of asexual offspring
    num_graphs = CONFIG['NUM_ASEXUAL'] + CONFIG['NUM_SEXUAL']
    gerrys, orig_score = run_chain_partition(SpecialEdgeGraph(graph), num_graphs)
    logging.info(f"Generated gerrymaders. Avg score: {orig_score}.")
    special_graphs = [
        SpecialEdgeGraph(graph) for _ in range(CONFIG['NUM_ASEXUAL'] + CONFIG['NUM_SEXUAL'])
    ]

    # Metamander around gerrys.
    for graph, gerry in zip(special_graphs, gerrys):
        _set_dual_distance(graph, gerry)
        faces = list(graph.dual.nodes(data=True))
        for face in faces:
            # Add a random edge to every face sufficiently far away from boundary
            # of corresponding gerrymandered partition.
            if face[1]['distance'] > CONFIG['DISTANCE_TO_METAMANDER']:
                graph.add_random_edge(face[0])

    return special_graphs, orig_score

def create_next_generation(
    prev_generation: List[SpecialEdgeGraph],
    alive: List,
) -> List[SpecialEdgeGraph]:
    """Creates the next generation of SpecialEdgeGraphs according to genetic
    algorithm

    Args:
        prev_generation: Previous generation. Each element
        is a tuple of (SpecialEdgeGraph, score)
        alive: List of previous special graphs that are still alive. Each element
        is a tuple of (SpecialEdgeGraph, score)
        special_graph[i]
    Returns:
        Next generation of SpecialEdgeGraphs
    """
    # Sort by score
    sorted_graphs_and_scores = sorted(
        prev_generation + alive,
        key=lambda x: x[1],
        reverse=True,
    )[:2]
    sorted_graphs = [g for g, _ in sorted_graphs_and_scores]

    # Form NUM_ASEXUAL mutations of the top graph
    asexual_offspring = [
        mutate_graph(deepcopy(sorted_graphs[0])) for _ in range(CONFIG['NUM_ASEXUAL'])
    ]

    # Form NUM_SEXUAL mutations of the top two graphs
    # Graphs are combined by adding them together. The addition operator is
    # overloaded such that this combines their special edges.
    sexual_offspring = []
    for _ in range(CONFIG['NUM_SEXUAL']):
        g1 = deepcopy(sorted_graphs[0])
        g2 = deepcopy(sorted_graphs[1])
        # Randomly remove a third of each special edges
        g1.remove_random_special_edges(int(len(g1.special_edges) / 3))
        g2.remove_random_special_edges(int(len(g2.special_edges) / 3))
        sexual_offspring.append(g1 + g2)

    logging.info(f"New alive: {len(set(sorted_graphs_and_scores) - set(alive))}")
    return asexual_offspring + sexual_offspring, sorted_graphs_and_scores


def main():
    """Runs genetic algorithm. Runs as a never ending loop, and saves intermediate
    results to disk. Therefore, the algorithm will run for arbitrarily long,
    and you can stop it once you are happy with the results. REMEMBER TO KILL
    THE PROCESS IF YOU'RE USING A SHARED MACHINE OR PAY FOR COMPUTE!

    At a high level, given a base graph, the algorithm attempts to find a
    SpecialEdgeGraph (which extends the base graph be adding edges) which
    maximizes the chosen election statistic when running a basic Recom
    chain on it.

    The algorithm consists of a sequence of "generations" of SpecialEdgeGraphs.
    We ran a Recom chain on each SpecialEdgeGraph, and the average election
    statistic of the chain is the 'score' of the SpecialEdgeGraph (note that
    this is by far the most computationally expensive part of the algorithm).
    The graphs with the highest score are kept. The next generation is then
    random mutations of the SpecialEdgeGraphs with the highest score, as well
    as random mutations of the highest scores "combined together" (see
    the __add__ magic method of SpecialEdgeGraph for details).

    In essence, it's survival of the fittest of graphs.
    """
    # Note: there's lot of inefficiencies in this code (e.g. duplicate creation
    # of identical SpecialEdgeGraphs, doing mutations serially instead of in
    # parallel, etc.). However, because the overwhelming majority of the time
    # is spent on the Recom chains, it makes sense to keep things simple for
    # this logic.
    logging.info(f'Experiment: {CONFIG["EXPERIMENT_NAME"]}')
    graph = load_graph()
    logging.info(f'Loaded graph')
    cur_generation, orig_score = init_special_graphs(graph)
    scores, alive = [orig_score], []
    with mp.Pool(processes=CONFIG['NUM_PROCESSES']) as pool:
        logging.info(f'Experiment initialized')
        generation = 0
        while True:
            logging.info(f'\n\n\nStarting generation {generation}')

            # Run Recom chain on each SpecialEdgeGraph of the current generation
            results = pool.starmap(
                run_chain_score,
                zip(
                    cur_generation,
                    # Id is combination of generation number and index in generation
                    (f'{generation}-{i}' for i in range(len(cur_generation))),
                ),
            )

            # Log and save results
            logging.info(f'Generation {generation}. Results: {results} ' \
                f'Num edges: {[g.num_special_edges for g in cur_generation]}')
            logging.info(f'This run: Mean: {mean(results)}, ' \
                f'Median: {median(results)}, Std: {stdev(results)}, ' \
                f'Min: {min(results)}, Max: {max(results)}, Orig: {orig_score}'
            )
            logging.info(f'Max ind: {results.index(max(results))}')

            scores.extend(results)
            max_score = max(scores)
            logging.info(f'Runs so far: Mean: {mean(scores)}, ' \
                f'Median: {median(scores)} Std: {stdev(scores)}, ' \
                f'Min: {min(scores)}, Max: {max_score}, Orig: {orig_score}'
            )

            # Save scores. Each write overwrites the previous one.
            with open(os.path.join(CONFIG['EXPERIMENT_NAME'], 'stats.json'), 'w') as f:
                json.dump(scores, f)

            # If graph has max score, save it. Wait at least 25 generations to
            # ensure that the new max is meaningful.
            if max(results) == max_score:
                ind = results.index(max_score)
                cur_generation[ind].save(
                    os.path.join(CONFIG['EXPERIMENT_NAME'],
                    f'{generation}-{max_score}.json'),
                )

            cur_generation, alive = create_next_generation(
                list(zip(cur_generation, results)),
                alive,
            )

            generation += 1


if __name__ == '__main__':
    # Override CONFIG with command line arguments
    args = parser.parse_args()
    CONFIG['SCORE_CHAIN_STEPS'] = args.gerry_chain_steps or CONFIG['SCORE_CHAIN_STEPS']
    CONFIG['EXPERIMENT_NAME'] = create_directory()
    logging.info(CONFIG)
    main()