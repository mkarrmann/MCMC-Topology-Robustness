from gerrychain.partition import Partition
from functools import partial
import os
import json
from special_edge_graph import SpecialEdgeGraph
from gerrychain import Graph
from config import CONFIG
from networkx import connected_components, is_connected
from gerrychain.constraints.contiguity import affected_parts
from gerrychain.proposals import recom
from gerrychain.constraints import (Validator, within_percent_of_ideal_population)
from gerrychain.updaters import Election, Tally, cut_edges
from gerrychain.accept import always_accept
from gerrychain.tree import recursive_seed_part
from gerrychain import Graph, MarkovChain

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
    filename = CONFIG['INPUT_GRAPH_FILE']
    if filename.split('.')[-1] == 'json':
        graph = Graph.from_json(CONFIG['INPUT_GRAPH_FILE'])
    else:
        graph = Graph.from_file(CONFIG['INPUT_GRAPH_FILE'])

    # Remove disconnected nodes:
    component = list(connected_components(graph))
    biggest_component_size = max(len(c) for c in component)
    problem_components = [c for c in component if len(c) != biggest_component_size]
    for c in problem_components:
        for n in c:
            graph.remove_node(n)

    assert is_connected(graph)

    # Set pos x,y attributes:
    for node in graph.nodes():
        # TODO
        if 'pos' in graph.nodes[node]:
            graph.nodes[node]['x'] = graph.nodes[node]['pos']['coordinates'][0]
            graph.nodes[node]['y'] = graph.nodes[node]['pos']['coordinates'][1]
        else:
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

def create_chain(special_graph: SpecialEdgeGraph, num_steps: int, assign: dict = None) -> MarkovChain:
    """Runs Recom on graph
    Args:
        special_graph: SpecialEdgeGraph object
        num_steps: number of steps in the chain. If None, uses value from CONFIG

    Returns:
        MarkovChain object
    """
    graph = special_graph.graph

    election = Election(
        CONFIG['ELECTION_NAME'],
        {
            'PartyA': CONFIG['PARTY_A_COL'],
            'PartyB': CONFIG['PARTY_B_COL'],
        }
    )
    updaters = {
        'population': Tally(CONFIG['POP_COL']),
        # TODO helps with a seeming bug
        CONFIG['POP_COL']: Tally(CONFIG['POP_COL']),
        'cut_edges': cut_edges,
        CONFIG['ELECTION_NAME']: election,
    }
    #num_parts = CONFIG['NUM_PARTS']
    #ideal_pop = sum([graph.nodes[node][CONFIG['POP_COL']]
    #                for node in graph.nodes()]) / num_parts
    #if not assign:
    #    assign = recursive_seed_part(
    #        graph,
    #        parts=range(num_parts),
    #        pop_target=ideal_pop,
    #        pop_col=CONFIG['POP_COL'],
    #        epsilon=CONFIG['EPSILON'],
    #    )
    init_part = Partition(graph, assign, updaters=updaters)
    popbound = within_percent_of_ideal_population(init_part, CONFIG['EPSILON'])
    # We check if the subgraphs are contiguous under the original graph, as
    # opposed to the modified special graph. This is because the set of valid
    # districting plans can conceivably be considered "politically relevant",
    # while the purpose of this experiment is to demonstrate that modifying
    # purely politically irrelevant features can have an impact of MCMC
    # analysis.
    is_contiguous = partial(is_original_contiguous,
                            special_graph=special_graph)

    proposal = partial(
        recom,
        pop_col=CONFIG['POP_COL'],
        pop_target= sum([
            graph.nodes[node][CONFIG['POP_COL']] for node in graph.nodes()
        ]) / len(init_part.parts),
        epsilon=CONFIG['EPSILON'],
    )

    #from gerrychain.constraints import contiguous

    return MarkovChain(
        proposal=proposal,
        constraints=Validator([popbound, is_contiguous]),
        accept=always_accept,
        initial_state=init_part,
        total_steps= num_steps,
    )