import json
import logging
import multiprocessing as mp
import os
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from functools import partial
from typing import Callable, List, Optional, Set, Union

import geopandas as gpd
import gerrychain
import networkx as nx
import toml
from gerrychain import MarkovChain, Partition
from gerrychain.accept import always_accept
from gerrychain.constraints import (
    Validator,
    contiguous,
    within_percent_of_ideal_population,
)
from gerrychain.proposals import recom
from gerrychain.tree import (
    bipartition_tree,
    bipartition_tree_random,
    recursive_seed_part,
)
from gerrychain.updaters import Election, Tally, cut_edges
from networkx import connected_components

# Default in macOS is spawn. However, this breaks the logic for using a single, shared
# config module that is dynamically modified. Override to use fork in all environments.
mp.set_start_method("fork")

# Maps from name of election statistics to the appropriate function
ELECTION_MAP = {
    "seats": (lambda x: x[CONFIG["ELECTION_NAME"]].seats("PartyA")),
    "won": (lambda x: x[CONFIG["ELECTION_NAME"]].seats("PartyA")),
    "efficiency_gap": (lambda x: x[CONFIG["ELECTION_NAME"]].efficiency_gap()),
    "mean_median": (lambda x: x[CONFIG["ELECTION_NAME"]].mean_median()),
    "mean_thirdian": (lambda x: x[CONFIG["ELECTION_NAME"]].mean_thirdian()),
    "partisan_bias": (lambda x: x[CONFIG["ELECTION_NAME"]].partisan_bias()),
    "partisan_gini": (lambda x: x[CONFIG["ELECTION_NAME"]].partisan_gini()),
}


CONFIG: dict = {}

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Copied from tree.py with fixes.
# TODO raise PR to gerrychain with fixes
def recursive_seed_part_inner(
    graph: nx.Graph,
    num_dists: int,
    pop_target: Union[float, int],
    pop_col: str,
    epsilon: float,
    method: Callable = partial(bipartition_tree, max_attempts=10000),
    node_repeats: int = 1,
    n: Optional[int] = None,
    ceil: None = None,
) -> List[Set]:
    """
    Inner function for recursive_seed_part.
    Returns a partition with ``num_dists`` districts balanced within ``epsilon`` of
    ``pop_target``.
    Splits graph into num_chunks chunks, and then recursively splits each chunk into
    ``num_dists``/num_chunks chunks.
    The number num_chunks of chunks is chosen based on ``n`` and ``ceil`` as follows:
        If ``n`` is None, and ``ceil`` is None, num_chunks is the largest prime factor
        of ``num_dists``.
        If ``n`` is None and ``ceil`` is an integer at least 2, then num_chunks is the
        largest prime factor of ``num_dists`` that is less than ``ceil``
        If ``n`` is a positive integer, num_chunks equals n.
    Finally, if the number of chunks as chosen above does not divide ``num_dists``, then
    this function bites off a single district from the graph and recursively partitions
    the remaining graph into ``num_dists``-1 districts.

    :param graph: The graph
    :param num_dists: number of districts to partition the graph into
    :param pop_target: Target population for each part of the partition
    :param pop_col: Node attribute key holding population data
    :param epsilon: How far (as a percentage of ``pop_target``) from ``pop_target`` the parts
        of the partition can be
    :param method: Function used to find balanced partitions at the 2-district level
    :param node_repeats: Parameter for :func:`~gerrychain.tree_methods.bipartition_tree` to use.
    :param n: Either a positive integer (greater than 1) or None. If n is a positive integer,
        this function will recursively create a seed plan by either biting off districts from
        graph or dividing graph into n chunks and recursing into each of these. If n is None,
        this function prime factors ``num_dists``=n_1*n_2*...*n_k (n_1 > n_2 > ... n_k) and
        recursively partitions graph into n_1 chunks.
    :param ceil: Either a positive integer (at least 2) or None. Relevant only if n is None.
        If ``ceil`` is a positive integer then finds the largest factor of ``num_dists`` less
        than or equal to ``ceil``, and recursively splits graph into that number of chunks, or
        bites off a district if that number is 1.
    :return: New assignments for the nodes of ``graph``.
    :rtype: List of lists, each list is a district
    """
    from gerrychain.tree import get_max_prime_factor_less_than, get_seed_chunks

    # Chooses num_chunks
    if n is None:
        if ceil is None:
            num_chunks = get_max_prime_factor_less_than(num_dists, num_dists)
        elif ceil >= 2:
            num_chunks = get_max_prime_factor_less_than(num_dists, ceil)
        else:
            raise ValueError("ceil must be None or at least 2")
    elif n > 1:
        num_chunks = n
    else:
        raise ValueError("n must be None or a positive integer")

    # base case
    if num_dists == 1:
        return [set(graph.nodes)]

    # bite off a district and recurse into the remaining subgraph
    elif num_chunks is None or num_dists % num_chunks != 0:
        remaining_nodes = set(graph.nodes)
        nodes = method(
            graph.subgraph(remaining_nodes),
            pop_col=pop_col,
            pop_target=pop_target,
            epsilon=epsilon,
            node_repeats=node_repeats,
        )
        remaining_nodes -= nodes
        assignment = [nodes] + recursive_seed_part_inner(
            graph.subgraph(remaining_nodes),
            num_dists - 1,
            pop_target,
            pop_col,
            epsilon,
            n=n,
            ceil=ceil,
            method=method,  # Added change here
        )

    # split graph into num_chunks chunks, and recurse into each chunk
    elif num_dists % num_chunks == 0:
        chunks = get_seed_chunks(
            graph, num_chunks, num_dists, pop_target, pop_col, epsilon, method=method
        )

        assignment = []
        for chunk in chunks:
            chunk_assignment = recursive_seed_part_inner(
                graph.subgraph(chunk),
                num_dists // num_chunks,
                pop_target,
                pop_col,
                epsilon,
                n=n,
                ceil=ceil,
                method=method,  # Added change here
            )
            assignment += chunk_assignment

    return assignment


# Monkey patch recursive_seed_part_inner to fix bug
import gerrychain.tree

gerrychain.tree.recursive_seed_part_inner = recursive_seed_part_inner


def condense(graph: nx.Graph, u: int, v: int, props: list[str]) -> nx.Graph:
    """
    Condense graph by contracting u and v into a single vertex.

    Args:
        graph: Graph to condense.
        u: Vertex u.
        v: Vertex v.
        props: List of properties to add together
    """

    total_props: dict[str, float] = {
        prop: graph.nodes[u][prop] + graph.nodes[v][prop] for prop in props
    }
    graph = nx.contracted_nodes(graph, u, v, self_loops=False, copy=True)

    for prop, val in total_props.items():
        graph.nodes[u][prop] = val

    return graph


def random_condense(
    graph: nx.Graph,
    assign: dict,
    props: list[str],
    max_pop: int | float = float("inf"),
) -> nx.Graph:
    """Randomly contracts a pair of adjacent nodes

    Args:
        graph (nx.Graph): Graph to contract
        assign (dict): Partition assignment
        props (list[str]): List of properties to add together.
    """

    if len(graph.edges) == 0:
        return graph

    i = 0
    while True:
        u, v = random.choice(list(graph.edges))
        # Only contract if it doesn't break the partition
        if (
            assign[u] == assign[v]
            and graph.nodes[u][CONFIG["POP_COL"]] + graph.nodes[v][CONFIG["POP_COL"]]
            < max_pop
        ):
            return condense(graph, u, v, props)
        else:
            logger.debug(f"Skipping contraction between {u} and {v}")
        i += 1
        # This is all obviously very naive and slow, but this is simple, and relatively
        # little time is spent contracting the graphs compared to running the chains,
        # so it's fine.
        if max_pop < float("inf") and i % 1000000 == 0:
            logger.warning(
                f"Increase max_pop from {max_pop} to {(max_pop := max_pop * 1.1)}"
            )


def generate_random_contractions(
    graph: nx.Graph,
    min_blocks: int,
    num_parts: int,
    ideal_pop: float,
    begin_epsilon: float,
    end_epsilon: float,
    pop_col: str,
    props: list[str],
) -> list[nx.Graph]:
    """Generates a random contraction sequence. Must be a connected graph or this will
    break!

    Args:
        graph (nx.Graph): Graph to contract
        min_blocks (int): Minimum number of blocks to contract to
        num_parts (int): Number of parts of each partition
        ideal_pop (float): Ideal population of each part
        begin_epsilon (float): Maximum deviation from ideal population to start with
        end_epsilon (float): Maximum deviation from ideal population to end with
        pop_col (str): Name of population column
        properties (list[str]): List of properties to add together

    Returns:
        list[tuple[int, int]]: List of contractions
    """

    beginning_blocks = len(graph)
    logger.info(f"Starting with {beginning_blocks} nodes")

    # y = mx + b
    m = (end_epsilon - begin_epsilon) / (min_blocks - beginning_blocks)

    contractions: list[nx.Graph] = [graph]

    while len(graph) >= min_blocks:
        # Generate a random part, and only include contractions that preserve partitions.
        # This helps enusre that we don't contract the graph in a way that preventts
        # creating equally-populated partitions. By generating a new seed part at each
        # step, we also avoid contracting "around" any single partition.
        epsilon = m * (len(graph) - beginning_blocks) + begin_epsilon
        logging.info(f"Using epsilon {epsilon} for generating seed part")
        assign = recursive_seed_part(
            graph,
            parts=range(num_parts),
            pop_target=ideal_pop,
            pop_col=pop_col,
            epsilon=epsilon,
            # Increase from default of 10k to 500k to avoid failure
            method=partial(bipartition_tree_random, max_attempts=500_000),
        )
        logger.info("Generated new seed part for contracting")
        for _ in range(CONFIG["CONTRACTIONS_PER_STEP"]):
            graph = random_condense(graph, assign=assign, props=props)
        contractions.append(graph)
        logger.info(f"Contractions {len(contractions)} to {len(graph)} nodes")

    return contractions


def _generate_gerrys(
    graph: gerrychain.Graph,
    num_parts: int,
    num_base_steps: int,
    ideal_pop: float,
    epsilon: float,
    is_low: bool = True,  # Whether to generate low or high gerrys TODO cleanup
) -> tuple[Partition, Partition, Partition, Partition]:
    chain = create_chain(
        graph,
        num_steps=num_base_steps,
    )

    gerries_1, gerries_2 = [], []
    for i, part in enumerate(chain):
        seats, efficiency_gap = ELECTION_MAP["seats"](part), ELECTION_MAP[
            "efficiency_gap"
        ](part)

        if len(gerries_1) == 0 or seats == gerries_1[-1][0]:
            gerries_1.append((seats, efficiency_gap, part))
        elif (seats < gerries_1[-1][0] and is_low) or (
            seats > gerries_1[-1][0] and not is_low
        ):
            gerries_2 = gerries_1
            gerries_1 = [(seats, efficiency_gap, part)]
        elif len(gerries_2) == 0 or (
            (seats < gerries_2[-1][0] and is_low)
            or (seats > gerries_2[-1][0] and not is_low)
        ):
            gerries_2 = [(seats, efficiency_gap, part)]
        elif seats == gerries_2[-1][0]:
            gerries_2.append((seats, efficiency_gap, part))

        if i % 500 == 0:
            logger.info(
                f"{i} steps completed. 1: {len(gerries_1)}, 2: {len(gerries_2)}"
            )

    assert len(gerries_1) > 0
    assert len(gerries_2) > 0

    gerries_1.sort(key=lambda x: x[1])
    gerries_2.sort(key=lambda x: x[1])
    gerries_res = [
        gerries_1[0][0:2],
        gerries_1[-1][0:2],
        gerries_2[0][0:2],
        gerries_2[-1][0:2],
    ]
    logger.info(f"Generated gerries: {gerries_res}")

    # Save gerries_res
    with open(
        os.path.join(CONFIG["EXPERIMENT_NAME"], f"gerries_res_{is_low}.json"),
        "w",
    ) as f:
        json.dump(gerries_res, f)

    assert type(gerries_1[0][-1]) == Partition
    assert type(gerries_1[-1][-1]) == Partition
    assert type(gerries_2[0][-1]) == Partition
    assert type(gerries_2[-1][-1]) == Partition
    return (gerries_1[0][-1], gerries_1[-1][-1], gerries_2[0][-1], gerries_2[-1][-1])


def _generate_gerrys_large_memory(
    graph: gerrychain.Graph,
    num_parts: int,
    num_base_steps: int,
    ideal_pop: float,
    epsilon: float,
) -> tuple[Partition, Partition, Partition, Partition]:
    """Similar to _generate_gerrys, but uses more memory in order to use a more
    naive approach, allowing for a larger diversity of gerrys to easily be
    returned. If this doesn't crash, then this should always be used.
    """
    logging.info("To begin chain to generate gerrys")
    chain = create_chain(
        graph,
        num_steps=num_base_steps,
    )
    logging.info("Chain complete")

    parts = [part for part in chain]

    # convert partition to nested dict, gathered by and sorted by seats won
    nested_parts_dict = defaultdict(list)
    for part in parts:
        nested_parts_dict[ELECTION_MAP["seats"](part)].append(part)

    sorted_seats_won = sorted(nested_parts_dict.keys())
    nested_parts = [nested_parts_dict[seats] for seats in sorted_seats_won]

    # Sort each nested list by efficiency gap
    for nested_list in nested_parts:
        nested_list.sort(key=lambda x: ELECTION_MAP["efficiency_gap"](x))

    GERRYS_TO_STORE = [
        0,
        1,
        2,
        len(nested_parts) - 3,
        len(nested_parts) - 2,
        len(nested_parts) - 1,
    ]

    # For each desired seats won, return the greatest and least efficiency gap
    # This is mostly just a natural way to ensure that we get two very different
    # gerrymanders per seats won
    return tuple(nested_parts[i][j] for i in GERRYS_TO_STORE for j in (0, -1))


def _contract_around_assignment(
    graph: nx.Graph,
    assign: dict,  # type: ignore
    props: list[str],
    contractions_per_step: int,
    max_pop: int | float = float("inf"),
) -> list[nx.Graph]:
    num_districts = len(set(assign.values()))
    graphs = [graph]
    i = 0

    while len(graph) > num_districts:
        i += 1
        # We continue to contract around the original assignment. Even though
        # old nodes will still remain in assign, the assignment of any given
        # node never changes, so will still be valid.
        graph = random_condense(graph, assign=assign, props=props, max_pop=max_pop)
        if i % contractions_per_step == 0:
            graphs.append(graph)
            logger.info(f"Contractions {len(graphs)} to {len(graph)} nodes")

    return graphs


def generate_gerry_contractions(
    graph: gerrychain.Graph,
    num_parts: int,
    num_base_steps: int,
    ideal_pop: float,
    epsilon: float,
    props: list[str],
):
    # gerries = _generate_gerrys(
    #    graph,
    #    num_parts,
    #    num_base_steps,
    #    ideal_pop,
    #    epsilon,
    #    True,
    # ) + _generate_gerrys(
    #    graph,
    #    num_parts,
    #    num_base_steps,
    #    ideal_pop,
    #    epsilon,
    #    False,
    # )

    gerries = _generate_gerrys_large_memory(
        graph,
        num_parts,
        num_base_steps,
        ideal_pop,
        epsilon,
    )

    if CONFIG["CAP_MAX_POP"]:
        largest_pop = int(max(n[CONFIG["POP_COL"]] for n in graph.nodes.values()))  # type: ignore
    else:
        largest_pop = float("inf")

    graphs: list[list[nx.Graph]] = [
        _contract_around_assignment(
            graph,
            gerry.assignment.to_dict(),
            props,
            CONFIG["CONTRACTIONS_PER_STEP"],
            max_pop=largest_pop,
        )
        for gerry in gerries
    ]

    return graphs, gerries


def save_graphs(graphs: list[nx.Graph]):
    json_graphs = list(
        map(
            nx.readwrite.json_graph.adjacency_data,
            graphs,
        )
    )

    # Modifies in place
    # for g in json_graphs:
    #    for node in g["nodes"]:
    #        if "geometry" in node:
    #            node["geometry"] = shapely.to_geojson(node["geometry"])

    # Contracted edges have tuple keys, which can't be JSON serialized
    for g in json_graphs:
        for node in g["adjacency"]:
            for edge in node:
                if "contraction" in edge:
                    del edge["contraction"]

    with open(os.path.join(CONFIG["EXPERIMENT_NAME"], "graphs.json"), "w") as f:
        # TODO remove contraction from each graph['adjacency], since it uses tuple
        # keys so can't be JSON serialized
        json.dump(json_graphs, f)


def from_shp_file(path: str) -> gerrychain.Graph:
    df = gpd.read_file(path)

    return gerrychain.Graph.from_geodataframe(df, ignore_errors=True)


def update_experiment_name():
    """Updates experiement name to ensure that new runs don't overwrite old ones.

    Not reentrant! This will fail if you try running multiple experiments at once.
    """
    CONFIG["EXPERIMENT_NAME"] = os.path.join(
        os.path.dirname(__file__),
        "output",
        f'{CONFIG["EXPERIMENT_NAME"]}',
    )
    if os.path.exists(
        CONFIG["EXPERIMENT_NAME"],
    ):
        i = 1
        while os.path.exists(
            f'{CONFIG["EXPERIMENT_NAME"]}_({i})',
        ):
            i += 1

        CONFIG["EXPERIMENT_NAME"] = os.path.join(
            f'{CONFIG["EXPERIMENT_NAME"]}_({i})',
        )


def clean_graph(graph: gerrychain.Graph):
    """Cleans graph in-place, by:

    1. Removing islands
    """
    # Remove islands
    components = list(connected_components(graph))
    biggest_component = max(components, key=len)
    problem_components = [c for c in components if c != biggest_component]
    for component in problem_components:
        for node in component:
            graph.remove_node(node)


def main():
    if CONFIG["INPUT_GRAPH_FILE"].endswith(".shp"):
        graph = from_shp_file(
            os.path.join(
                os.path.dirname(__file__),
                os.path.join("input", CONFIG["INPUT_GRAPH_FILE"]),
            )
        )
    elif CONFIG["INPUT_GRAPH_FILE"].endswith(".json"):
        graph = gerrychain.Graph.from_json(
            os.path.join(
                os.path.dirname(__file__),
                os.path.join("input", CONFIG["INPUT_GRAPH_FILE"]),
            )
        )
    else:
        raise ValueError("Input file must be a shapefile")

    clean_graph(graph)

    if CONFIG.get("CONTRACTIONS_TYPE", "random") == "random":
        contractions = generate_random_contractions(
            graph,
            CONFIG["MIN_BLOCKS"],
            CONFIG["NUM_PARTS"],
            sum([graph.nodes[node][CONFIG["POP_COL"]] for node in graph.nodes()])
            / CONFIG["NUM_PARTS"],
            CONFIG["BEGIN_CONTRACT_EPSILON"],
            CONFIG["END_CONTRACT_EPSILON"],
            CONFIG["POP_COL"],
            [
                CONFIG["POP_COL"],
                CONFIG["PARTY_A_COL"],
                CONFIG["PARTY_B_COL"],
            ],
        )
        assigns = [None for _ in range(len(contractions))]

    elif CONFIG["CONTRACTIONS_TYPE"] == "gerry":
        nested_contractions, gerries = generate_gerry_contractions(
            graph,
            CONFIG["NUM_PARTS"],
            CONFIG["BASE_CHAIN_NUM_STEPS"],
            sum([graph.nodes[node][CONFIG["POP_COL"]] for node in graph.nodes()])
            / CONFIG["NUM_PARTS"],
            CONFIG["EPSILON"],
            [CONFIG["POP_COL"], CONFIG["PARTY_A_COL"], CONFIG["PARTY_B_COL"]],
        )
        # Save length of each list in contractions
        contraction_lens = [len(c) for c in nested_contractions]
        with open(
            os.path.join(
                CONFIG["EXPERIMENT_NAME"],
                "contraction_lens.json",
            ),
            "w",
        ) as f:
            json.dump(contraction_lens, f)

        # Flatten nested contractions
        contractions = [graph for contract in nested_contractions for graph in contract]

        # Assignmnents, where assigns[i] is the assignment for contractions[i]
        assigns = [
            gerry.assignment.to_dict()
            for gerry, list_of_contractions in zip(gerries, nested_contractions)
            for _ in list_of_contractions
        ]

    else:
        raise ValueError("Invalid contractions type")

    logger.info(f"Saving {len(contractions)} graphs")
    save_graphs(contractions)

    with mp.Pool(processes=CONFIG["NUM_PROCESSES"]) as pool:
        logger.info("Creating chains")

        raw_results = pool.starmap(
            run_chain,
            zip(
                (i for i in range(len(contractions))),
                contractions,
                (CONFIG["CHAIN_NUM_STEPS"] for _ in range(len(contractions))),
                tuple(assigns),
            ),
        )

        logger.info("Chains complete")

    with open(
        os.path.join(
            CONFIG["EXPERIMENT_NAME"],
            "raw_results.json",
        ),
        "w",
    ) as f:
        json.dump(raw_results, f)

    results = list(map(statistics.mean, list(raw_results)))

    logger.info(f"Results: {results}")

    with open(
        os.path.join(
            CONFIG["EXPERIMENT_NAME"],
            "results.json",
        ),
        "w",
    ) as f:
        json.dump(results, f)


def run_chain(
    run_id: int, graph: gerrychain.Graph, num_steps: int, assign: Optional[dict] = None
) -> list[float]:
    """Runs a chain on a single graph

    Args:
        run_id (int): ID of this run
        graph (gerrychain.Graph): Graph to run chain on
        num_steps (int): Number of steps to run chain for
        assign (Optional[dict], optional): Initial assignment. Defaults to None.
    """

    chain = create_chain(graph, num_steps=num_steps, assign=assign)
    stats = []
    try:
        for i, part in enumerate(chain):
            stat = ELECTION_MAP[CONFIG["ELECTION_STATISTIC"]](part)
            stats.append(stat)
            if i % 500 == 0:
                logger.info(f"{run_id}: {i} steps completed")
    except Exception as e:
        logger.error(f"Error in run {run_id}: {e}")

    return stats


def create_chain(
    graph: gerrychain.Graph,
    num_steps: Optional[int] = None,
    assign: Optional[dict] = None,
) -> MarkovChain:
    """Runs Recom on graph
    Args:
        graph: GerryChain.Graph object
        num_steps: number of steps in the chain. If None, uses value from CONFIG

    Returns:
        MarkovChain object
    """

    election = Election(
        CONFIG["ELECTION_NAME"],
        {
            "PartyA": CONFIG["PARTY_A_COL"],
            "PartyB": CONFIG["PARTY_B_COL"],
        },
    )
    updaters = {
        "population": Tally(CONFIG["POP_COL"]),
        # TODO helps with a seeming bug
        CONFIG["POP_COL"]: Tally(CONFIG["POP_COL"]),
        "cut_edges": cut_edges,
        CONFIG["ELECTION_NAME"]: election,
    }
    num_parts = CONFIG["NUM_PARTS"]
    ideal_pop = (
        sum([graph.nodes[node][CONFIG["POP_COL"]] for node in graph.nodes()])
        / num_parts
    )

    if not assign:
        assign = recursive_seed_part(
            graph,
            parts=range(num_parts),
            pop_target=ideal_pop,
            pop_col=CONFIG["POP_COL"],
            epsilon=CONFIG["EPSILON"],
            # Increase from default of 10k to 5m to avoid failure
            method=partial(bipartition_tree_random, max_attempts=5_000_000),
        )
    else:
        # In case graph was modified, ensure that assign keys only contains nodes
        # of graph, else Gerrychain throughs an error
        # in graph.nodes() is O(1) for networkx.Graph
        assign = {k: v for k, v in assign.items() if k in graph.nodes()}

    init_part = Partition(graph, assign, updaters=updaters)
    popbound = within_percent_of_ideal_population(init_part, CONFIG["EPSILON"])

    proposal = partial(
        recom,
        pop_col=CONFIG["POP_COL"],
        pop_target=sum([graph.nodes[node][CONFIG["POP_COL"]] for node in graph.nodes()])
        / len(init_part.parts),
        epsilon=CONFIG["EPSILON"],
    )

    return MarkovChain(
        proposal=proposal,
        constraints=Validator([popbound, contiguous]),
        accept=always_accept,
        initial_state=init_part,
        total_steps=num_steps,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Passed argument is config file if provided, else default to
    # random_contract_tx_config.yaml
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "random_contract_tx_config.toml"

    # Load config file
    with open(os.path.join(os.path.dirname(__file__), config_file), "r") as f:
        CONFIG = toml.load(f)

    update_experiment_name()

    os.makedirs(
        CONFIG["EXPERIMENT_NAME"],
        exist_ok=True,
    )

    # Write config to output directory
    with open(
        os.path.join(
            CONFIG["EXPERIMENT_NAME"],
            "config.toml",
        ),
        "w",
    ) as f:
        toml.dump(CONFIG, f)

    with open(
        os.path.join(
            CONFIG["EXPERIMENT_NAME"],
            "git_sha.txt",
        ),
        "w",
    ) as f:
        f.write(
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )

    main()
