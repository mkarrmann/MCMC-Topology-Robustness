import argparse
import os
import json
import itertools
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
from collections import Counter
from gerrychain import Graph

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="File to read from", type=str)
# TODO you'll need to change these depending on the graph you're working with:
X_POS = 'C_X'
Y_POS = 'C_Y'

def load_graph(file):
    """Loads graph
    """
    if file.split('.')[-1] != 'json':
        with open(file) as f:
            file = file.split('.')[0] + '.json'
            Graph.from_file(f).to_json(file)
    with open(file) as f:
        data = json.load(f)
    graph = json_graph.adjacency_graph(data)
    print(f'Node count: {len(graph.nodes())}')
    for node in graph.nodes():
        graph.nodes[node]['pos2'] = np.array([graph.nodes[node]['pos']['coordinates'][0], graph.nodes[node]['pos']['coordinates'][1]])
    return graph

def remove_straight_line_violations(g):
    """
    Gets the list of edges that cross each other when treated as a straight line
    embedding.
    """
    def _do_line_segments_intersect(a: np.array, b: np.array, c: np.array, d: np.array) -> bool:
        """Determines if the line segments AB and CD intersect.

        We do not consider a single shared vertex (e.g. B=C) as an intersection.

        Credit: https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

        Args:
            a, b (np.array): Endpoints of line segment AB.
            c, d (np.array): Endpoints of line segment CD.

        Returns:
            bool: Whether the line segments intersect.
        """
        def _is_cc(a: np.array, b: np.array, c: np.array) -> bool:
            """Determines if the points A, B, and C are in counter-clockwise
            order.
            """
            # Compares slopes of line segments AB and AC.
            # TODO could be replaced with cross product.
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        # First check if the lines segments share a vertex.
        # Note: == between np.array returns array of elementwise equality.
        if all(a == c) or all(a == d) or all(b == c) or all(b == d):
            # Return False if they share exactly one vertex, True is they
            # share two vertices (i.e. are the same line segment).
            return (all(a == c) and all(b == d)) or (all(a == d) and all(b == c))

        return _is_cc(a, c, d) != _is_cc(b, c, d) and _is_cc(a, b, c) != _is_cc(a, b, d)
    violations = []
    # O(|E|^2) algorithm :( Will take on order of 15 minutes. Probably not possible
    # to meaningfully optimize this.
    for e1, e2 in itertools.combinations(g.edges(), 2):
        if _do_line_segments_intersect(g.nodes[e1[0]]['pos2'], g.nodes[e1[1]]['pos2'], g.nodes[e2[0]]['pos2'], g.nodes[e2[1]]['pos2']):
            violations.append((e1, e2))
        print(e1, e2, len(violations))

    print(violations)
    # Greedily remove the edge with the most violations.
    # Note that this isn't guaranteed to remove the least number of edges,
    # but the optimal algorithm is very difficult and this is good enough.
    num_removed = 0
    while violations:
        count = Counter(edge for intersect in violations for edge in intersect)
        most_common = count.most_common(1)[0][0]
        g.remove_edge(*most_common)
        violations = [intersect for intersect in violations if most_common not in intersect]
        print(most_common, len(violations))
        num_removed += 1
    print(f"Removed {num_removed} edges.")

    return g

def save_graph(g, file):
    """Saves graph to file.
    """
    file = os.path.join("processed_data", os.path.basename(file))
    # Pos is not JSON serializable, so we remove it.
    for node in g.nodes():
        del g.nodes[node]['pos2']
    with open(file, 'w') as f:
        f.write(json.dumps(json_graph.adjacency_data(g), indent=2))

if __name__=='__main__':
    args = parser.parse_args()
    g = load_graph(args.file)
    g = remove_straight_line_violations(g)
    save_graph(g, args.file)