from gerrychain import Graph
import numpy as np
import itertools

def load_graph(file, x_pos, y_pos):
    """Loads graph from json file, along with additional preprocessing steps.
    Returns:
        GerryChain.Graph object
    """
    graph = Graph.from_json(file)
    for node in graph.nodes():
        graph.nodes[node]['pos'] = np.array([graph.nodes[node][x_pos], graph.nodes[node][y_pos]])
    return graph


def is_straightline_embedding(file, x_pos, y_pos):
    """
    Check if an embedding is a straight line embedding.
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
    g = load_graph(file, x_pos, y_pos)
    count = 0
    for e1, e2 in itertools.combinations(g.edges(), 2):
        print(e1, e2, count)
        if _do_line_segments_intersect(g.nodes[e1[0]]['pos'], g.nodes[e1[1]]['pos'], g.nodes[e2[0]]['pos'], g.nodes[e2[1]]['pos']):
            count += 1
    return count

if __name__=='__main__':
    print(is_straightline_embedding('processed_data/NC.json', 'C_X', 'C_Y'))