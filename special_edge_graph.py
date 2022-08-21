import json
from networkx.readwrite import json_graph
import networkx as nx
import numpy as np
from typing import Tuple, List
from collections import defaultdict
from copy import deepcopy
import itertools
import logging
import traceback
import matplotlib.pyplot as plt

# For reproducibility, uses gerrychain's built-in random module
# to ensure a consistent seed.
from gerrychain.random import random

class SpecialEdgeGraph:
    """A wrapper around a base graph, which must be a straight-line embedding.
    Handles adding and removing additional special edges to the base graph while
    preserving the straight-line embedding. Furthermore, special edges can only
    be added to bounded faces, not the unbounded face.
    """

    class Face:
        """A face of a graph. Represented as a list of nodes. Constructed by
        traversing nodes clockwise, induced by a given directed edge."""

        def __init__(
            self,
            directed_edge: Tuple,
            graph_nodes: dict,
            root: int = None
        ) -> None:
            """Constructs a face from a directed edge.
            The face is constructed by repeatedly selecting neighbor of the
            current node which immediately follows the previous node, under the
            clockwise ordering induced by the current node. This will traverse
            a face in counter-clockwise order, unless the face is the unbounded
            face.

            Under this algorithm, every edge corresponds uniquely to a face.
            Namely, one of (up to) two faces the edge lies on. However, the
            converse does not hold- multiple directed edges correspond to the
            same face.

            Args:
                directed_edge (Tuple): Tuple of two node id representing a
                directed edge.
                graph_nodes (Dict): Dictionary mapping node id to node object.
                root (int): Hash of root face (face of original graph face
                lies is). If none, indicates that is itself a root face.
            """

            curr, prev = directed_edge[1], directed_edge[0]
            # Nodes of face. First, add the two nodes of the edge.
            face_nodes= [prev, curr]

            # Iterate around face. Each node has a rotation system which
            # maps from each of its neighbors to the next one in clockwise
            # order. This traverses a face single bounded face in counter-clockwise
            # order, or the unbounded face in clockwise order.
            # Draw a picture if the reason this works in unclear!
            # Iterate until curr and prev are equal to their starting values.
            # Note that the same node can appear multiple times in the face,
            # so it would be insufficient to just check if curr is equal to its
            # starting value.
            while True:
                succ = graph_nodes[curr]['rotation'][prev]
                face_nodes.append(succ)
                prev = curr
                curr = succ
                if curr == directed_edge[1] and prev == directed_edge[0]:
                    break

            # Remove the last two nodes, which are repeats of the two starting
            # nodes.
            self.nodes = face_nodes[:-2]
            # Save map from node ids to nodes. TODO it's kinda ugly to save the
            # entire node map in each face. It's also the simplest solution
            # and works just fine.
            self._graph_nodes = graph_nodes
            # If root is none, set root to hash of self. Else to provided root.
            self.root = root if root is not None else hash(self)
            # Don't compute orientation immediately. Instead, compute it lazily
            self._is_unbounded = None
            logging.debug(f"Face {hash(self)} with root {self.root} constructed. {self.nodes}")

        def __len__(self) -> int:
            """Returns the number of nodes in the face."""
            return len(self.nodes)

        def __hash__(self) -> int:
            # Hash is determined solely by the set of nodes in the face and
            # the number of nodes, not their order. This way, the same face
            # that was traversed with a different starting node will still
            # have the same hash. Note that it is possible for nodes to
            # be duplicated when traversing a face, and that it is possible
            # for two faces to have the same set of nodes, but different
            # number of nodes, and therefore be different faces (consider
            # the interior vs exterior faces of the triangle_plus_edge
            # test case). Therefore, we also multiple the hash by the number
            # of nodes, to distinguish between these faces.
            return hash((frozenset(self.nodes), len(self.nodes)))

        def __eq__(self, __o: object) -> bool:

            if not isinstance(__o, SpecialEdgeGraph.Face):
                return False
            return frozenset(self.nodes) == frozenset(__o.nodes) \
                and len(self.nodes) == len(__o.nodes)

        def __iter__(self):
            """Iterates over the nodes in the face."""
            return iter(self.nodes)

        def get_pred_and_succ(self, node: int) -> Tuple[int, int]:
            """Returns the predecessor and successor of a node in the face.
            Args:
                node (int): Node id.
            Returns:
                Tuple[int, int]: Tuple of predecessor and successor node id.
            """
            ind = self.nodes.index(node)
            return self._graph_nodes[self.nodes[ind - 1]], \
                self._graph_nodes[self.nodes[(ind + 1) % len(self.nodes)]]

        @property
        def is_unbounded(self) -> bool:
            """Returns True if the face is the unbounded face. This is true if
            and only if the orientation of the face is clockwise."""
            if self._is_unbounded is None:
                # The orientation of the face is determined by the orientation of
                # any two edges which meet at a point of the face's convex hull.
                # Furthermore, the vertex with the smallest x-coordinate
                # (and smallest y-coordinate if x-coordinates are
                # equal) is the guaranteed to be in the convex hull.
                # See https://en.wikipedia.org/wiki/Curve_orientation#Orientation_of_a_simple_polygon
                # for discussion
                # Note that we do not have to worry about the indeterminate case
                # (vectors are colinear) because we choose the vertex with the
                # smallest x-coordinate, making it impossible for the two vectors
                # to be colinear.
                # Additionally, note that the above discussion only applies to simple
                # polygons, while the faces of straight line embeddings are slight
                # generalizations, as traversing the edges of a face may entail
                # crossing the same edge multiple times. However, the unbounded
                # face is guaranteed to be (the exterior of) a simple polygon, so
                # this algorithm still holds in this case. In the case where we end
                # up comparing the orientation of the same edge against itself, then
                # the orientation is not clockwise, and therefore once again we
                # get the correct answer.
                min_vertex_id, min_vertex = min(enumerate(self.nodes),
                    key=lambda v: (self._graph_nodes[v[1]]['x'], self._graph_nodes[v[1]]['y'])
                )
                # Three vertices which form the two vectors (b is a shared vertex of the two).
                a, b, c = self._graph_nodes[self.nodes[min_vertex_id - 1]], \
                    self._graph_nodes[min_vertex], \
                    self._graph_nodes[self.nodes[(min_vertex_id + 1) % len(self.nodes)]]
                self._is_unbounded = self.is_clockwise(a,b,c)
                logging.debug(f'Face {hash(self)} is unbounded: {self._is_unbounded}')
            return self._is_unbounded

        @staticmethod
        def is_clockwise(a: dict, b: dict, c: dict) -> bool:
            """Given three points, A, B, C in 2D space, returns if the vector
            AC is oriented in a clockwise direction relative to AB.

            Args:
                a, b, c (dict): Three points in 2D space. Must have the keys
                'x' and 'y'.

            Returns:
                bool: True if AC is oriented clockwise relative to AB.
            """
            # By considering the 2D vectors AB and AC, as vectors in 3D space,
            # their relative orientation can be determined by their cross-product.
            # Specifically, by the sign of z-component of the cross-product.
            # A positive sign indicates that AC is oriented counter-clockwise
            # relative to AB, and a negative sign indicates clockwise.
            return  (b['x'] - a['x']) * (c['y'] - \
                    b['y']) - (b['y'] - \
                    a['y']) * (c['x'] - b['x']) < 0

    def __init__(self, graph: nx.Graph) -> None:
        # Deep copy graph to ensure multiple SpecialEdgeGraphs can freely
        # modify their graph. There's an obvious performance downside here,
        # but the expectation is that constructing the graph should be trivial
        # time-wise compared to the rest of the algorithm.
        self.graph = deepcopy(graph)
        # Special edges maps from hash of root face to special edges
        self.special_edges = defaultdict(list)
        # Dual graph- graph of faces, with edges representing shared edges
        self.dual = self._construct_restricted_dual()
        assert nx.is_connected(self.graph)
        assert nx.is_connected(self.dual)

    def __add__(self, other):
        """Adds two SpecialEdgeGraphs together. Returns new graphs. For root
        faces where exactly one of the two graphs has special edges, the
        special edges are added to the new graph. For root faces where both
        graphs have special edges, the special edges of one of the two graphs
        is added (randomly chosen).

        Args:
            other (SpecialEdgeGraph): Another SpecialEdgeGraph. Must share the
            same base graph as self.

        Returns:
            SpecialEdgeGraph: New SpecialEdgeGraph with special edges added.
        """
        # Reconstruct the base graph:
        base_graph = nx.Graph()
        base_graph.add_nodes_from(self.graph.nodes(data=True))
        for e in self.graph.edges(data=True):
            # Only add non-special edges.
            if not e[2].get('special', False):
                base_graph.add_edge(e[0], e[1])
                nx.set_edge_attributes(base_graph, {(e[0], e[1]): e[2]})

        # Re-add special edges. Only add special edges for a given root face from
        # exactly one of the two graphs.
        for root_face in \
        set(self.special_edges.keys()).union(set(other.special_edges.keys())):
            if self.special_edges[root_face] or other.special_edges[root_face]:
                if self.special_edges[root_face] and other.special_edges[root_face]:
                    # Add the special edges of one of the two graphs.
                    g = random.choice([self, other])
                    edges = [
                        (*tuple(e), g.graph.get_edge_data(*tuple(e))) \
                        for e in g.special_edges[root_face]
                    ]
                    base_graph.add_edges_from(edges)
                elif self.special_edges[root_face]:
                    edges = [
                        (*tuple(e), self.graph.get_edge_data(*tuple(e))) \
                        for e in self.special_edges[root_face]
                    ]
                    base_graph.add_edges_from(edges)
                else:
                    edges = [
                        (*tuple(e), other.graph.get_edge_data(*tuple(e))) \
                        for e in other.special_edges[root_face]
                    ]
                    base_graph.add_edges_from(edges)

        return SpecialEdgeGraph(base_graph)

    def __eq__(self, other) -> bool:
        """Returns True if two SpecialEdgeGraphs are equal. Two SpecialEdgeGraphs
        are equal if they have the same set of faces. Assuming the graphs are
        well-formed, this implies they must share the same base graph, special
        edges, and edges of the dual graph, so we don't need to check those.

        Returns:
            bool: True if the two SpecialEdgeGraphs are equal.
        """
        return nx.utils.nodes_equal(self.dual, other.dual)

    def __hash__(self) -> int:
        return hash((self.graph, self.dual))

    @property
    def num_special_edges(self) -> int:
        """Returns the number of special edges in the graph.
        """
        return sum(len(v) for v in self.special_edges.values())

    def _set_all_rotation_systems(self) -> None:
        """Sets the rotation system for each node of the graph.
        The "rotation system" a node is the clockwise ordering of its neighbors.
        It is represented as a dictionary where the node id of each neighbor
        maps to the node id of the next neighbor in the rotation system.

        Assumes each node has x and y attributes, and that
        the graph is a straight-line embedding.
        """

        # First set position attribute for each node. Use numpy array for simple
        # vector addition.
        for v in self.graph.nodes():
            self.graph.nodes[v]['pos'] = np.array([
                self.graph.nodes[v]['x'],
                self.graph.nodes[v]['y']
            ])

        # For each node, set the rotation system.
        for v in self.graph.nodes():
            self._set_rotation_system(v)

    def _set_rotation_system(self, node: int) -> None:
        """Sets the rotation system for a node.

        Args:
            node (int): Node id of the node to set the rotation system for.
        """
        neighbor_list = list(self.graph.neighbors(node))
        # Position of neighbors relative to v.
        locations = (self.graph.nodes[w]['pos'] -
            self.graph.nodes[node]['pos'] for w in neighbor_list)
        # Angle of neighbors relative to v.
        angles = [float(np.arctan2(x[0], x[1])) for x in locations]
        # Sort list of neighbors by angle.
        neighbor_list.sort(key=dict(zip(neighbor_list, angles)).get)
        # Rotation system points each neighbor to the next one clockwise,
        # wrapping around the list.
        rotation_system = {w: neighbor_list[(i+1) % len(neighbor_list)]
            for i, w in enumerate(neighbor_list)}
        self.graph.nodes[node]["rotation"] = rotation_system

    def _construct_restricted_dual(self) -> nx.Graph():
        """Constructs the restricted dual graph of the graph. The restricted
        dual is the graph of all unbounded faces of the graph, with edges
        representing the existence of a shared edge in the original graph.

        Returns:
            nx.Graph: Restricted dual graph of the graph.
        """
        # Sets rotation system for each node for graph.
        self._set_all_rotation_systems()

        self.dual = nx.Graph()

        # Add the face corresponding to each direct edge. Note that the __hash__
        # and __eq__ ensure that duplicate faces are not added.
        for e in self.graph.edges():
            faces = []
            # Add face corresponding to both e and its reverse. Do not add
            # unbounded face.
            # If edge has root face set, use that to construct face. Else,
            # root hash is set to None, and set e's root to the new face.
            for directed_edge in (e, e[::-1]):
                face = self.Face(
                    directed_edge,
                    self.graph.nodes,
                    # Special edges map from directed edge to the corresponding
                    # root face (which, for special edges, should be the same
                    # in both directions, since the corresponding special faces
                    # must be subfaces of the same root face). Use the root face
                    # of the edge if it is set, otherwise set to None, which
                    # indicates the newly constructed Face is itself a root face.
                    root=self.graph.edges[e].get(directed_edge, None)
                )
                if not face.is_unbounded:
                    self.dual.add_node(face)
                    # Map from directed edge to its corresponding root face
                    self.graph.edges[e][directed_edge] = face.root
                    faces.append(face)

            # If edge is already labelled as special, add it to special edges.
            if self.graph.edges[e].get('special', False):
                self.special_edges[self.graph.edges[e][e]].append(frozenset(e))

            # Add edge between between faces in the dual graph
            if len(faces) == 2:
                self._add_shared_edge(faces[0], faces[1], e)

        return self.dual

    def _add_shared_edge(self, face1: Face, face2: Face, edge: tuple) -> None:
        """Adds a shared edge between two faces in the dual graph.

        Args:
            face1 (Face): First face in the shared edge.
            face2 (Face): Second face in the shared edge.
            edge (tuple): Directed edge between the two faces.
        """
        if not self.dual.has_edge(face1, face2):
            self.dual.add_edge(face1, face2, shared=[frozenset(edge)])
        else:
            self.dual.edges[face1, face2]['shared'].append(frozenset(edge))


    def _add_special_edge(self, face: Face, node1: int, node2: int) -> bool:
        """Adds special edge to the graph if is valid, else does nothing.
        Returns True if the edge was added, False otherwise.

        Args:
            face (Face): Face of the graph to which the special edge belongs.
            node1 (int): Node id of first node of the edge.
            node2 (int): Node id of second node of the edge.

        Returns:
            bool: True if the edge was added, False otherwise.
        """
        if self._is_edge_valid(face, node1, node2):
            logging.debug(f'Adding special edge {node1} {node2}')
            self.graph.add_edge(node1, node2, special=True)
            self.special_edges[face.root].append(frozenset((node1, node2)))
            # Update rotation systems of the two nodes
            self._set_rotation_system(node1)
            self._set_rotation_system(node2)

            # New faces to add
            f1 = self.Face((node1, node2), self.graph.nodes, root=face.root)
            f2 = self.Face((node2, node1), self.graph.nodes, root=face.root)

            # Add faces to dual graph
            self.dual.add_node(f1)
            self.dual.add_node(f2)

            # Transfer the edges (in the dual graph) of the parent face
            # to the appropriate child. Convert to list to avoid modifying
            # the EdgeView object while iterating over it
            edges = list(self.dual.edges(face, data=True))
            for e in edges:
                for shared_edge in e[2]['shared']:
                    for f in (f1, f2):
                        if all(n in f.nodes for n in shared_edge):
                            other_face = e[0] if e[0] != face else e[1]
                            self._add_shared_edge(f, other_face, shared_edge)

            # Remove parent face and replace with the two new faces, with
            # an edge connecting them
            self.dual.remove_node(face)
            self.dual.add_edge(f1, f2, shared=[frozenset((node1, node2))])

            # New edge must be internal to some root face. Map both directions
            # of new edge to that root face.
            self.graph.edges[node1, node2].update(
                {(node1, node2): face.root, (node2, node1): face.root}
            )
            return True
        return False

    def _remove_special_edge(self, node1: int, node2: int) -> bool:
        """Removes special edge from the graph if it exists, else does nothing.
        Returns True if the edge was removed, False otherwise.

        Args:
            node1 (int): Node id of first node of the edge.
            node2 (int): Node id of second node of the edge.
        """
        if self.graph.has_edge(node1, node2):
            if self.graph.edges[node1, node2].get('special', False):
                logging.debug(f'Removing special edge {node1} {node2}')
                # Remove the two faces and replace with the "parent face"
                # To do so, create two dummy Faces, corresponding to the two
                # directions of the edge, and then remove them from the set of
                # special faces. Note that the __hash__ and __eq__ are overridden
                # so that these can be treated as the faces present in the dual
                # graph.
                dummy_f1 = self.Face((node1, node2), self.graph.nodes)
                dummy_f2 = self.Face((node2, node1), self.graph.nodes)

                # Remove the special edge from the graph
                root = self.graph.edges[node1, node2][(node1, node2)]
                self.graph.remove_edge(node1, node2)
                # Update rotation systems of the two nodes
                self._set_rotation_system(node1)
                self._set_rotation_system(node2)
                # Index 0 and 1 of the nodes of either dummy face are the two
                # nodes of the edge that was removed. Index 1 and 2 are the
                # next pair when traversing the face. Now that the special
                # is removed, they induce the parent face. Recreate this face
                # and add it to the set of faces.
                face = SpecialEdgeGraph.Face(
                    tuple((dummy_f1.nodes + [dummy_f1.nodes[0]])[2:4]),
                    self.graph.nodes,
                    root=root,
                )
                self.dual.add_node(face)
                # Transfer edges in dual graph of the two child faces to the
                # new parent face.  Convert to list to avoid modifying
                # the EdgeView object while iterating over it
                edges = list(self.dual.edges([dummy_f1, dummy_f2], data=True))
                for e in edges:
                    other_node = None
                    if e[0] != dummy_f1 and e[0] != dummy_f2:
                        other_node = e[0]
                    elif e[1] != dummy_f1 and e[1] != dummy_f2:
                        other_node = e[1]
                    if other_node is not None:
                        for shared_edge in e[2]['shared']:
                            self._add_shared_edge(
                                face, other_node, tuple(shared_edge)
                            )

                # Remove the two child faces (using the dummy faces)
                self.dual.remove_node(dummy_f1)
                if dummy_f1 != dummy_f2:
                    self.dual.remove_node(dummy_f2)

                # Remove from dict of special edges
                self.special_edges[root].remove(frozenset((node1, node2)))
                return True
        return False


    def _is_edge_valid(self, face: Face, node1: int, node2: int) -> bool:
        """Determines if an edge between two nodes preserves the straight-line
        embedding. Assumes the nodes lie on face.

        Args:
            face (Face): Face containing the nodes.
            node1, node2 (int): Nodes ids to check if valid.
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
                # If they share at least one vertex, then they intersect iff
                # the line segments coming out of the shared vertex are colinear
                # (this handles the case where the edges are identical, or
                # otherwise overlap).
                # np.array's elementwise equality makes the logic for this
                # more difficult than it needs to be, sorry...
                shared_vertex, v0, v1 = None, None, None
                # Iterate over (a,b) and (b,a):
                for p, not_p in itertools.permutations([a,b], 2):
                    if all(p == c):
                        shared_vertex = p
                        v0 = not_p
                        v1 = d
                    elif all(p == d):
                        shared_vertex = p
                        v0 = not_p
                        v1 = c
                assert shared_vertex is not None
                # Check if angles are equal
                return np.arctan2(v1[1] - shared_vertex[1], v1[0] - shared_vertex[0]) == \
                    np.arctan2(v0[1] - shared_vertex[1], v0[0] - shared_vertex[0])

            return _is_cc(a, c, d) != _is_cc(b, c, d) and _is_cc(a, b, c) != _is_cc(a, b, d)

        def _is_interior(a: np.array, b: np.array, c: np.array, d: np.array) -> bool:
            """Determines if edge BD goes through the interior of the face, given
            edges AB and BC.
            Args:
                a (np.array): Predecessor of B
                b (np.array): One of nodes edge is added to. We check if BD
                goes through the interior of the face relative to B
                c (np.array): Successor of B
                d (np.array): One of nodes edge is added to. Note that AD
                may not go through the interior of the face relative to d.
            """
            def _relative_angle(theta1: float, theta2: float) -> float:
                """Determines the angle of theta2 relative to theta1."""
                # First translate from [-pi, pi] to [0, 2pi]
                if theta1 < 0:
                    theta1 += 2 * np.pi
                if theta2 < 0:
                    theta2 += 2 * np.pi
                if theta1 >= theta2:
                    return theta1 - theta2
                return 2 * np.pi - (theta2 - theta1)
            # Note that because the edges traverse the face in counter-clockwise
            # direction, the interior is to the left of the edge (or to the right
            # of the reverse edge, BA). That is, the interior points next to AB
            # are more clockwise relative to B than A. Therefore, BD is interior
            # if its angle relative to B is clockwise relative to BA, but less
            # so than BC.
            # Sorry if that comment doesn't help... I recommend drawing a picture
            # of a simple polygon. Imagine the edges traversing the face in a
            # counter-clockwise direction, and note how, looking at a single directed,
            # edge, we can tell which side is interior.
            theta_a, theta_c, theta_d = [np.arctan2(x[1] - b[1], x[0] - b[0]) for x in (a, c, d)]
            return _relative_angle(theta_a, theta_c) > _relative_angle(theta_a, theta_d)

        logging.debug(f'Checking if edge {node1} {node2} on face {hash(face)} is valid')
        graph_node1, graph_node2 = self.graph.nodes[node1], self.graph.nodes[node2]
        # Add an edge between two nodes that lie on the same face preserves the
        # straight-line embedding if and only if the edge between them,
        # represented by a line segment, does not intersect any other edge
        # of the face and goes through the interior of the face.
        # First check if edge goes through interior of face.
        pred, succ = face.get_pred_and_succ(node1)
        if not _is_interior(pred['pos'], graph_node1['pos'], succ['pos'], graph_node2['pos']):
            logging.debug(f'Edge {node1} {node2} does not go through interior of face')
            return False

        # Check each edge of the face- return False if edge intersects any edge,
        # else return True. Each edge corresponds to a consecutive pair of
        # nodes in the face, which we iterator over:
        for v1, v2 in zip(face.nodes, face.nodes[1:] + [face.nodes[0]]):
            if _do_line_segments_intersect(
                graph_node1['pos'],
                graph_node2['pos'],
                self.graph.nodes[v1]['pos'],
                self.graph.nodes[v2]['pos']
            ):
                logging.debug('Failed because edge intersects another edge of face')
                return False
        return True

    def add_random_edge(self, face) -> Tuple[int, int]:
        """Attempts to add a random edge to the face. Only adds edge if it
        preserves the straight-line embedding.

        Args:
            face (Face): Face to add edge to.

        Returns:
            Tuple[int, int]: Nodes of the edge added. None if no edge was added.
        """
        # All pairs of nodes
        pairs = list(itertools.combinations(face.nodes, 2))
        # Randomly select a pair of nodes and attempt to add edge between them.
        # If edge is invalid, remove pair and randomly select another pair.
        # Continue until a valid edge is found, or all pairs are exhausted.
        while pairs:
            pair = random.choice(pairs)
            if self._add_special_edge(face, pair[0], pair[1]):
                return (pair[0], pair[1])
            pairs.remove(pair)
        # TODO cache that face cannot have edge added to it
        return None


    def add_random_special_edges(self, n: int) -> List:
        """Adds upto n random special edges to the graph. Special edges connect
        two nodes which were share a common face, and preserve the straight-line
        embedding.

        This only attempts to add n edges, because it is not guaranteed that such
        n edges exist, or if they do exist they may be expensive to find. So,
        this function attempts to add n edges, and then it is the responsibility
        of the caller to determine if it should try again.

        Args:
            n (int): Number of special edges to add attempt to add

        Returns:
            List: List of edges added. First element is the edge, second is the face.
        """
        # It is only possible to add an edge to a face with at least 4 nodes.
        candidate_faces = [f for f in self.dual if len(f) >= 4]
        special_faces = random.sample(
            candidate_faces, max(min(n, len(candidate_faces)), 0)
        )
        # Attempts to add an edge to each face. Returns list of edges added.
        return list(filter(lambda x: x[0] is not None, \
            ((self.add_random_edge(f), f) for f in special_faces))
        )

    def remove_random_special_edges(self, n: int) -> List:
        """Removes n random special edges (or all special edges, if it is less
        than n)

        Args:
            n (int): Number of random special edges to remove.
        """
        # All special edges
        edges = [edge for face_edges in self.special_edges.values() for edge in face_edges]
        to_remove = random.sample(edges, max(min(n, len(edges)), 0))
        for edge in to_remove:
            self._remove_special_edge(*tuple(edge))
        return to_remove

    def is_original_subgraph_connected(self, subgraph_nodes: List[int]) -> bool:
        """Determines if the original subgraph (excluding special edges)
        is connected.
        """
        def _bfs(start: int, subgraph_nodes: set) -> set[int]:
            """Performs a breadth-first search from start. Only traverse
            non-special edges.

            Args:
                start (int): Node to start search from.
                subgraph_nodes (set): Set of nodes to include in search

            Returns:
                set[int]: Set of nodes in the search.
            """
            queue = [start]
            visited = set()
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                for n in self.graph.neighbors(node):
                    # Only traverse non-special edges and only go to nodes in
                    # subgraph_nodes.
                    if n in subgraph_nodes and n not in visited and \
                        not self.graph.edges()[node, n].get('special', False):
                        queue.append(n)
            return visited
        logging.debug(f'Checking if original subgraph {subgraph_nodes} is connected')
        return len(_bfs(subgraph_nodes[0], set(subgraph_nodes))) == len(subgraph_nodes)

    def save(self, filename: str):
        """Saves the graph to a JSON file. Filters out attributes that are not
        JSON serializable.
        """
        g = json_graph.adjacency_data(self.graph)
        for n in g['nodes']:
            del n['pos']
        for adj in g['adjacency']:
            for i, e in enumerate(adj):
                adj[i] = {k: v for k, v in e.items() if type(k) != tuple}
        with open(filename, 'w') as f:
            f.write(json.dumps(g, indent=2))

