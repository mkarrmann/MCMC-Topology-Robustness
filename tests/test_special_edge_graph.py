import networkx as nx
from special_edge_graph import SpecialEdgeGraph
from copy import deepcopy

# Note: Tests are technically random. The advantage of this is that fewer
# tests cases cover a larger number of cases. The (large) disadvantage is that
# the tests need to be run multiple times to confirm a true "pass".
# Honestly there's a few reasons that these are bad tests, but there's good
# coverage and they get the job done.

def test_triangle_graph():
    """Tests the SpecialEdgeGraph class on a triangle graph with nodes
    at (0,0), (0,1), (1,0)."""
    # Create Triangle Graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2)])
    G.nodes[0]['x'] = 0
    G.nodes[0]['y'] = 0
    G.nodes[1]['x'] = 1
    G.nodes[1]['y'] = 0
    G.nodes[2]['x'] = 0
    G.nodes[2]['y'] = 1
    graph = SpecialEdgeGraph(G)
    assert len(graph.dual) == 1
    # There are no faces to add edges to, should return 0
    assert len(graph.add_random_special_edges(n=1)) == 0
    # There is no special edge to remove, should return 0
    assert len(graph.remove_random_special_edges(n=1)) == 0
    # Check that is_original_subgraph_connected returns True, since graph
    # is connected
    assert graph.is_original_subgraph_connected(list(graph.graph.nodes().keys()))
    # Check dual (with single node) is connected
    assert nx.is_connected(graph.dual)
    # Add graph to identical copy of itself. Result should be equal to original
    # because there are no special edges.
    assert graph + SpecialEdgeGraph(G) == graph

def test_two_triangle_graph():
    """Tests the SpecialEdgeGraph class on a two-triangle graph with nodes
    at (0,0), (0,1), (1,0), (1,1)."""
    # Create Two Triangle Graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    G.nodes[0]['x'] = 0
    G.nodes[0]['y'] = 0
    G.nodes[1]['x'] = 1
    G.nodes[1]['y'] = 0
    G.nodes[2]['x'] = 0
    G.nodes[2]['y'] = 1
    G.nodes[3]['x'] = 1
    G.nodes[3]['y'] = 1
    graph = SpecialEdgeGraph(G)
    assert len(graph.dual) == 2
    # There are no faces to add edges to, should return 0
    assert len(graph.add_random_special_edges(n=1)) == 0
    # There is no special edge to remove, should return 0
    assert len(graph.remove_random_special_edges(n=1)) == 0
    # Check that is_original_subgraph_connected returns True, since graph
    # is connected
    assert graph.is_original_subgraph_connected(list(graph.graph.nodes().keys()))
    # Check dual is connected
    assert nx.is_connected(graph.dual)
    # Add graph to identical copy of itself. Result should be equal to original
    # because there are no special edges.
    assert graph + SpecialEdgeGraph(G) == graph

def test_square_graph():
    """Tests the SpecialEdgeGraph class on a square graph with nodes at
    (0,0), (0,1), (1,0), (1,1)
    """
    # Create Square Graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    G.nodes[0]['x'] = 0
    G.nodes[0]['y'] = 0
    G.nodes[1]['x'] = 1
    G.nodes[1]['y'] = 0
    G.nodes[2]['x'] = 0
    G.nodes[2]['y'] = 1
    G.nodes[3]['x'] = 1
    G.nodes[3]['y'] = 1
    graph = SpecialEdgeGraph(G)
    assert len(graph.dual) == 1
    square_face = list(graph.dual)[0]
    # There is exactly one face to add edges to, should return 1
    assert len(graph.add_random_special_edges(n=1)) == 1
    # Check that face was increased to 2
    assert len(graph.dual) == 2
    # Check that both faces are new (not equal to the original face)
    assert all([f != square_face for f in graph.dual])
    # There are no faces to add edges to, should return 0
    assert len(graph.add_random_special_edges(n=1)) == 0
    # Assert that exactly one special edge was added
    assert len([e for e in graph.graph.edges.data() if e[2].get('special', False)]) == 1
    len(graph.special_edges[hash(square_face)]) == 1
    # Check that dual is connected
    assert nx.is_connected(graph.dual)
    # Adding graph to itself should return itself
    assert graph + graph == graph
    # Adding graph to base graph should return graph with special edges
    assert graph + SpecialEdgeGraph(G) == graph
    assert SpecialEdgeGraph(G) + graph == graph
    # Remove special edge
    assert len(graph.remove_random_special_edges(n=1)) == 1
    # Removing special edge return face to the original one
    assert len(graph.dual) == 1 and tuple(graph.dual)[0] == square_face
    # Graph is now equal to base graph
    assert graph == SpecialEdgeGraph(G)
    # Adding graph to itself should return other copy of base graph
    assert graph + graph == SpecialEdgeGraph(G)
    # Check that is_original_subgraph_connected returns True, since graph
    # is connected
    assert graph.is_original_subgraph_connected(list(graph.graph.nodes().keys()))

    # Create two graphs with two different possible special edges
    g1, g2 = SpecialEdgeGraph(G), SpecialEdgeGraph(G)
    assert g1._add_special_edge(square_face, 1, 2)
    assert g2._add_special_edge(square_face, 0, 3)
    # Graphs should not be equal
    assert g1 != g2
    # Adding graphs together should be equal to exactly one of the graphs
    g3 = g1 + g2
    assert (g3 == g1) != (g3 == g2)

def test_non_convex_quadrilateral():
    """Tests the SpecialEdgeGraph class on a non-convex quadrilateral graph
    with nodes at (0,0), (0,1), (-2,2), (2,2)."""
    # Create Non-Convex Quadrilateral Graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    G.nodes[0]['x'] = 0
    G.nodes[0]['y'] = 0
    G.nodes[1]['x'] = 2
    G.nodes[1]['y'] = 2
    G.nodes[2]['x'] = -2
    G.nodes[2]['y'] = 2
    G.nodes[3]['x'] = 0
    G.nodes[3]['y'] = 1
    graph = SpecialEdgeGraph(G)
    assert len(graph.dual) == 1
    original_face = list(graph.dual)[0]
    # Should add a special edge to face
    assert len(graph.add_random_special_edges(n=1)) == 1
    # Should have doubled the number of faces
    assert len(graph.dual) == 2
    # Check that dual is connected
    assert nx.is_connected(graph.dual)
    # Check that both faces are new (not equal to the original face)
    assert all([f != original_face for f in graph.dual])
    # There are no faces to add edges to, should return 0
    assert len(graph.add_random_special_edges(n=1)) == 0
    # Assert that exactly one special edge was added
    assert len([e for e in graph.graph.edges.data() if e[2].get('special', False)]) == 1
    assert len(graph.special_edges[hash(original_face)]) == 1
    # Adding graph to itself should return itself
    assert graph + graph == graph
    # Adding graph to base graph should return graph with special edges
    assert graph + SpecialEdgeGraph(G) == graph
    assert SpecialEdgeGraph(G) + graph == graph
    # Remove special edge
    assert len(graph.remove_random_special_edges(n=1)) == 1
    # Removing special edge return face to the original one
    assert len(graph.dual) == 1 and tuple(graph.dual)[0] == original_face
    # Graph is now equal to base graph
    assert graph == SpecialEdgeGraph(G)
    # Adding graph to itself should return other copy of base graph
    assert graph + graph == SpecialEdgeGraph(G)
    # Check that is_original_subgraph_connected returns True, since graph
    # is connected
    assert graph.is_original_subgraph_connected(list(graph.graph.nodes().keys()))

def test_obtuse_angle():
    """Tests the SpecialEdgeGraph class on quadrilateral with an obtuse interior
    angle, with nodes at (0,0), (0,1), (1,10), (-1,10)."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    G.nodes[0]['x'] = 0
    G.nodes[0]['y'] = 0
    G.nodes[1]['x'] = 1
    G.nodes[1]['y'] = 10
    G.nodes[2]['x'] = -1
    G.nodes[2]['y'] = 10
    G.nodes[3]['x'] = 0
    G.nodes[3]['y'] = 1
    graph = SpecialEdgeGraph(G)
    assert len(graph.dual) == 1
    original_face = list(graph.dual)[0]
    # Should add a special edge to face
    assert len(graph.add_random_special_edges(n=1)) == 1
    # Should have doubled the number of faces
    assert len(graph.dual) == 2
    # Check that dual is connected
    assert nx.is_connected(graph.dual)
    # Check that both faces are new (not equal to the original face)
    assert all([f != original_face for f in graph.dual])
    # There are no faces to add edges to, should return 0
    assert len(graph.add_random_special_edges(n=1)) == 0
    # Assert that exactly one special edge was added
    assert len([e for e in graph.graph.edges.data() if e[2].get('special', False)]) == 1
    assert len(graph.special_edges[hash(original_face)]) == 1
    # Adding graph to itself should return itself
    assert graph + graph == graph
    # Adding graph to base graph should return graph with special edges
    assert graph + SpecialEdgeGraph(G) == graph
    assert SpecialEdgeGraph(G) + graph == graph
    # Remove special edge
    assert len(graph.remove_random_special_edges(n=1)) == 1
    # Removing special edge return face to the original one
    assert len(graph.dual) == 1 and tuple(graph.dual)[0] == original_face
    # Graph is now equal to base graph
    assert graph == SpecialEdgeGraph(G)
    # Adding graph to itself should return other copy of base graph
    assert graph + graph == SpecialEdgeGraph(G)
    # Check that is_original_subgraph_connected returns True, since graph
    # is connected
    assert graph.is_original_subgraph_connected(list(graph.graph.nodes().keys()))


def test_pentagon():
    """Nodes at (-1,0), (1,0), (2,2), (0, 4), (-2, 2)"""
    G = nx.Graph()
    G.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,0)])
    G.nodes[0]['x'] = -1
    G.nodes[0]['y'] = 0
    G.nodes[1]['x'] = 1
    G.nodes[1]['y'] = 0
    G.nodes[2]['x'] = 2
    G.nodes[2]['y'] = 2
    G.nodes[3]['x'] = 0
    G.nodes[3]['y'] = 4
    G.nodes[4]['x'] = -2
    G.nodes[4]['y'] = 2
    graph = SpecialEdgeGraph(G)
    assert len(graph.dual) == 1
    original_face = list(graph.dual)[0]
    # Should add a special edge to face
    assert len(graph.add_random_special_edges(n=2)) == 1
    # Should have doubled the number of faces
    assert len(graph.dual) == 2
    # Check that dual is connected
    assert nx.is_connected(graph.dual)
    # Check that both faces are new (not equal to the original face)
    assert all([f != original_face for f in graph.dual])
    # There is an additional edge to add, should return 1
    assert len(graph.add_random_special_edges(n=1)) == 1
    # Assert that exactly two special edges were added
    assert len([e for e in graph.graph.edges.data() if e[2].get('special', False)]) == 2
    assert len(graph.special_edges[hash(original_face)]) == 2
    # Adding graph to itself should return itself
    assert graph + graph == graph
    # Adding graph to base graph should return graph with special edges
    assert graph + SpecialEdgeGraph(G) == graph
    assert SpecialEdgeGraph(G) + graph == graph
    # Remove both special edges
    assert len(graph.remove_random_special_edges(n=2)) == 2
    # Removing special edges return face to the original one
    assert len(graph.dual) == 1 and tuple(graph.dual)[0] == original_face
    # Graph is now equal to base graph
    assert graph == SpecialEdgeGraph(G)
    # Adding graph to itself should return other copy of base graph
    assert graph + graph == SpecialEdgeGraph(G)
    # Check that is_original_subgraph_connected returns True, since graph
    # is connected
    assert graph.is_original_subgraph_connected(list(graph.graph.nodes().keys()))

    # Create two graphs with the two different possible special edges
    g1, g2 = SpecialEdgeGraph(G), SpecialEdgeGraph(G)
    assert g1._add_special_edge(original_face, 0, 2)
    assert g2._add_special_edge(original_face, 0, 3)
    # Graphs should not be equal
    assert g1 != g2
    # Adding graphs together should be equal to exactly one of the graphs
    g3 = g1 + g2
    assert (g3 == g1) != (g3 == g2)

# IMPORTANT edge cases to handle: case where face isn't a simple polygon
# We are only guaranteed the graph is straight line- each face is a slight
# generalization of a simple polygon, as edges can be reused and edges can
# intersect at vertices
def test_triangle_plus_edge():
    """Tests the SpecialEdgeGraph class on a triangle graph with nodes
    at (0,0), (0,1), (1,0), plus a node at (1/4, 1/4)"""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
    G.nodes[0]['x'] = 0
    G.nodes[0]['y'] = 0
    G.nodes[1]['x'] = 1
    G.nodes[1]['y'] = 0
    G.nodes[2]['x'] = 0
    G.nodes[2]['y'] = 1
    G.nodes[3]['x'] = 1/4
    G.nodes[3]['y'] = 1/4
    graph = SpecialEdgeGraph(G)
    assert len(graph.dual) == 1
    original_face = list(graph.dual)[0]
    # Should add a special edge to face
    assert len(graph.add_random_special_edges(n=1)) == 1
    # Should have doubled the number of faces
    assert len(graph.dual) == 2
    # Check that dual is connected
    assert nx.is_connected(graph.dual)
    # Check that both faces are new (not equal to the original face)
    assert all([f != original_face for f in graph.dual])
    # There is an additional edge to add, should return 1
    assert len(graph.add_random_special_edges(n=1)) == 1
    # Assert that exactly two special edges were added
    assert len([e for e in graph.graph.edges.data() if e[2].get('special', False)]) == 2
    assert len(graph.special_edges[hash(original_face)]) == 2
    # Adding graph to itself should return itself
    assert graph + graph == graph
    # Adding graph to base graph should return graph with special edges
    assert graph + SpecialEdgeGraph(G) == graph
    assert SpecialEdgeGraph(G) + graph == graph
    # Remove both special edges
    assert len(graph.remove_random_special_edges(n=2)) == 2
    # Removing special edges return face to the original one
    assert len(graph.dual) == 1 and tuple(graph.dual)[0] == original_face
    # Graph is now equal to base graph
    assert graph == SpecialEdgeGraph(G)
    # Adding graph to itself should return other copy of base graph
    assert graph + graph == SpecialEdgeGraph(G)
    # Check that is_original_subgraph_connected returns True, since graph
    # is connected
    assert graph.is_original_subgraph_connected(list(graph.graph.nodes().keys()))

    # Create two graphs with the two different possible special edges
    g1, g2 = SpecialEdgeGraph(G), SpecialEdgeGraph(G)
    assert g1._add_special_edge(original_face, 0, 3)
    assert g2._add_special_edge(original_face, 1, 3)
    # Graphs should not be equal
    assert g1 != g2
    # Adding graphs together should be equal to exactly one of the graphs
    g3 = g1 + g2
    assert (g3 == g1) != (g3 == g2)

def test_triangle_plus_diamond():
    """Tests the SpecialEdgeGraph class on a triangle graph with nodes
    at (-4,0), (0,4), (4,0), plus a diamond at (0,2), (-1,1), (1,1), (0,0)"""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 6), (6, 3)])
    G.nodes[0]['x'] = -4
    G.nodes[0]['y'] = 0
    G.nodes[1]['x'] = 0
    G.nodes[1]['y'] = 4
    G.nodes[2]['x'] = 4
    G.nodes[2]['y'] = 0
    G.nodes[3]['x'] = 0
    G.nodes[3]['y'] = 0
    G.nodes[4]['x'] = -1
    G.nodes[4]['y'] = 1
    G.nodes[5]['x'] = 0
    G.nodes[5]['y'] = 2
    G.nodes[6]['x'] = 1
    G.nodes[6]['y'] = 1
    graph = SpecialEdgeGraph(G)
    assert len(graph.dual) == 2
    original_faces = list(graph.dual)
    # Should add two special edges to faces
    assert len(graph.add_random_special_edges(n=2)) == 2
    # Should have doubled the number of faces
    assert len(graph.dual) == 4
    # Check that dual is connected
    assert nx.is_connected(graph.dual)
    # There is either 1 or 2 additional edges to add, should return 1 or 2
    num_added = len(graph.add_random_special_edges(n=2))
    assert num_added == 1 or num_added == 2
    assert len([e for e in graph.graph.edges.data() if e[2].get('special', False)]) == 2 + num_added
    assert len([edge for edges in graph.special_edges.values() for edge in edges]) == 2 + num_added
    # Adding graph to itself should return itself
    assert graph + graph == graph
    # Adding graph to base graph should return graph with special edges
    assert graph + SpecialEdgeGraph(G) == graph
    assert SpecialEdgeGraph(G) + graph == graph
    # Remove all special edges
    assert len(graph.remove_random_special_edges(n=4)) == 2 + num_added
    # Removing special edges return face to the original one
    assert len(graph.dual) == 2 and all([f in original_faces for f in graph.dual])
    # Graph is now equal to base graph
    assert graph == SpecialEdgeGraph(G)
    # Adding graph to itself should return other copy of base graph
    assert graph + graph == SpecialEdgeGraph(G)
    # Check that is_original_subgraph_connected returns True, since graph
    # is connected
    assert graph.is_original_subgraph_connected(list(graph.graph.nodes().keys()))

    # Create three graphs with the three different possible special edges
    g1, g2, g3 = SpecialEdgeGraph(G), SpecialEdgeGraph(G), SpecialEdgeGraph(G)
    # TODO we don't have any way of guaranteeing the order of the faces, so
    # this could break for any arbitrary reason. It's also just simplest to
    # make this assumption...
    assert g1._add_special_edge(original_faces[0], 0, 4)
    assert g1._add_special_edge(original_faces[1], 4, 6)
    assert g2._add_special_edge(original_faces[0], 6, 2)
    assert g3._add_special_edge(original_faces[1], 3, 5)
    # Graphs should not be equal
    assert g1 != g2 and g1 != g3 and g2 != g3
    # Adding graphs together should create new graphs (although they may
    # be equal to g1)
    g4, g5, g6, g7 = g1 + g2, g1 + g3, g2 + g3, g1 + g2 + g3
    assert all(g2 != g for g in [g4, g5, g6, g7])
    assert all(g3 != g for g in [g4, g5, g6, g7])


def test_triangle_plus_diamond_plus_line():
    """Tests the SpecialEdgeGraph class on a triangle with nodes at (-4,0),
    (0,4), and (4,0), plus a diamond at (0,3), (-1,2), (1,2), (0,1), and a
    line connecting the two shapes meeting at (0,0)"""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 6), (6,7), (7, 4)])
    G.nodes[0]['x'] = -4
    G.nodes[0]['y'] = 0
    G.nodes[1]['x'] = 0
    G.nodes[1]['y'] = 4
    G.nodes[2]['x'] = 4
    G.nodes[2]['y'] = 0
    G.nodes[3]['x'] = 0
    G.nodes[3]['y'] = 0
    G.nodes[4]['x'] = 0
    G.nodes[4]['y'] = 1
    G.nodes[5]['x'] = -1
    G.nodes[5]['y'] = 2
    G.nodes[6]['x'] = 0
    G.nodes[6]['y'] = 3
    G.nodes[7]['x'] = 1
    G.nodes[7]['y'] = 2
    graph = SpecialEdgeGraph(G)
    assert len(graph.dual) == 2
    original_faces = list(graph.dual)
    # Should add two special edges to faces
    assert len(graph.add_random_special_edges(n=2)) == 2
    # Should have doubled the number of faces
    assert len(graph.dual) == 4
    # Check that dual is connected
    assert nx.is_connected(graph.dual)
    # There is either 1 or 2 additional edges to add, should return 1 or 2
    num_added = len(graph.add_random_special_edges(n=2))
    assert num_added == 1 or num_added == 2
    assert len([e for e in graph.graph.edges.data() if e[2].get('special', False)]) == 2 + num_added
    assert len([edge for edges in graph.special_edges.values() for edge in edges]) == 2 + num_added
    # Adding graph to itself should return itself
    assert graph + graph == graph
    # Adding graph to base graph should return graph with special edges
    assert graph + SpecialEdgeGraph(G) == graph
    assert SpecialEdgeGraph(G) + graph == graph
    # Remove all special edges
    assert len(graph.remove_random_special_edges(n=4)) == 2 + num_added
    # Removing special edges return face to the original one
    assert len(graph.dual) == 2 and all([f in original_faces for f in graph.dual])
    # Graph is now equal to base graph
    assert graph == SpecialEdgeGraph(G)
    # Adding graph to itself should return other copy of base graph
    assert graph + graph == SpecialEdgeGraph(G)
    # Check that is_original_subgraph_connected returns True, since graph
    # is connected
    assert graph.is_original_subgraph_connected(list(graph.graph.nodes().keys()))


    # Create three graphs with the three different possible special edges
    g1, g2, g3 = SpecialEdgeGraph(G), SpecialEdgeGraph(G), SpecialEdgeGraph(G)
    # TODO we don't have any way of guaranteeing the order of the faces, so
    # this could break for any arbitrary reason. It's also just simplest to
    # make this assumption...
    assert g1._add_special_edge(original_faces[0], 0, 4)
    assert g1._add_special_edge(original_faces[1], 4, 6)
    assert g2._add_special_edge(original_faces[0], 6, 2)
    assert g3._add_special_edge(original_faces[1], 5, 7)
    # Graphs should not be equal
    assert g1 != g2 and g1 != g3 and g2 != g3
    # Adding graphs together should create new graphs (although they may
    # be equal to g1)
    g4, g5, g6, g7 = g1 + g2, g1 + g3, g2 + g3, g1 + g2 + g3
    assert all(g2 != g for g in [g4, g5, g6, g7])
    assert all(g3 != g for g in [g4, g5, g6, g7])


def test_funky_star_shape():
    """Tests the SpecialEdgeGraph class on a weird star-like shape, with nodes
    at (0,0), (2,-4), (3,1), (4,2), (1,3), (0,1), (-1,3), (-4,2), (-3,1), (-2,-1)
    """
    G = nx.Graph()
    G.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,0)])
    G.nodes[0]['x'] = 0
    G.nodes[0]['y'] = 0
    G.nodes[1]['x'] = 2
    G.nodes[1]['y'] = -4
    G.nodes[2]['x'] = 3
    G.nodes[2]['y'] = 1
    G.nodes[3]['x'] = 4
    G.nodes[3]['y'] = 2
    G.nodes[4]['x'] = 1
    G.nodes[4]['y'] = 3
    G.nodes[5]['x'] = 0
    G.nodes[5]['y'] = 1
    G.nodes[6]['x'] = -1
    G.nodes[6]['y'] = 3
    G.nodes[7]['x'] = -4
    G.nodes[7]['y'] = 2
    G.nodes[8]['x'] = -3
    G.nodes[8]['y'] = 1
    G.nodes[9]['x'] = -2
    G.nodes[9]['y'] = -1
    graph = SpecialEdgeGraph(G)
    assert len(graph.dual) == 1
    original_face = list(graph.dual)[0]
    # Should add one special edge to face
    assert len(graph.add_random_special_edges(n=1)) == 1
    # Should have doubled the number of faces
    assert len(graph.dual) == 2
    # Check that dual is connected
    assert nx.is_connected(graph.dual)
    # There is at least one additional edge to add, should return 1
    assert len(graph.add_random_special_edges(n=1)) == 1
    # Assert that exactly two special edges were added
    assert len([e for e in graph.graph.edges.data() if e[2].get('special', False)]) == 2
    assert len([edge for edges in graph.special_edges.values() for edge in edges]) == 2
    # Adding graph to itself should return itself
    assert graph + graph == graph
    # Adding graph to base graph should return graph with special edges
    assert graph + SpecialEdgeGraph(G) == graph
    assert SpecialEdgeGraph(G) + graph == graph
    # Remove all special edges
    assert len(graph.remove_random_special_edges(n=2)) == 2
    # Removing special edges return face to the original one
    assert len(graph.dual) == 1 and tuple(graph.dual)[0] == original_face
    # Check that dual is connected
    assert nx.is_connected(graph.dual)
    # Graph is now equal to base graph
    assert graph == SpecialEdgeGraph(G)
    # Adding graph to itself should return other copy of base graph
    assert graph + graph == SpecialEdgeGraph(G)
    # Check that is_original_subgraph_connected returns True, since graph
    # is connected
    assert graph.is_original_subgraph_connected(list(graph.graph.nodes().keys()))

    # Create two graphs with two different possible special edges
    g1, g2 = SpecialEdgeGraph(G), SpecialEdgeGraph(G)
    assert g1._add_special_edge(original_face, 0, 4)
    assert g1._add_special_edge(list(g1.dual)[0], 6, 9)
    assert g2._add_special_edge(original_face, 9, 5)
    # Graphs should not be equal
    assert g1 != g2
    # Adding graphs together should be equal to exactly one of the graphs
    g3 = g1 + g2
    assert (g3 == g1) != (g3 == g2)

def test_grid_graph():
    """Tests the SpecialEdgeGraph class on a 20x20 grid graph
    """
    # Create Grid Graph. Note, the node ids are their position in the Cartesian
    # grid
    G = nx.grid_graph(dim=[20, 20])
    for n in G.nodes():
        G.nodes[n]['x'] = n[0]
        G.nodes[n]['y'] = n[1]
    graph = SpecialEdgeGraph(G)
    assert len(graph.dual) == 361 # (19-1)^2
    # Check that dual is connected
    assert nx.is_connected(graph.dual)
    # Should add a special edge to each face
    assert len(graph.add_random_special_edges(n=10000)) == 361
    # Should have doubled the number of faces
    assert len(graph.dual) == 361 * 2
    # Check that dual is connected
    assert nx.is_connected(graph.dual)
    # There are no faces to add edges to, should return 0
    assert len(graph.add_random_special_edges(n=1)) == 0
    # Assert that 361 special edges were added
    assert len([e for e in graph.graph.edges.data() if e[2].get('special', False)]) == 361
    assert len(graph.special_edges) == 361
    # Adding graph to itself should return itself
    assert graph + graph == graph
    # Adding graph to base graph should return graph with special edges
    assert graph + SpecialEdgeGraph(G) == graph
    assert SpecialEdgeGraph(G) + graph == graph
    # Remove 300 special edges
    assert len(graph.remove_random_special_edges(n=300)) == 300
    # Assert that there are now 61 special edges
    assert len([e for e in graph.graph.edges.data() if e[2].get('special', False)]) == 61
    assert len([edge for edges in graph.special_edges.values() for edge in edges]) == 61
    # Check that there are 361 + 61 faces
    assert len(graph.dual) == 422
    # Check that dual is connected
    assert nx.is_connected(graph.dual)
    # Check that is_original_subgraph_connected returns True, since graph
    # is connected
    assert graph.is_original_subgraph_connected(list(graph.graph.nodes().keys()))
#
    # Get all special edges
    special_edges = [e for edge_list in graph.special_edges.values() for e in edge_list]
    for e in special_edges:
        # Check that the subgraph induced by the special edge (the two nodes) is
        # connected when including special edges, but not otherwise
        assert nx.is_connected(graph.graph.subgraph(e))
        assert not graph.is_original_subgraph_connected(tuple(e))

    # Create new grid graph
    graph2 = SpecialEdgeGraph(G)
    # Add random special edges to graph2
    assert len(graph2.add_random_special_edges(n=100)) == 100
    graph3 = graph + graph2
    # Up to 100 new special edges and faces have been added
    assert 461 <= len(graph3.dual) <= 522
    num_special_edges = len([e for e in graph3.graph.edges.data() if e[2].get('special', False)])
    assert 100 <= num_special_edges <= 161
    num_special_edges = len([e for edges in graph3.special_edges.values() for e in edges])
    assert 100 <= num_special_edges <= 161