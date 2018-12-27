import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def draw_grid_graph(graph):
    node_positions = graph.nodes(data=False)
    pos = dict(zip(node_positions, node_positions))

    nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=True,
        node_size=250,
        node_color='lightblue')

    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(
        graph,
        pos=pos,
        edge_labels=edge_labels)


def grid_2d_graph_with_diagonal_edges(m, n, *args, **kwargs):
    graph = nx.grid_2d_graph(m, n, *args, **kwargs)

    for source, target in graph.edges:
        graph[source][target]['weight'] = 1.0

    width, height = m, n

    if isinstance(m, np.ndarray):
        x_index, y_index = m, n
        width, height = m.size, n.size
    elif isinstance(m, int):
        x_index, y_index = range(m), range(n)
        width, height = m, n
    else:
        raise NotImplementedError(m, n)

    for i, iv in enumerate(x_index):
        for j, jv in enumerate(y_index):
            start = (iv, jv)
            for di in (-1, 1):
                for dj in (-1, 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < width and 0 <= jj < height:
                        end = (x_index[ii], x_index[jj])
                        graph.add_edge(start, end, weight=1.0)

    return graph


def remove_walls_from_graph(graph, walls):
    nodes_np = np.array(graph.nodes)
    nodes_inside_walls_mask = np.zeros(nodes_np.shape[0], dtype=np.bool)
    for wall in walls:
        nodes_inside_wall_mask = wall.contains_point(nodes_np)
        nodes_inside_walls_mask |= nodes_inside_wall_mask

    nodes_inside_walls = nodes_np[nodes_inside_walls_mask]
    graph.remove_nodes_from(map(tuple, nodes_inside_walls))

    # We still have edges crossing the edges of the walls

    graph.remove_edges_from([
        edge for edge in graph.edges
        for wall in walls
        if wall.contains_segment(edge)
    ])

    return graph


def get_shortest_paths(graph):
    return nx.shortest_path(graph)


def get_shortest_distances(all_pairs_shortest_paths):
    observation_pairs, distances = [], []
    for start, all_shortest_paths in all_pairs_shortest_paths.items():
        for end, path in all_shortest_paths.items():
            observation_pairs.append([start, end])
            distances.append(len(path) - 1)

    observation_pairs = np.array(observation_pairs)
    distances = np.array(distances)

    return observation_pairs, distances
