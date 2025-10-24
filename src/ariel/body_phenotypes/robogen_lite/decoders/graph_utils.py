"""Graph utility functions for robot morphologies.

This module provides shared utilities for working with NetworkX graphs
representing robot morphologies, including JSON serialization and visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph
from networkx.readwrite import json_graph

# Global constants
SEED = 42
DPI = 300


def save_graph_as_json(
    graph: DiGraph[Any],
    save_file: Path | str | None = None,
) -> None:
    """
    Save a directed graph as a JSON file.

    Parameters
    ----------
    graph : DiGraph
        The directed graph to save.
    save_file : Path | str | None, optional
        The file path to save the graph JSON, by default None
    """
    if save_file is None:
        return

    data = json_graph.node_link_data(graph, edges="edges")
    json_string = json.dumps(data, indent=4)

    with Path(save_file).open("w", encoding="utf-8") as f:
        f.write(json_string)


def load_graph_from_json(
    load_file: Path | str,
) -> DiGraph[Any]:
    """
    Load a directed graph from a JSON file.

    Parameters
    ----------
    load_file : Path | str
        The file path to load the graph JSON.

    Returns
    -------
    DiGraph
        The loaded directed graph.
    """
    with Path(load_file).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return json_graph.node_link_graph(
        data,
        directed=True,
        multigraph=False,
        edges="edges",
    )


def draw_graph(
    graph: DiGraph[Any],
    title: str = "NetworkX Directed Graph",
    save_file: Path | str | None = None,
) -> None:
    """
    Draw a directed graph.

    Parameters
    ----------
    graph : DiGraph
        The directed graph to draw.
    title : str
        The title of the graph.
    save_file : Path | str | None, optional
        The file path to save the graph image, by default None
    """
    plt.figure()

    pos = nx.spectral_layout(graph)

    pos = nx.spring_layout(graph, pos=pos, k=1, iterations=20, seed=SEED)

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=150,
        node_color="#FFFFFF00",
        edgecolors="blue",
        font_size=8,
        width=0.5,
    )

    edge_labels = nx.get_edge_attributes(graph, "face")

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_color="red",
        font_size=8,
    )

    plt.title(title)

    # Save the graph visualization
    if save_file:
        plt.savefig(save_file, dpi=DPI)
    else:
        # Show the plot
        plt.show()
