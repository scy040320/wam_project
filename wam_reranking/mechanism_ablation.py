"""Matched inference ablations: change only dependency edges, never weights."""
from .belief import DEFAULT_GRAPH, DependencyGraph
from .evidence_arbitration import prepare_arbitrated_decisions

METHODS = ("value_only", "candidate_only", "no_dependency", "full")
EMPTY_GRAPH = DependencyGraph(())


def prepare_mechanism_decisions(*, method, **kwargs):
    if method not in ("no_dependency", "full"):
        raise ValueError("Attribution gate is reserved for attribution arms")
    if "graph" in kwargs:
        raise ValueError("Graph is fixed by the preregistered method arm")
    graph = EMPTY_GRAPH if method == "no_dependency" else DEFAULT_GRAPH
    return prepare_arbitrated_decisions(graph=graph, **kwargs)
