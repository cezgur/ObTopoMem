# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, overload, Callable, FrozenSet, DefaultDict
import numpy as np
import igraph as ig
import leidenalg as la
from collections import defaultdict
from abc import ABC, abstractmethod
from bidict import bidict
import heapq
import matplotlib.pyplot as plt
from igraph.drawing.colors import ClusterColoringPalette


class Graph(ABC):
    def __init__(self,
                 nodes: Optional[List[Dict[str, Any]]] = None,
                 edges: Optional[List[Dict[str, Any]]] = None,
                 sim_attr: str = "weight",
                 directed: bool = False,
                 *args,
                 **kwargs):
        self.sim_attr = sim_attr
        self._node_id_to_node_idx_map: Optional[bidict] = None

        self._g = ig.Graph(directed=directed)

        if nodes is not None:
            self.add_nodes(nodes)

        if edges is not None:
            self.add_edges(edges)

        self.populateGraph(*args,
                           **kwargs)

        self.communities = None
        self._layout = None

    @property
    def g(self) -> ig.Graph:
        return self._g

    @property
    def node_ids(self) -> List[str]:
        return list(self._node_id_to_node_idx_map.keys()) if self._node_id_to_node_idx_map is not None else []

    @abstractmethod
    def populateGraph(self, *args, **kwargs) -> None: ...

    @overload
    def node_id_node_idx(self, node_id_x_: str) -> Optional[int]:...
    @overload
    def node_id_node_idx(self, node_id_x_: int) -> Optional[str]:...
    def node_id_node_idx(self, node_id_x_: Union[str, int]) -> Optional[Union[int, str]]:
        if self._node_id_to_node_idx_map is None:
            return None

        return self._node_id_to_node_idx_map.get(node_id_x_) if isinstance(node_id_x_, str) else self._node_id_to_node_idx_map.inv.get(node_id_x_)

    def get_edge_parameter(self,
                           node_ids: Tuple[str, str],
                           attr: Optional[str] = None,
                           default: Any = 0) -> Any:
        if attr is None:
            attr = self.sim_attr

        u_id, v_id = node_ids
        u_idx, v_idx = self.node_id_node_idx(u_id), self.node_id_node_idx(v_id)
        if u_idx is None or v_idx is None:
            if u_idx is None:
                raise ValueError(f"node_id: {u_id} is not found.")
            if v_idx is None:
                raise ValueError(f"node_id: {v_id} is not found.")

        u_idx = int(u_idx)
        v_idx = int(v_idx)

        g = self.g
        directed = g.is_directed()

        eid = g.get_eid(u_idx, v_idx, directed=directed, error=False)
        if eid == -1:
            return default

        edge = g.es[eid]
        return edge[attr]

    def add_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        if not nodes:
            return

        # Validate keys
        if "id" not in nodes[0]:
            raise ValueError("Each node must contain an 'id' key.")

        node_keys = set(nodes[0].keys())
        for i, d in enumerate(nodes, start=1):
            if set(d.keys()) != node_keys:
                raise ValueError(f"Node at position {i} has mismatched keys: {set(d.keys())} vs {node_keys}.")

        # Ensure map exists
        if self._node_id_to_node_idx_map is None:
            self._node_id_to_node_idx_map = bidict()

        # Filter out nodes whose id is already exists
        new_nodes: List[Dict[str, Any]] = []
        for d in nodes:
            nid = d["id"]
            if nid in self._node_id_to_node_idx_map:
                continue
            new_nodes.append(d)

        if not new_nodes:
            return

        # Build attributes from filtered nodes
        attrs = {k: [d[k] for d in new_nodes] for k in node_keys}

        # Add vertices to self._g
        start = len(self._g.vs)
        self._g.add_vertices(len(new_nodes), attributes=attrs)

        # Build/extend id-index map with finalized positions
        for i, nid in enumerate(attrs["id"]):
            assert self._node_id_to_node_idx_map is not None
            self._node_id_to_node_idx_map[nid] = start + i

    def add_edges(self, edges: List[Dict[str, Any]]) -> None:
        if not edges:
            return
        if "node_ids" not in edges[0] or self.sim_attr not in edges[0]:
            raise ValueError("Each edge must contain 'node_ids' and the similarity attribute.")

        # Build edge list and attributes
        e_pairs: List[Tuple[int, int]] =  []
        attrs: Dict[str, List[Any]] = defaultdict(list)

        for e in edges:
            u_id, v_id = e["node_ids"]
            u = self.node_id_node_idx(u_id)
            v = self.node_id_node_idx(v_id)
            if u is None or v is None:
                continue
            e_pairs.append((int(u), int(v)))
            for k, val in e.items():
                if k == "node_ids":
                    continue
                attrs[k].append(val)

        if not e_pairs:
            return

        self._g.add_edges(e_pairs, attributes=dict(attrs))
        dist_attr = f"__{self.sim_attr}_dist_rational__"
        if dist_attr in self.g.es.attribute_names():
            del self.g.es[dist_attr]

    def change_edges(self, edges: List[Dict[str, Any]]) -> None:
        if not edges:
            return
        if "node_ids" not in edges[0] or self.sim_attr not in edges[0]:
            raise ValueError("Each edge must contain 'node_ids' and the similarity attribute.")

        g = self.g

        directed = g.is_directed()

        for e in edges:
            u_id, v_id = e["node_ids"]
            u = self.node_id_node_idx(u_id)
            v = self.node_id_node_idx(v_id)

            # Skip if nodes do not exist
            if u is None or v is None:
                continue

            u = int(u)
            v = int(v)

            # Get eid of existing edge (or -1 if doesn't exist)
            eid = g.get_eid(u, v, directed=directed, error=False)
            if eid == -1:
                continue

            es = g.es[eid]

            # Update attributes (except node_ids)
            for k, val in e.items():
                if k == "node_ids":
                    continue
                es[k] = val

        dist_attr = f"__{self.sim_attr}_dist_rational__"
        if dist_attr in self.g.es.attribute_names():
            del self.g.es[dist_attr]


    def _cluster_leiden(self,
                         resolution: float = 1.0,
                         iterations: int = -1,
                         seed: int = 0,
                         attribute_out: str = "leiden_comm"):
        """
        Runs Leiden, then splits any internally-disconnected communities into
        connected components. Writes final membership to vertex attr "attribute_out".
        Returns refined membership, communities, and both pre- and post-refine quality.
        """
        g = self.g
        if g.vcount() == 0:
            return {"membership": [], "num_comms": 0, "quality_initial": 0.0,
                    "quality_refined": 0.0, "communities": []}

        if g.ecount() == 0:
            membership = list(range(g.vcount()))
            g.vs[attribute_out] = membership

            return {
                "membership": membership,
                "num_comms": g.vcount(),
                "quality_initial": 0.0,
                "quality_refined": 0.0,
                "communities": [[i] for i in membership]
            }

        objective = la.RBConfigurationVertexPartition

        # ---- 1. Run Leiden (RBConfiguration objective -- means Modularity at resolution = 1)
        part = la.find_partition(
            g,
            objective,
            weights=self.sim_attr,
            resolution_parameter=resolution,
            seed=seed,
            n_iterations=iterations
        )

        membership = np.asarray(part.membership, dtype=int)
        quality_initial = float(part.quality())
        g.vs[attribute_out] = membership.tolist()

        # ---- 2. Refine: split any disconnected pieces within each Leiden community
        refined = np.full(len(g.vs), -1, dtype=int)
        next_label = 0
        for cid in np.unique(membership):
            vids = np.where(membership == cid)[0]
            if vids.size == 0:
                continue
            sub = g.subgraph(vids)
            comps = sub.components()
            for comp in comps:
                comp_global = [int(vids[i]) for i in comp]
                refined[comp_global] = next_label
                next_label += 1

        # Write refined labels to graph
        g.vs[attribute_out] = refined.tolist()

        # ---- 3. Build communities from the refined membership
        if next_label == 0:
            return {
                "membership": refined.tolist(),
                "num_comms": 0,
                "quality_initial": quality_initial,
                "quality_refined": quality_initial,
                "communities": []
            }

        num_comms = int(next_label)
        communities: List[List[int]] = [[] for _ in range(num_comms)]
        for vid, cid in enumerate(refined):
            if cid >= 0:
                communities[int(cid)].append(int(vid))

        # ---- 4. Recompute quality for the REFINED partition (same objective).
        vp_refined = objective(
            g,
            weights=self.sim_attr,
            resolution_parameter=resolution,
            initial_membership=refined.tolist()
        )
        quality_refined = float(vp_refined.quality())

        return {
            "membership": refined.tolist(),
            "num_comms": num_comms,
            "quality_initial": quality_initial,
            "quality_refined": quality_refined,
            "communities": communities
        }

    def _cluster_infomap(self,
                         trials: int = 100,
                         attribute_out: str = "infomap_comm") -> Dict:
        """
        Runs Infomap, then splits any internally-disconnected communities into
        connected components. Writes final membership to vertex attr 'attribute_out'.

        Returns:
            {
                "membership": refined_membership_list,
                "num_comms": int,
                "quality_initial": float # modularity of raw infomap partition,
                "quality_refined": float # modularity after splitting disconnected components,
                "communities": List[List[int]]
            }

        Notes:
            - Infomap's own objective (map equation) is not exposed by igraph's API,
            so 'quality_*' here is graph modularity of the partition, using 'self.sim_attr'
            as edge weights. It is for diagnostics / comparisons only.
        """
        g: ig.Graph = self.g

        if g.vcount() == 0:
            return {
                "membership": [],
                "num_comms": 0,
                "quality_initial": 0.0,
                "quality_refined": 0.0,
                "communities": []
            }

        if g.ecount() == 0:
                    membership = list(range(g.vcount()))
                    g.vs[attribute_out] = membership

                    return {
                        "membership": membership,
                        "num_comms": g.vcount(),
                        "quality_initial": 0.0,
                        "quality_refined": 0.0,
                        "communities": [[i] for i in membership]
                    }

        # ---- 1. Run Infomap on the full graph.
        part = g.community_infomap(
            edge_weights=self.sim_attr,
            trials=trials
        )

        membership = np.asarray(part.membership, dtype=int)

        # Modularity of the raw Infomap partition (diagnostics only).
        quality_initial = float(g.modularity(membership.tolist(),
                                             weights=self.sim_attr))

        # Write initial labels to graph.
        g.vs[attribute_out] = membership.tolist()

        # ---- 2. Refine: split any disconnected pieces within in each Infomap community
        refined = np.full(len(g.vs), -1, dtype=int)
        next_label = 0
        for cid in np.unique(membership):
            vids = np.where(membership == cid)[0]
            if vids.size == 0:
                continue

            sub = g.subgraph(vids)
            comps = sub.components()

            for comp in comps:
                comp_global = [int(vids[i]) for i in comp]
                refined[comp_global] = next_label
                next_label += 1

        # Write refined labels
        g.vs[attribute_out] = refined.tolist()

        # ---- 3. Build communities for refined membership
        if next_label == 0:
            return {
                "membership": refined.tolist(),
                "num_comms": 0,
                "quality_initial": quality_initial,
                "quality_refined": quality_initial,
                "communities": []
            }

        num_comms = int(next_label)
        communities: List[List[int]] = [[] for _ in range(num_comms)]
        for vid, cid in enumerate(refined):
            if cid >= 0:
                communities[int(cid)].append(int(vid))

        # ---- 4. Recompute quality for the refined partition
        quality_refined = float(
            g.modularity(refined.tolist(), weights=self.sim_attr)
        )

        return {
            "membership": refined.tolist(),
            "num_comms": num_comms,
            "quality_initial": quality_initial,
            "quality_refined": quality_refined,
            "communities": communities
        }

    def _get_widest_path_similarity(self,
                                    node1_id: str,
                                    node2_id: str,
                                    sim_cache: Optional[Dict[FrozenSet[str], float]] = None) -> float:
        """
        Computes the widest-path similarity (maximum bottleneck value) between two nodes.
        """
        if node1_id not in self.node_ids or node2_id not in self.node_ids:
            return 0.0

        pair = frozenset((node1_id, node2_id))
        if sim_cache is not None and pair in sim_cache:
            return sim_cache[pair]

        idx1 = self.node_id_node_idx(node1_id)
        idx2 = self.node_id_node_idx(node2_id)
        if idx1 is None or idx2 is None or idx1 == idx2:
            return 0.0

        pq = [(-1.0, idx1)]
        max_strengths: DefaultDict[int, float] = defaultdict(float)
        max_strengths[idx1] = 1.0

        while pq:
            neg_strength, current = heapq.heappop(pq)
            strength = -neg_strength
            if strength < max_strengths[current]:
                continue
            if current == idx2:
                if sim_cache is not None:
                    sim_cache[pair] = strength
                return strength

            for neighbor in self.g.neighbors(current):
                eid = self.g.get_eid(current, neighbor)
                w = self.g.es[eid][self.sim_attr]
                new_strength = min(strength, w)
                if new_strength > max_strengths[neighbor]:
                    max_strengths[neighbor] = new_strength
                    heapq.heappush(pq, (-new_strength, neighbor))

        if sim_cache is not None:
            sim_cache[pair] = 0.0
        return 0.0

    def _ensure_dist_attr(self,
                          dist_attr: Optional[str] = None,
                          c_quantile: float = 0.9,
                          eps: float = 1e-12) -> str:
        """
        Ensures that a distance attribute is created from similarity weights
        using a rational squash: a = w / (w + c), then d = -log(a).

        This maps arbitrary positive weights (possibly >1) into additive,
        non-negative edge distances for shortest-path computations.

        Args:
            dist_attr: optional name for the distance edge attribute.
            c_quantile: quantile of weights used to set the scale c.
            eps: numerical stability constant.

        Returns:
            str: name of the created distance attribute.
        """
        if dist_attr is None:
            dist_attr = f"__{self.sim_attr}_dist_rational__"

        if dist_attr not in self.g.es.attribute_names():
            import numpy as np
            w = np.asarray(self.g.es[self.sim_attr], float)
            w = np.clip(w, eps, None)

            # determine smooth scale c from a high quantile
            c = float(np.quantile(w, c_quantile))
            c = max(c, eps)

            # rational squash: a = w / (w + c)  in (0,1)
            a = w / (w + c)
            a = np.clip(a, eps, 1.0)

            # additive distance
            d = -np.log(a)
            self.g.es[dist_attr] = d.tolist()

        return dist_attr

    def _get_shortest_path_similarity(self,
                                      node1_id: str,
                                      node2_id: str,
                                      sim_cache=None,
                                      eps: float = 1e-12) -> float:
        if node1_id not in self.node_ids or node2_id not in self.node_ids:
            return 0.0
        pair = frozenset((node1_id, node2_id))
        if sim_cache is not None and pair in sim_cache:
            return sim_cache[pair]

        i = self.node_id_node_idx(node1_id); j = self.node_id_node_idx(node2_id)
        if i is None or j is None:
            return 0.0

        dist_attr = self._ensure_dist_attr(eps=eps)
        mode = "ALL"
        d = self.g.shortest_paths(source=i, target=j, weights=dist_attr, mode=mode)[0][0]
        if d == float("inf"):
            s = 0.0
        else:
            s = 1.0 / (1.0 + d)    # or np.exp(-d/sigma)

        if sim_cache is not None:
            sim_cache[pair] = s
        return s

    def representative_gonzalez(self,
                                num_representatives: int,
                                *,
                                sim_to_dist: Callable,
                                epsilon: float = 1e-12,
                                start: Optional[int] = None) -> List[str]:
        """
        Gonzalez greedy k-center using WIDEST-PATH similarity:

        distance(u,v) = sim_to_dist( widest_path_similarity(u,v), epsilon )

        Args:
            num_representatives (int): Number of representatives (k).
            sim_to_dist (Callable): Function that maps similarity to distance.
                Examples:
                    lambda s, eps: 1.0 / (s + eps)
                    lambda s, eps: (1.0 - s) + eps
            epsilon (float): Small constant to avoid division by zero or to stabilize conversion.
            start (Optional[int]): Optional vertex index to use as the first center.

        Returns:
            List[str]: Representative node IDs.
        """
        g = self.g
        n = g.vcount()
        if num_representatives <= 0 or n == 0:
            return []

        k = min(num_representatives, n)

        node_indices = range(n)

        # Choose first center
        if start is None:
            deg = np.zeros(n, dtype=float)
            for e in g.es:
                w = float(e[self.sim_attr])
                u, v = e.tuple
                deg[u] += w
                deg[v] += w
            start = int(np.argmax(deg))
        else:
            if not (0 <= start < n):
                raise ValueError("start index out of range.")

        centers_idx: List[int] = [start]

        sim_cache: Dict[FrozenSet[str], float] = {}

        # Initialize distances to first center
        first_id = self.node_id_node_idx(start)
        assert first_id is not None
        d_to_C = np.zeros(n, dtype=float)
        for i in node_indices:
            if i == start:
                d_to_C[i] = 0.0
            else:
                second_id = self.node_id_node_idx(i)
                assert second_id is not None
                s = self._get_widest_path_similarity(first_id, second_id, sim_cache)
                d_to_C[i] = sim_to_dist(s, epsilon)

        # Greedy farthest first
        for _ in range(1, k):
            d_to_C[centers_idx] = -np.inf
            far_idx = int(np.argmax(d_to_C))
            centers_idx.append(far_idx)
            c_id = self.node_id_node_idx(far_idx)
            assert c_id is not None

            for i in node_indices:
                if i in centers_idx:
                    continue
                i_id = self.node_id_node_idx(i)
                assert i_id is not None
                s = self._get_widest_path_similarity(c_id, i_id, sim_cache)
                d = sim_to_dist(s, epsilon)
                if d < d_to_C[i]:
                    d_to_C[i] = d

        centers_id = [self.node_id_node_idx(c) for c in centers_idx]
        assert None not in centers_id
        return centers_id # type: ignore

    def representative_MMR(self,
                           num_representatives: int,
                           *,
                           lambd: float = 0.4,
                           start: Optional[int] = None,
                           normalize: bool = True,
                           eps: float = 1e-12) -> List[str]:
        """
        Maximal Marginal Relevance (MMR) representative selection on the graph
        using SHORTEST-PATH based similarity for redundancy.

        Score (x | S) = λ * Rel(x) - (1-λ) * max_{s∈S} Sim(x, s)

        - Rel(x): relevance / coverage proxy (similarity-weighted degree under self.sim_attr)
        - Sim(-,-): shortest-path based similarity via _get_shortest_path_similarity

        Args:
            num_representatives (int): Number of representative nodes to obtain (k).
            lambd (float): λ
            start (int): The optional first representative node index.
            normalize (bool): Whether to use normalization or not.
            eps (float): Small constant to avoid division by zero or to stabilize conversion.
        """
        g = self.g
        n = g.vcount()
        if num_representatives <= 0 or n == 0:
            return []

        k = min(num_representatives, n)

        if g.ecount() == 0:
            if start is None:
                start = 0
            elif not (0 <= start < n):
                raise ValueError("start index out of range.")

            selected = [start]
            selected.extend(i for i in range(n) if i != start)

            representatives = []
            for i in selected[:k]:
                node_id = self.node_id_node_idx(i)
                assert node_id is not None
                representatives.append(node_id)

            return representatives

        # Relevance proxy: similarity weighted degree
        rel = np.zeros(n, dtype=float)
        edge_weights_list = g.es[self.sim_attr]
        for eid, e in enumerate(g.es):
            w = float(edge_weights_list[eid])
            u, v = e.tuple
            rel[u] += w
            rel[v] += w

        # Normalization if normalize
        if normalize:
            max_rel  = float(np.max(rel)) if n > 0 else 1.0
            rel = rel / (max_rel + eps)

        # Upper bound for similarity scale (for normalizing redundancy if desired)
        sim_scale = float(np.max(edge_weights_list)) if len(edge_weights_list) else 1.0
        if sim_scale <= 0:
            sim_scale = 1.0 # avoid divide-by-zero; keeps redundancy term at 0

        # Init selection S
        selected: List[int] = []
        sim_cache: Dict[FrozenSet[str], float] = {}

        # Choose start: highes rel if not provided
        if start is None:
            start = int(np.argmax(rel))
        else:
            if not (0 <= start < n):
                raise ValueError(f"start index out of range.")
        selected.append(start)

        # Redundancy tracker: r[i] = max_{s∈S} Sim(i, s)
        r = np.zeros(n, dtype=float)
        s_id = self.node_id_node_idx(start)
        assert s_id is not None
        for i in range(n):
            if i == start:
                r[i] = 1.0
            else:
                i_id = self.node_id_node_idx(i)
                assert i_id is not None
                s = self._get_shortest_path_similarity(s_id, i_id, sim_cache)
                r[i] = s

        # Optional normalization of redundancy to [0, 1] using sim_cache
        if normalize:
            r = r / (sim_scale + eps)

        # Greedy MMR
        while len(selected) < k:
            # Score(x) = λ*Rel(x) - (1-λ)*Redundancy(x)
            scores = lambd * rel - (1.0 - lambd) * r

            # Never pick an already-selected index
            scores[selected] = -np.inf

            nxt = int(np.argmax(scores))
            selected.append(nxt)

            # Incrementally update redundancy wrt the newly added center
            c_id = self.node_id_node_idx(nxt)
            assert c_id is not None
            for i in range(n):
                if i == nxt:
                    continue
                i_id = self.node_id_node_idx(i)
                assert i_id is not None
                s = self._get_shortest_path_similarity(c_id, i_id, sim_cache)
                if normalize:
                    s = s / (sim_scale + eps)
                # r[i] = max(r[i], Sim(i, new_center))
                if s > r[i]:
                    r[i] = s

        representatives: List[str] = []
        for representative_ind in selected:
            representative_id = self.node_id_node_idx(representative_ind)
            assert representative_id is not None
            representatives.append(representative_id)

        return representatives

    def visualize_graph_view(
        self,
        sim_attr: Optional[str] = None,
        save_folder: Optional[str] = None,
        communities_in: Optional[List[List[str]]] = None,
        only_connected: bool = True
    ) -> None:
        """
        Visualize the Graph with optional community coloring.

        Args:
            sim_attr: name of the edge attribute storing similarity/weight.
            communities_in: Optional list of communities containing node IDs.
                        If None, uses self.communities (vertex indices).
                        Vertices not included in any community are drawn gray.
            save_folder: Optional path to save the figure. If None, shows the plot.
        """
        if sim_attr is None:
            sim_attr = self.sim_attr

        if communities_in is None:
            communities = self.communities
        else:
            communities = []
            for community in communities_in:
                node_indices = []
                for node_id in community:
                    node_idx = self.node_id_node_idx(node_id)
                    assert node_idx is not None
                    node_indices.append(node_idx)
                communities.append(node_indices)

        if only_connected:
            g_full = self.g
            degrees = g_full.degree()
            connected_vertices = [i for i, deg in enumerate(degrees) if deg >= 1]
            if not connected_vertices:
                connected_vertices = list(range(g_full.vcount()))
            g = g_full.subgraph(connected_vertices)
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(connected_vertices)}
        else:
            g = self.g


        # --- Layout using similarity weights ---
        if self._layout is None:
            if g.ecount() == 0:
                # No edges → FR layout will fail → use deterministic fallback
                self._layout = g.layout_circle()
            else:
                # Normal case → FR with weights
                self._layout = g.layout_fruchterman_reingold(weights=g.es[sim_attr])
        layout = self._layout

        # --- Edge thickness normalization ---
        # weights = g.es[sim_attr]
        # if len(weights) > 0:
        #     w_min, w_max = min(weights), max(weights)
        #     if w_max > w_min:
        #         edge_widths = [1 + 4 * (w - w_min) / (w_max - w_min) for w in weights]
        #     else:
        #         edge_widths = [1 for _ in weights]
        # else:
        #     edge_widths = []

        # --- Default (uncolored) style ---
        visual_style = {
            "layout": layout,
            "bbox": (800, 800),
            "margin": 40,
            "vertex_size": 12,
            "vertex_color": "lightgray",
            "vertex_label": g.vs["name"] if "name" in g.vs.attributes() else None
            # "edge_width": edge_widths,
        }

        # --- If communities provided: color by community ---
        if communities is not None:
            vcount = g.vcount()

            # membership: -1 means unassigned
            membership = [-1] * vcount

            # assign community IDs
            for comm_id, comm_vertices in enumerate(communities):
                for v_old in comm_vertices:
                    if only_connected:
                        new_v = old_to_new.get(v_old) # type: ignore
                    else:
                        new_v = v_old
                    if new_v is None:
                        # that vertex was isolated and is not in this subgraph
                        continue
                    membership[new_v] = comm_id

            num_comms = len(communities)

            palette = ClusterColoringPalette(num_comms)

            # --- Build vertex colors ---
            vertex_colors = []
            for m in membership:
                if m == -1:
                    # unassigned → gray
                    vertex_colors.append("lightgray")
                else:
                    vertex_colors.append(palette[m])

            visual_style["vertex_color"] = vertex_colors

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(8, 8))
        ig.plot(g, target=ax, **visual_style)
        plt.axis("off")
        plt.tight_layout()

        if save_folder:
            plt.savefig(save_folder, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            plt.show()
