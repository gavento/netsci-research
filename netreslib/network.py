import json
from pathlib import Path

import attr
import h5py
import hdf5plugin
import networkx as nx
import numpy as np

from . import utils


@attr.s
class Network:
    base_path: Path = attr.ib()
    json_path: Path = attr.ib()
    h5_path: Path = attr.ib()
    h5_file: h5py.File = attr.ib()
    _network: nx.Graph = attr.ib(default=None)
    attribs: dict = attr.ib(factory=dict)
    created: str = attr.ib(factory=utils.now_isofmt)

    @classmethod
    def open(cls, json_path: Path):
        json_path = Path(json_path)
        net = cls._open_skip_json(json_path)
        with utils.open_file(json_path, mode="r") as f:
            d = json.load(f)
            net.created = d["created"]
            net.attribs = d["attribs"]
        return net

    def write(self, indent=2):
        """
        Write all data without closing data files etc.
        """
        with utils.open_file(self.json_path, mode="w") as f:
            json.dump(
                {
                    "updated": utils.now_isofmt(),
                    "created": self.created,
                    "network": utils.jsonize(self.attribs),
                },
                f,
                indent=indent,
            )
            f.write("\n")
        self.h5_file.flush()

    def export_graphml(self, compress_gzip=True):
        """
        Export the graph as Graphml, compressed by default.
        """
        path = self.base_path.with_name(self.base_path.name + ".graphml.gz")
        if not compress_gzip:
            path = path.with_suffix("")
        nx.write_graphml(self.network, path)

    @classmethod
    def from_edges(
        cls,
        json_path: Path,
        n: int,
        edges: np.ndarray,
        digraph: bool,
        label="",
    ) -> "Network":
        json_path = Path(json_path)
        assert not json_path.exists()
        net = cls._open_skip_json(json_path)
        m = edges.shape[0]
        assert edges.shape == (m, 2)
        edges = np.int32(edges)
        assert np.all(edges >= 0)
        assert np.all(edges < n)

        net.created = utils.now_isofmt()
        net.attribs["n"] = n
        net.attribs["m"] = m
        net.attribs["digraph"] = digraph
        net.attribs["label"] = label
        net.attribs["stats"] = {}
        net.attribs["origin"] = {}
        net.add_array("/edges", edges)
        net.write()
        return net

    @classmethod
    def from_graph(cls, json_path: Path, g: nx.Graph, label="") -> "Network":
        g = nx.convert_node_labels_to_integers(g)
        json_path = Path(json_path)
        if g.size() == 0:
            edges = np.zeros((0, 2), dtype=np.int32)
        else:
            edges = np.array(g.edges(), dtype=np.int32)
        net = cls.from_edges(
            json_path,
            n=g.order(),
            edges=edges,
            digraph=isinstance(g, nx.DiGraph),
            label=label,
        )
        net._network = g
        return net

    def get_directed_edges(self) -> np.ndarray:
        edges = self.h5_file["/edges"][()]
        if not self["digraph"]:
            edges = np.concatenate((edges, edges[:, 1::-1]))
        return edges

    def add_array(self, name: str, array_data: np.ndarray, compress: bool = True):
        c = hdf5plugin.Blosc(cname="zstd") if compress else None
        if array_data.nbytes < 1024:
            c = None
        self.h5_file.create_dataset(name, data=array_data, compression=c)

    @property
    def network(self):
        if self._network is None:
            self._network = nx.DiGraph() if self["digraph"] else nx.Graph()
            self._network.add_nodes_from(range(self["n"]))
            self._network.add_edges_from(self["edges"])
        return self._network

    @property
    def n(self):
        return self.attribs["n"]

    @property
    def m(self):
        return self.attribs["m"]

    def compute_distance_stats(self):
        sts = self.attribs["stats"]
        g = self.network
        cc = list(nx.connected_components(g))
        sts["components"] = len(cc)
        if len(cc) > 1:
            sts["diameter"] = np.inf
            sts["radius"] = np.inf
            sts["distance_mean"] = np.inf
        else:
            print(self.attribs)
            sts["diameter"] = nx.algorithms.distance_measures.diameter(g)
            sts["radius"] = nx.algorithms.distance_measures.radius(g)
            sts["distance_mean"] = nx.algorithms.shortest_paths.generic.average_shortest_path_length(g)

    def compute_clustering_stats(self):
        sts = self.attribs["stats"]
        g = self.network
        sts["transitivity"] = nx.algorithms.cluster.transitivity(g)
        c = list(nx.algorithms.cluster.clustering(g).values())
        sts["clustering_mean"] = np.mean(c)
        sts["clustering_std"] = np.std(c)

    def compute_degree_stats(self):
        """Compute degree distribution properties"""
        sts = self.attribs["stats"]
        g = self.network
        degs = [k for _, k in g.degree()] or [0]
        sts["degree_mean"] = np.mean(degs)
        sts["degree_std"] = np.std(degs)
        sts["degree_median"] = np.median(degs)
        sts["degree_min"] = np.min(degs)
        sts["degree_max"] = np.max(degs)

    def __getitem__(self, name: str) -> dict:
        if name in self.attribs:
            return self.attribs[name]
        elif name in self.h5_file:
            return self.h5_file[name][()]
        else:
            raise KeyError(f"{name!r} not an attribute nor a data array")

    @classmethod
    def _open_skip_json(cls, json_path: Path):
        """Return a Network instance woth open H5 file, not opening the JSON info file."""
        json_path = Path(json_path)
        base_path = utils.file_basic_path(json_path, ".json")
        h5_path = base_path.with_name(base_path.name + ".h5")
        h5_file = h5py.File(h5_path, mode="a")
        return cls(
            base_path=base_path,
            json_path=json_path,
            h5_path=h5_path,
            h5_file=h5_file,
        )
