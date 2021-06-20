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

    @classmethod
    def open(cls, json_path: Path):
        json_path = Path(json_path)
        net = cls._open_skip_json(json_path)
        with utils.open_file(json_path, mode="r") as f:
            d = json.load(f)
            net.attribs = d["attribs"]
        return net

    def write(self, indent=2):
        """
        Write all data without closing data files etc.
        """
        with utils.open_file(self.json_path, mode="w") as f:
            json.dump(
                {"updated": utils.now_isofmt(), "attribs": utils.jsonize(self.attribs)},
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
        cls, json_path: Path, n: int, edges: np.ndarray, digraph: bool
    ) -> "Network":
        json_path = Path(json_path)
        assert not json_path.exists()
        net = cls._open_skip_json(json_path)
        m = edges.shape[0]
        assert edges.shape == (m, 2)
        edges = np.int32(edges)
        assert np.all(edges >= 0)
        assert np.all(edges < n)

        net.attribs["n"] = n
        net.attribs["m"] = m
        net.attribs["digraph"] = digraph
        net.attribs["created"] = utils.now_isofmt()
        net.attribs["name"] = ""
        net.add_array("/edges", edges)
        net.write()
        return net

    @classmethod
    def from_graph(cls, json_path: Path, g: nx.Graph) -> "Network":
        json_path = Path(json_path)
        edges = np.array(g.edges(), dtype=np.int32)
        return cls.from_edges(
            json_path, n=g.order(), edges=edges, digraph=isinstance(g, nx.DiGraph)
        )

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

    def __getitem__(self, name: str) -> dict:
        if name in self.attribs:
            return self.attribs[name]
        elif name in self.h5_file:
            return self.h5_file[name][()]
        else:
            raise KeyError(f"{name!r} not an attribute nor a data array")

    @classmethod
    def _open_skip_json(cls, json_path: Path):
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
