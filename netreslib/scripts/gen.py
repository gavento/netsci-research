import logging
from pathlib import Path

import click
import networkx as nx
import numpy as np
from netreslib import utils, network

from .cli import cli

log = logging.getLogger(__name__)


@cli.command()
@click.argument("n", type=int)
@click.argument("k", type=int)
@click.option("-s", "--seed", type=int, default=None)
@click.option("-o", "--output_dir", type=click.Path(exists=True), default=Path("."))
def gen_ba(n: int, k: int, seed: int, output_dir: Path):
    assert output_dir.is_dir()
    if seed is None:
        seed = np.random.randint(1000000)
    path = output_dir / f"ba-n{n}-k{k}-s{seed:06}.json"
    name = f"Barabasi-Albert network, n={n}, k={k}, seed={seed}"
    with utils.logged_time(f"Creating {path}: {name}"):
        g = nx.random_graphs.barabasi_albert_graph(n, k, seed=seed)
        net = network.Network.from_graph(path, g)
        net.attribs["generator"] = dict(type="barabasi_albert", n=n, k=k, seed=seed)
        net.attribs["name"] = name
        net.write()
