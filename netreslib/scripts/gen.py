import itertools
import logging
import re
from pathlib import Path

import click
import networkx as nx
import numpy as np
import tqdm
from netreslib import network, utils

from .cli import cli

log = logging.getLogger(__name__)


@cli.command()
@click.argument("ns", type=str)
@click.argument("ks", type=str)
@click.option("-s", "--seed", type=int, default=None)
@click.option("-i", "--instances", type=int, default=1)
@click.option("-o", "--output_dir", type=click.Path(exists=True), default=".")
def gen_ba(ns: str, ks: str, seed: int, instances: int, output_dir: str):
    output_dir = Path(output_dir)
    assert output_dir.is_dir()
    if seed is not None and instances > 1:
        raise click.UsageError("Do not combine --seed with --instances")
    ns = [int(x) for x in re.split("[, ]+", ns)]
    ks = [int(x) for x in re.split("[, ]+", ks)]
    log.info(
        f"Generrating {instances * len(ns) * len(ks)} Barabasi-Albert graphs for n={ns}, k={ks}"
    )
    for n, k, _i in tqdm.tqdm(
        itertools.product(ns, ks, range(instances)),
        desc="Generating graphs",
        total=instances * len(ns) * len(ks),
    ):
        gen_ba_one(n, k, seed=seed, output_dir=output_dir)


def gen_ba_one(n: int, k: int, seed: int, output_dir: Path):
    if seed is None:
        seed = np.random.RandomState().randint(1000000)
    path = output_dir / f"ba-n{n}-k{k}-s{seed:06}.json"
    name = f"Barabasi-Albert network, n={n}, k={k}, seed={seed}"
    with utils.logged_time(f"Creating {path}: {name}"):
        g = nx.random_graphs.barabasi_albert_graph(n, k, seed=seed)
        net = network.Network.from_graph(path, g)
        net.attribs["generator"] = dict(type="barabasi_albert", n=n, k=k, seed=seed)
        net.attribs["name"] = name
        net.write()
