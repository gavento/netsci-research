import itertools
import logging
import re
from pathlib import Path

import click
import networkx as nx
import numpy as np
import tqdm
from netreslib import network, utils
import re
import numpy as np

from .cli import cli

log = logging.getLogger(__name__)


class Generator:
    PARAMS = {"n": int}
    NAME = "None"

    def __init__(self, args, out_dir=".", seed=None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.param_samplers = {}

        self.out_dir = Path(out_dir)
        assert self.out_dir.is_dir()

        params = {}
        for praw in args:
            m = re.fullmatch(r"([^=]+)=(.*)", praw)
            if m:
                params[m.groups()[0]] = m.groups()[1]
            else:
                raise click.UsageError(
                    f"Can't parse parameter {praw!r} (hint: write params as 'n=1', 'k=3..6', 'd=1.0, 1.5, 2.0' etc.)"
                )

        for p in sorted(self.PARAMS):
            if p not in params:
                raise click.UsageError(f"Missing parameter {p}")
            f = utils.parse_generator(params.pop(p), self.rng.randint(0, 1 << 32))

            self.param_samplers[p] = lambda t=self.PARAMS[p], f=f: t(f())

        if params:
            raise click.UsageError(f"Unused parameters {sorted(params)}")

    def sample_g(self, n, seed):
        raise NotImplemented()

    def generate(self):
        params = {p: pgen() for p, pgen in self.param_samplers.items()}

        def fmt(x):
            return f"{x:.3}" if isinstance(x, float) else str(x)

        pstr = "_".join(f"{p}{fmt(pv)}" for p, pv in params.items())
        plstr = " ".join(f"{p}={fmt(pv)}" for p, pv in params.items())
        seed = self.rng.randint(0, 1000000)
        path = self.out_dir / f"{self.__class__.__name__}_{pstr}_s{seed:06}.json"
        g = self.sample_g(seed=seed, **params)

        net = network.Network.from_graph(
            path, g, label=f"{self.__class__.__name__} {plstr}"
        )
        net.attribs["origin"] = dict(
            model=self.__class__.__name__,
            model_name=self.NAME,
            seed=seed,
            **params,
        )
        net.compute_degree_stats()
        net.compute_clustering_stats()
        net.compute_distance_stats()
        net.write()


class BA(Generator):
    PARAMS = {"n": int, "k": int}
    NAME = "Barabasi-Albert"

    def sample_g(self, n, k, seed):
        return nx.random_graphs.barabasi_albert_graph(n, k, seed=seed)


class GND(Generator):
    PARAMS = {"n": int, "d": float}
    NAME = "Erdos-Renyi (avg deg)"

    def sample_g(self, n, d, seed):
        return nx.random_graphs.gnp_random_graph(n, d / (n - 1), seed=seed)


class GNP(Generator):
    PARAMS = {"n": int, "p": float}
    NAME = "Erdos-Renyi (edge prob)"

    def sample_g(self, n, p, seed):
        return nx.random_graphs.gnp_random_graph(n, p, seed=seed)


class WS(Generator):
    PARAMS = {"n": int, "k": int, "p": float}
    NAME = "Watts–Strogatz"

    def sample_g(self, n, k, p, seed):
        return nx.random_graphs.watts_strogatz_graph(n, k, p, seed=seed)


KINDS = {
    "BA": BA,
    "GND": GND,
    "GNP": GNP,
    "WS": WS,
}


@cli.command()
@click.argument("kind", nargs=1)
@click.argument("params", nargs=-1)
@click.option("-s", "--seed", type=int, default=None)
@click.option("-i", "--instances", type=int, default=1)
@click.option("-o", "--output_dir", type=click.Path(exists=True), default=".")
def gen(kind: str, params: list, seed: int, instances: int, output_dir: str):
    if kind not in KINDS:
        raise click.UsageError(f"Unknown graph model {kind} (known: {sorted(KINDS)})")
    gi = KINDS[kind](params, out_dir=output_dir, seed=seed)
    log.info(f"Generating {instances} {kind} graphs with {' '.join(params)}")
    for _i in tqdm.tqdm(range(instances), desc="Generating graphs"):
        gi.generate()
