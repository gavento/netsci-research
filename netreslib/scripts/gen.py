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


def parse_generator(ctx, val, rng_name="rng"):
    """
    Parse list of numbers, a range, or a distribution to be drawn from.
    Ex.: "1,2,3" "1.0, 2.3," "2..5" "U(1.0,2)" "LU(1, 1000)"
    """
    seed = ctx.obj[rng_name].randint(0, 1<<32)
    try:
        return utils.parse_generator(val, seed=seed)
    except Exception as e:
        raise click.UsageError(f"Error parsing {val!r} ({e})") from e


@cli.group()
@click.argument("n")
@click.option("-s", "--seed", type=int, default=None)
@click.option("-i", "--instances", type=int, default=1)
@click.option("-o", "--output_dir", type=click.Path(exists=True), default=".")
@click.pass_context
def gen(ctx, n: str, seed: int, instances: int, output_dir: str):
    ctx.ensure_object(dict)
    o = ctx.obj

    o["output_dir"] = Path(output_dir)
    assert o["output_dir"].is_dir()

    o["seed"] = seed
    o["rng"] = np.random.RandomState(seed)
    o["instances"] = instances
    o["n"] = parse_generator(ctx, n)


@gen.command()
@click.argument("k")
@click.pass_context
def ba(ctx, k: str):
    o = ctx.obj
    ngen = o["n"]
    kgen = parse_generator(ctx, k)

    for _i in tqdm.tqdm(range(o["instances"]), desc="Generating graphs"):
        n = int(ngen())
        k = int(kgen())
        seed = o["rng"].randint(0, 1000000)
        path = o["output_dir"] / f"ba-n{n}-k{k}-s{seed:06}.json"
        g = nx.random_graphs.barabasi_albert_graph(n, k, seed=seed)
        net = network.Network.from_graph(path, g, label=f"Barabasi-Albert n={n} k={k}")
        net.attribs["origin"] = dict(
            model="barabasi_albert",
            k=k,
            seed=seed,
        )
        net.write()
