import tqdm
import logging
from pathlib import Path

import click
import networkx as nx
import numpy as np
from netreslib import utils, network

from .cli import cli

log = logging.getLogger(__name__)


@cli.command()
@click.argument("json_paths", type=click.Path(exists=True), nargs=-1)
@click.option("-p", "--plot", is_flag=True)
def compute_stats(json_paths: list, plot: bool):
    log.info(f"Computing stats for {len(json_paths)} networks ...")
    for json_path in tqdm.tqdm(json_paths):
        compute_stats_one(json_path, plot)


def compute_stats_one(json_path: Path, plot: bool):
    net = network.Network.open(json_path)
    g = net.network
    degs = [d for _, d in g.degree]
    net.attribs["stats"] = dict(
        deg_min=np.min(degs),
        deg_max=np.max(degs),
        deg_mean=np.mean(degs),
        deg_std=np.std(degs),
    )
    net.write()
