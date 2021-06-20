import logging
from pathlib import Path

import click
import networkx as nx
import numpy as np
from netreslib import utils, network

from .cli import cli

log = logging.getLogger(__name__)


@cli.command()
@click.argument("json_path", type=click.Path(exists=True))
def info(json_path: Path):
    net = network.Network.open(json_path)
    print(
        f"""{net['name'] or "Unknown"} (created {net['created']}):
  n={net['n']}, m={net['m']}"""
    )
