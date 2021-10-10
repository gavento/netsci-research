import json
import logging
from pathlib import Path

import click
import tqdm
from netreslib import network, utils

from .cli import cli

log = logging.getLogger(__name__)


@cli.command()
@click.argument("json_paths", nargs=-1, type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path())
def collect(json_paths: list, output: str):
    output = Path(output)
    res = []
    for jp in tqdm.tqdm(json_paths, desc="Collecting"):
        net = network.Network.open(jp)
        net.attribs["path"] = str(jp)
        res.append(net.attribs)

    with utils.open_file(output, mode="wt") as f:
        json.dump(res, f, indent=None)

    log.info(f"Collected {len(json_paths)} networks into {output}")
