from typing import Annotated

from irdl import get_fabian
from typer import Option, run


def reduce_fabian(hato: Annotated[str, Option()] = None):
    """A placeholder function for reducing Fabian data."""
    fabian = get_fabian(hato=hato)


def main():
    run(reduce_fabian)
