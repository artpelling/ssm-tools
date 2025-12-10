import typer

from irdl.downloader import CACHE_DIR

app = typer.Typer()


@app.command()
def fabian(
    kind: str = typer.Option("measured", help="Type of HRTF to download. Either 'measured' or 'modeled'."),
    hato: int = typer.Option(
        0,
        help=(
            "Head-above-torso-rotation of HRTFs in degrees. Either 0, 10, 20, 30, 40, 50, 310, 320, 330, 340 or 350."
        ),
    ),
    path: str = typer.Option(
        CACHE_DIR,
        help=(
            "Path to the directory where the data should be stored. Will be overwritten, if the environment variable"
            " `IRDL_DATA_DIR` is set. Default is the user cache directory."
        ),
    ),
):
    """Download and extract the FABIAN HRTF Database v4 from DepositOnce.

    DOI: 10.14279/depositonce-5718.5
    """
    from irdl.fabian import get_fabian

    get_fabian(kind=kind, hato=hato, path=path)


@app.command()
def miracle(
    scenario: str = typer.Option("A1", help="Name of the scenario to download. Either 'A1', 'A2', 'D1', or 'R2'."),
    path: str = typer.Option(
        CACHE_DIR,
        help=(
            "Path to the directory where the data should be stored. Will be overwritten, if the environment variable"
            " `IRDL_DATA_DIR` is set. Default is the user cache directory."
        ),
    ),
):
    """Download and extract the MIRACLE database from DepositOnce.

    DOI: 10.14279/depositonce-20837
    """
    from irdl.miracle import get_miracle

    get_miracle(scenario=scenario, path=path)
