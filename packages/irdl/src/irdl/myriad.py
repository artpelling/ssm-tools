import pooch as po
import pyfar as pf

from pathlib import Path
from zipfile import ZipFile

from irdl.downloader import pooch_from_doi, process


def get_myriad(room="SAL", path=po.os_cache("irdl")):
    """Download and extract the MYRIAD database from Zenodo.

    DOI: 10.5281/zenodo.7389996

    Parameters
    ----------

    room : str
        Name of the room to download. Either 'SAL' or 'AIL'.
    path : str or `pathlib.Path`
        Path to the directory where the data should be stored. Will be overwritten, if the
        environment variable `IRDL_DATA_DIR` is set. Default is the user cache directory.

    Returns
    -------
    data : dict
        Dictionary containing the impulse responses and the source and receiver coordinates. The
        impulse responses are stored in the key 'impulse_response' as a :class:`pyfar.Signal`. The
        source and receiver coordinates are stored as :class:`pyfar.Coordinates` in the keys
        'source_coordinates' and 'receiver_coordinates', respectively.
    """

    assert room in ["SAL", "AIL"], "room must be either 'SAL' or 'AIL'"

    path = Path(path) / "MYRIAD"
    doi = "10.5281/zenodo.7389996"
    zipfile = "MYRiAD_V2_econ.zip"

    pup = pooch_from_doi(doi, path=path)
    pup.fetch(zipfile, progressbar=True)

    logger = po.get_logger()

    root = Path("MYRiAD_V2_econ") / "audio"

    speakers = {
        "AIL": [
            "SL1",
            "SL2",
            "SL3",
            "SL4",
            "SL5",
            "SL6",
            "SL7",
            "SL8",
            "SU1",
            "SU2",
            "SU3",
            "SU4",
            "SU5",
            "SU6",
            "SU7",
            "SU8",
            "SU9",
            "SU10",
            "SU11",
            "SU12",
            "ST1",
            "ST2",
            "ST3",
            "ST4",
        ],
        "SAL": [
            "S0_1",
            "S0_2",
            "S-30_1",
            "S30_1",
            "S-45_2",
            "S45_2",
            "S-60_1",
            "S60_1",
            "S-90_1",
            "S90_1",
        ],
    }

    mics = [
        "BTELB",
        "BTELF",
        "BTERB",
        "BTERF",
        "DHL",
        "DHR",
        "XM1",
        "XM2",
        "XM3",
        "XM4",
        "XM5",
    ]

    @process
    def extract(file, process=True):
        if process:
            with ZipFile(path / zipfile, "r") as zf:
                for ls in speakers[room]:
                    for p in ["P1", "P2"]:
                        for m in mics:
                            wf = Path(room) / ls / p / (m + "_RIR.wav")
                            name = str(root / wf)
                            zf.getinfo(name).filename = str(wf)
                            logger.info(f"Extracting {name} to {path / wf}")
                            zf.extract(name, path=file.parent)

                return file

    return extract(path / "MYRiAD_econ.sofa", action="fetch", pup=pup)
