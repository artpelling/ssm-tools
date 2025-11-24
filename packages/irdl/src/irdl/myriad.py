import numpy as np
import pooch as po
import pyfar as pf

from itertools import chain
from pathlib import Path
from zipfile import ZipFile

from irdl.downloader import pooch_from_doi, process


def get_myriad(room="SAL", array="circular", config="P1", path=po.os_cache("irdl")):
    """Download and extract the MYRIAD database from Zenodo.

    DOI: 10.5281/zenodo.7389996

    Parameters
    ----------

    room : str
        Name of the room to download. Either 'SAL' or 'AIL'.
    array : str
        Type of microphone array to download. Either 'circular', 'dummy head', 'external' or
        'behind-the-ear'.
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
    assert array in ["circular", "dummy head", "external", "behind-the-ear"], "mics must be one of ['circular', 'dummy head', 'external', 'behind-the-ear']"
    if room == "SAL":
        assert array != "circular", "circular microphone array not available in SAL room"
    elif room == "AIL":
        assert config in ["P1", "P2"], "config must be either 'P1' or 'P2' for AIL room"

    path = Path(path) / "MYRIAD"
    doi = "10.5281/zenodo.7389996"
    zipfile = "MYRiAD_V2_econ.zip"

    pup = pooch_from_doi(doi, path=path)
    pup.fetch(zipfile, progressbar=True)

    logger = po.get_logger()

    root = Path("MYRiAD_V2_econ")

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
    mics = {
        "circular": ["CMA10_-90", "CMA10_0", "CMA10_90", "CMA10_180", "CMA20_-135", "CMA20_-90", "CMA20_-45", "CMA20_0", "CMA20_45", "CMA20_90", "CMA20_135", "CMA20_180"],
        "dummy head": ["DHL", "DHR"],
        "external": ["XM1", "XM2", "XM3", "XM4", "XM5"],
        "behind-the-ear": ["BTELB", "BTELF", "BTERB", "BTERF"],
    }

    sampling_rate = 44100
    n_samples = 132300

    def iter_files(room, array, config):
        for ls in speakers[room]:
            wf = Path("audio") / room / ls
            if room == "AIL":
                wf /= config
            for m in mics[array]:
                yield wf / f"{m}_RIR.wav"

        yield Path("coord") / f"{room}.csv"

    @process
    def extract(file, process=True):
        if process:
            with ZipFile(path / zipfile, "r") as zf:
                for wf in chain(iter_files(room, array, config)):
                    member = str(root / wf)
                    zf.getinfo(member).filename = str(wf.relative_to(root))

                    logger.info(f"Extracting {member} to {path / wf}")
                    zf.extract(member, path=file.parent)

        irs = np.zeros((len(mics[array])*len(speakers[room]), n_samples))
        for i, wf in enumerate(iter_files(room, array, config)):
            irs[i] = pf.io.read_audio(path / wf).time

        print(path)
        coords = np.genfromtxt(path / f"{room}.csv", delimiter=",", names=True)
        print(coords)
        return {
            "impulse_response": pf.Signal(irs.reshape(len(speakers[room]), len(mics[array]), n_samples), sampling_rate=sampling_rate),
        }

    return extract(path / "MYRiAD_econ.sofa", action="fetch", pup=pup)
