import pyfar as pf
import pooch as po

from pathlib import Path

from irdl.repositories import doi_to_repository


def pooch_from_doi(doi, path=po.os_cache("irdl")):
    """Create a Pooch instance from a DOI.

    Parameters
    ----------
    doi : str
        The DOI of the archive.
    path : str
        Path to the directory where the data should be stored.

    Returns
    -------
    pup : Pooch
        The Pooch instance.
    """
    pup = po.create(path=path, base_url=doi, retry_if_failed=2, env="IRDL_DATA_DIR")
    repository = doi_to_repository(doi)
    repository.populate_registry(pup)
    for file in pup.registry.keys():
        pup.urls[file] = repository.download_url(file_name=file)
    return pup


def process(func):
    """Decorator to process downloaded files.

    The decorated function should take two arguments: the input file
    name and the output file name. The decorator checks if the output
    file already exists and is up to date. If so, it returns the output
    file name. Otherwise, it calls the decorated function to process
    the input file and create the output file.
    """

    def check_process(fname, action, pup=None):
        logger = po.get_logger()
        fname = Path(fname)
        outfile = (fname.parent.parent / "processed" / fname.stem).with_suffix(".far")
        if outfile.exists() and action == "fetch":
            logger.info(
                f"Processed file '{outfile}' exists and '{fname}' is up to date."
            )
            return pf.io.read(outfile)
        else:
            logger.info(f"Processing file '{fname}' and writing to '{outfile}'.")
            outfile.parent.mkdir(parents=True, exist_ok=True)
            return func(fname, outfile)

    return check_process
