#!/usr/bin/env python3

from pooch.downloaders import (
    DataRepository,
    FigshareRepository,
    ZenodoRepository,
    DataverseRepository,
    doi_to_url,
)
from pooch.utils import parse_url


DEFAULT_TIMEOUT = 30  # seconds


class DSpaceRepository(DataRepository):
    def __init__(self, doi, archive_url):
        self.archive_url = archive_url
        self.doi = doi
        self._api_response = None

    @classmethod
    def initialize(cls, doi, archive_url):
        """
        Initialize the data repository if the given URL points to a
        corresponding repository.

        Initializes a data repository object. This is done as part of
        a chain of responsibility. If the class cannot handle the given
        repository URL, it returns `None`. Otherwise a `DSpaceRepository`
        instance is returned.

        Parameters
        ----------
        doi : str
            The DOI that identifies the repository
        archive_url : str
            The resolved URL for the DOI
        """

        # Check whether this is a Figshare URL
        parsed_archive_url = parse_url(archive_url)
        if parsed_archive_url["netloc"] != "depositonce.tu-berlin.de":
            return None

        return cls(doi, archive_url)

    @property
    def api_response(self):
        if self._api_response is None:
            # Lazy import requests to speed up import time
            import requests  # pylint: disable=C0415

            article_id = self.archive_url.split("/")[-1]
            bundles = requests.get(
                f"https://api-depositonce.tu-berlin.de/server/api/core/items/{article_id}/bundles",
                timeout=DEFAULT_TIMEOUT,
            ).json()["_embedded"]["bundles"]
            for b in bundles:
                if b["name"] == "ORIGINAL":
                    break
            bitstreams = requests.get(
                b["_links"]["bitstreams"]["href"],
                timeout=DEFAULT_TIMEOUT,
            ).json()["_embedded"]["bitstreams"]
            self._api_response = {
                bs["name"]: {
                    "url": bs["_links"]["content"]["href"],
                    "checksum": f"{bs['checkSum']['checkSumAlgorithm']}:{bs['checkSum']['value']}",
                }
                for bs in bitstreams
            }

        return self._api_response

    def download_url(self, file_name):
        """
        Use the repository API to get the download URL for a file given
        the archive URL.

        Parameters
        ----------
        file_name : str
            The name of the file in the archive that will be downloaded.

        Returns
        -------
        download_url : str
            The HTTP URL that can be used to download the file.
        """
        return self.api_response[file_name]["url"]

    def populate_registry(self, pooch):
        """
        Populate the registry using the data repository's API

        Parameters
        ----------
        pooch : Pooch
            The pooch instance that the registry will be added to.
        """

        for name, info in self.api_response.items():
            pooch.registry[name] = info["checksum"]


def doi_to_repository(doi):
    """
    Instantiate a data repository instance from a given DOI.

    This function implements the chain of responsibility dispatch
    to the correct data repository class.

    Parameters
    ----------
    doi : str
        The DOI of the archive.

    Returns
    -------
    data_repository : DataRepository
        The data repository object
    """

    # This should go away in a separate issue: DOI handling should
    # not rely on the (non-)existence of trailing slashes. The issue
    # is documented in https://github.com/fatiando/pooch/issues/324
    if doi[-1] == "/":
        doi = doi[:-1]

    repositories = [
        FigshareRepository,
        ZenodoRepository,
        DSpaceRepository,
        DataverseRepository,
    ]

    # Extract the DOI and the repository information
    archive_url = doi_to_url(doi)

    # Try the converters one by one until one of them returned a URL
    data_repository = None
    for repo in repositories:
        if data_repository is None:
            data_repository = repo.initialize(
                archive_url=archive_url,
                doi=doi,
            )

    if data_repository is None:
        repository = parse_url(archive_url)["netloc"]
        raise ValueError(
            f"Invalid data repository '{repository}'. "
            "To request or contribute support for this repository, "
            "please open an issue at https://github.com/fatiando/pooch/issues"
        )

    return data_repository
