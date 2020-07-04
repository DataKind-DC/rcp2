"""Utilities for downloading project source data.

This module makes it easy to download raw source data from online sources to
their standard locations in a local project directory. It leverages the Pooch_
package to verify the integrity of the downloaded files and to avoid
downloading files that are already available locally.

Top level function :func:`src.data.download.fetch` downloads source data files.

Top level variable :data:`src.data.download.SOURCES` is a dict with all the
source data files that are registered for this project. Here's a snippet: ::

    {
        # National Fire Incident Reporting System (NFIRS) data.
        "nfirs.csv": {
            "downloader": "download_from_google_drive",
            "sha256": "0fcd2c4edae304dbb21c1b0dc6ca9afd17d7d65f21e51cd26571f9d42db7f825",
            "url": "https://drive.google.com/uc?id=1ENJZwazX7hJ4GwI03DKgX51y-644x-cZ",
        },
        ...
    }

Add more entries to make new files downloadable. New entries should have:

1. A SHA256 hash value to verify download integrity.
2. A URL for the data source.

You can use :func:`pooch.file_hash` or :mod:`hashlib` to get file hash values.

Optionally, a source can specify a "downloader" function with special
instructions for downloading a file. The function must be defined somewhere
and registered in the "downloaders" dict in :func:`src.data.download.fetch`.
See Pooch documentation on `custom downloaders`_ for details.

Run this module as a script to download project data.

.. _Pooch: https://www.fatiando.org/pooch/latest/index.html
.. _custom downloaders: https://www.fatiando.org/pooch/latest/usage.html#custom-downloaders

Attributes:
    SOURCES: (dict): A registry of project data sources.
    POOCH (pooch.Pooch): An object managing file downloads.

"""
import gdown
import pooch
from src import utils


# A registry of all project data sources.
SOURCES =  {
    "nfirs.csv": {
        "downloader": "download_from_google_drive",
        "sha256": "0fcd2c4edae304dbb21c1b0dc6ca9afd17d7d65f21e51cd26571f9d42db7f825",
        "url": "https://drive.google.com/uc?id=1ENJZwazX7hJ4GwI03DKgX51y-644x-cZ",
    },
}


# A Pooch instance for downloading project files. Used by ``fetch``.
POOCH = pooch.create(
    path=utils.DATA["raw"],
    base_url="",
    registry={k: v["sha256"] for k, v in SOURCES.items()},
    urls={k: v["url"] for k, v in SOURCES.items()},
)


def fetch(fname):
    """Fetch a project file.
    
    This function downloads a source file to the project raw data directory.
    The function first checks for an up-to-date local copy of the file. If it
    finds one, then it checks its SHA256 hash against the known hash registered
    for the file. If the hashes match, then this function skips the download.
    
    If the function doesn't find a local copy of the file, or if the local hash
    doesn't match the registry, then this function downloads the file. The
    function also compares the SHA256 hash of the downloaded file to the one in
    the registry and raises an error if they don't match.
    
    See :data:`src.data.download.SOURCES` for a list of available files, along
    with their custom downloaders, SHA256 hashes, and source URLs.
    
    Args:
        fname: The base name of the file to fetch (e.g., "my-file.csv").
        
    Returns:
        str: The path to the downloaded file.

    """
    downloaders = {
        "download_from_google_drive": download_from_google_drive,
    }
    downloader = downloaders.get(SOURCES[fname].get("downloader"))
    return POOCH.fetch(fname, downloader=downloader)


def download_from_google_drive(url, output_file, pooch):
    """A downloader to fetch files from Google Drive.
    
    Unlike some data sources, large files on Google Drive can't be downloaded
    by just pointing to a URL. This function uses a workaround from the gdown_
    package so that a ``pooch.Pooch`` instance can fetch Google Drive files. 
    
    Don't call this function directly. Instead, it works as the ``download``
    argument for ``Pooch.fetch``. See the `Pooch documentation`_ for details.
    
    To get the ``url`` for a file in Google Drive, use this template: ::
    
      "https://drive.google.com/uc?id=MY-FILE-ID"
      
    To fill in ``MY-FILE-ID``, right click the file in Google Drive, and select
    "Get shareable link," copying out the UUID in the generated link.
    
    .. _gdown: https://github.com/wkentaro/gdown
    .. _Pooch documentation: https://www.fatiando.org/pooch/latest/usage.html#custom-downloaders
    
    Args:
        url: The URL for the file to download.
        output_file: The path to the output file.
        pooch: The Pooch instance calling this method.
        
    Returns:
        str: Output file name.
    
    """
    gdown.download(url, output=output_file)