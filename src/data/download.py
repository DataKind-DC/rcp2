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
            "fname": "nfirs.csv",
            "url": "https://drive.google.com/uc?id=1ENJZwazX7hJ4GwI03DKgX51y-644x-cZ",
            "known_hash": "0fcd2c4edae304dbb21c1b0dc6ca9afd17d7d65f21e51cd26571f9d42db7f825",
            "downloader": "download_from_google_drive",
        },
        ...
    }

Add more entries to make new files downloadable. New entries should have:

- ``fname``: The file basename.
- ``url``: The URL for download from.
- ``known_hash``: The file's SHA256 hash value to verify download integrity.

You can use :func:`pooch.file_hash` or :mod:`hashlib` to get file hash values.

Optionally, a source can specify ``downloader`` and ``processor`` functions
with special instructions for downloading and processing (e.g., unzipping) a
file. The values for these items can be functions or or strings that are mapped
to functions in either ``src.data.download.DOWNLOADERS`` or
``src.data.download.PROCESSORS``. For details, see Pooch documentation on
`custom downloaders`_ and `post-processing hooks`_.

Run this module as a script to download project data.

.. _Pooch: https://www.fatiando.org/pooch/latest/index.html
.. _custom downloaders: https://www.fatiando.org/pooch/latest/usage.html#custom-downloaders
.. _post-processing hooks: https://www.fatiando.org/pooch/latest/usage.html#post-processing-hooks

Attributes:
    SOURCES (dict): A registry of project data sources.
    DOWNLOADERS (dict): A registry of special downloading functions.
    PROCESSORS (dict): A registry of special post-processing functions.

"""
import gdown
import pooch
from src import utils


# A registry of all project data sources.
SOURCES = {
    "nfirs.csv": {
        "fname": "nfirs.csv",
        "url": "https://drive.google.com/uc?id=1ENJZwazX7hJ4GwI03DKgX51y-644x-cZ",
        "known_hash": "0fcd2c4edae304dbb21c1b0dc6ca9afd17d7d65f21e51cd26571f9d42db7f825",
        "downloader": "download_from_google_drive",
    },
}


# A registry of special downloading functions.
DOWNLOADERS = {
    "download_from_google_drive": lambda x, y, z: download_from_google_drive(x, y, z),
}


# A registry of special post-processing functions.
PROCESSORS = {
}


def fetch(url, fname=None, known_hash=None,
          path=None, downloader=None, processor=None):
    """Fetch a project file.
    
    This function downloads a source file to the project raw data directory.
    The function first checks for an up-to-date local copy of the file in
    ``path``. If it finds one, then it checks its SHA256 hash against the
    ``known_hash``. If the hashes match, then this function skips the download.

    If the function doesn't find a local copy of the file, or if the local hash
    doesn't match the known hash, then this function downloads the file. The
    function also compares the SHA256 hash of the downloaded file to the known
    hash and raises an error if they don't match.

    See :data:`src.data.download.SOURCES` for a dict of registered project
    files. Each item's contents can serve as arguments for this function. ::
    
      >>>from src.data import download
      >>>download.fetch(**download.SOURCES["my-file.csv"])
      "/path/to/my-file.csv"
    
    Args:
        url (str): The URL to download data from.
        fname (str): The base name of the file to fetch (e.g., "my-file.csv").
        known_hash (str): The file's SHA256 hash value.
        path (str): Directory in which to store the file.
        downloader (str, callable): A special downloading function.
        processor (str, callable): A special post-processing function.
        
    Returns:
        str: The path to the downloaded file.

    """
    # Use the default raw data directory if none is given.
    if not path:
        path = utils.DATA["raw"]

    # Look up the downloader function if needed.
    if type(downloader) == str:
        downloader = DOWNLOADERS[downloader]

    # Look up the processor function if needed.
    if type(processor) == str:
        processor = PROCESSORS[processor]

    return pooch.retrieve(url, known_hash, fname=fname, path=path,
                          downloader=downloader, processor=processor)


def download_from_google_drive(url, output_file, pooch):
    """A downloader to fetch files from Google Drive.
    
    Unlike some data sources, large files on Google Drive can't be downloaded
    by just pointing to a URL. This function uses a workaround from the gdown_
    package so that a ``pooch.Pooch`` instance can fetch Google Drive files. 
    
    Don't call this function directly. Instead, it works as the ``download``
    argument for ``pooch.Pooch.fetch`` or ``pooch.retrieve``. See the `Pooch
    documentation`_ for details.
    
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


if __name__ == "__main__":
    # Download the project source data.
    fetch(**SOURCES["nfirs.csv"])
