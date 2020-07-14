import pytest
import responses
from src.data import download


def declare_action(fname, action, pooch):
    """Declare the download action taken.
    
    This function helps us know if ``src.data.download.fetch`` downloaded a
    missing file, fetched an available file, or updated on old file.
    
    Args:
        fname (str): The file name.
        action (str): "download", "fetch", or "update".
        pooch (pooch.Pooch): The caller.

    """
    return action


source = {
    "fname": "data.txt",
    "url": "https://my-data-source.com/data.txt",
    "known_hash": "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03",
    "processor": declare_action,
}


@responses.activate
def test_fetch_download(tmp_path):
    """Download a file when no local copy exists."""
    responses.add(method=responses.GET, url=source["url"], body="hello\n")
    assert download.fetch(**source, path=tmp_path) == "download"
    
    
@responses.activate
def test_fetch_fetch(tmp_path):
    """Fetch a local file when it exists."""
    responses.add(method=responses.GET, url=source["url"], body="hello\n")
    assert download.fetch(**source, path=tmp_path) == "download"
    assert download.fetch(**source, path=tmp_path) == "fetch"
    
    
@responses.activate
def test_fetch_update(tmp_path):
    """Update a local file when it is out of date."""
    responses.add(method=responses.GET, url=source["url"], body="hello\n")
    with open(tmp_path / "data.txt", "w") as f:
        f.write("goodbye\n")
    assert download.fetch(**source, path=tmp_path) == "update"


@responses.activate
def test_fetch_corrupt(tmp_path):
    """Raise an error for a download with an unrecognized checksum."""
    responses.add(method=responses.GET, url=source["url"], body="goodbye\n")
    with pytest.raises(ValueError):
        download.fetch(**source, path=tmp_path)
