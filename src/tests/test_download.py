import pooch
import pytest
import responses
from src.data import download


@responses.activate
def test_fetch_download(tmp_path):
    """Download a file when no local copy exists."""
    url = "https://my-data-source.com/data.txt"
    sha256 = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    responses.add(method=responses.GET, url=url, body="hello\n")
    sources = {"data.txt": {"sha256": sha256, "url": url, "processor": "declare_action"}}
    path = tmp_path
    assert download.fetch("data.txt", path, sources)[1] == "download"
    
    
@responses.activate
def test_fetch_fetch(tmp_path):
    """Fetch a local file when it exists."""
    url = "https://my-data-source.com/data.txt"
    sha256 = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    responses.add(method=responses.GET, url=url, body="hello\n")
    sources = {"data.txt": {"sha256": sha256, "url": url, "processor": "declare_action"}}
    path = tmp_path
    assert download.fetch("data.txt", path, sources)[1] == "download"
    assert download.fetch("data.txt", path, sources)[1] == "fetch"
    
    
@responses.activate
def test_fetch_update(tmp_path):
    """Fetch a local file when it exists."""
    url = "https://my-data-source.com/data.txt"
    sha256 = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    responses.add(method=responses.GET, url=url, body="hello\n")
    sources = {"data.txt": {"sha256": sha256, "url": url, "processor": "declare_action"}}
    path = tmp_path
    with open(path / "data.txt", "w") as f:
        f.write("goodbye\n")
    assert download.fetch("data.txt", path, sources)[1] == "update"


@responses.activate
def test_fetch_corrupt(tmp_path):
    """Fetch a local file when it exists."""
    url = "https://my-data-source.com/data.txt"
    sha256 = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    responses.add(method=responses.GET, url=url, body="goodbye\n")
    sources = {"data.txt": {"sha256": sha256, "url": url, "processor": "declare_action"}}
    path = tmp_path
    with pytest.raises(ValueError):
        download.fetch("data.txt", path, sources)[1]
