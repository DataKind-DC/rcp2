from src import utils


def test_root():
    assert utils.ROOT.exists()
    
    
def test_data():
    for path in utils.DATA.values():
        assert path.exists()