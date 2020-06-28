import geopandas
import pytest
from src.data import raw


def test_read_shapefiles_states():
    df = raw.read_shapefiles(states=["AL", "AK"])
    assert type(df) == geopandas.GeoDataFrame
    assert df.shape == (1348, 13)
    assert df.STATEFP10.isin(["01", "02"]).all()

    
def test_read_shapefiles_fips():
    df = raw.read_shapefiles(fips=["01", "02"])
    assert type(df) == geopandas.GeoDataFrame
    assert df.shape == (1348, 13)
    assert df.STATEFP10.isin(["01", "02"]).all()

    
def test_read_shapefiles_glob():
    df = raw.read_shapefiles(glob="tl_2010_0[12]*.shp")
    assert type(df) == geopandas.GeoDataFrame
    assert df.shape == (1348, 13)
    assert df.STATEFP10.isin(["01", "02"]).all()

    
def test_read_shapefiles_errors():
    with pytest.raises(ValueError):
        raw.read_shapefiles(states=["AL"], fips=["01"])
    with pytest.raises(raw.BadPathError):
        raw.read_shapefiles(fips=["ZZ"])
    with pytest.raises(raw.BadPathError):
        raw.read_shapefiles(glob="*")