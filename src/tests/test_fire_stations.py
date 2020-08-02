import geopandas
import numpy as np
import shapely
from src.data import fire_stations


def test_calculate_points():
    lon = np.random.uniform(0, 90, size=2)
    lat = np.random.uniform(-180, 180, size=2)
    result = fire_stations.calculate_points(lon, lat)
    assert type(result) == geopandas.GeoSeries
    assert result.shape == (2,)
    assert type(result.iloc[0]) == shapely.geometry.point.Point