"""General project utilities.

The following top-level variables make it easier to reference specific
directories in other scripts and notebooks:

- :data:`src.utils.ROOT`: The project root (usually named ``rcp2/``).
- :data:`src.utils.DATA`: The project data directory and subdirectories.

.. note::
   The ``DATA`` variable assumes that you have downloaded and set up the data
   directory as follows: ::

     Data/
     ├── 03_Shapefiles
     │   ├── 2010_Census_shapefiles
     │   └── SVI2016_US_shapefiles
     ├── Master Project Data
     ├── processed
     └── raw

"""
import pathlib


# The project root diretory.
ROOT = pathlib.Path(__file__, "..", "..").resolve()


# Paths to project data directories.
DATA = {
    "data": ROOT / "Data",
    "raw": ROOT / "Data" / "raw",
    "interim": ROOT / "Data" / 'interim',
    "master": ROOT / "Data" / "Master Project Data",
    "processed": ROOT / "Data" / "processed",
    "shapefiles": ROOT / "Data" / "03_Shapefiles",
    "shapefiles-census": ROOT / "Data" / "03_Shapefiles" / "2010_Census_shapefiles",
    "shapefiles-svi": ROOT / "Data" / "03_Shapefiles" / "SVI2016_US_shapefiles",
}