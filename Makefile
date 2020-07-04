SHELL=/bin/bash

# Activate the project conda environment.
# Inspired by https://stackoverflow.com/a/55696820.
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh; conda activate rcp2

# Download the project source data.
download :
	 $(CONDA_ACTIVATE); python -m src.data.download
