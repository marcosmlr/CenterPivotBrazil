#
# project build script.
#

.PHONY: default install clean list
ENV_NAME = `cat environment.yml | grep name: | sed s/name://g | sed -e 's/^[ \t]*//'`

default:        
	@echo $(ENV_NAME)' build script'
	@echo
	@echo 'usage: make <target>'
	@echo
	@echo 'Example: make list'

install:
	@echo
	@echo 'Creating Python environment using conda:'
	conda create -y -n CenterPivotBrazil -c conda-forge python=3.6 matplotlib gdal pandas geopandas numpy scipy scikit-learn opencv rasterio rasterstats cartopy imbalanced-learn

env_export:
	conda env export --no-builds -c conda-forge | cut -d '=' -f1 | grep -v "^prefix: " > environment.yml

clean:
	@echo 'cleaning up temporary files'
	find . -name '*.pyc' -type f -exec rm {} ';'
	find . -name '__pycache__' -type d -print | xargs rm -rf
	@echo 'NOTE: you should clean up the following occasionally (by hand)'
	git clean -fdn
list:
	@cat Makefile | grep "^[a-z]" | grep -v "^cat " | awk '{print $$1}' | sed "s/://g"
