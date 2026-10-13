.PHONY: install test export preprocess train predict change map all
install:
	pip install -r requirements.txt && pip install -e .
test:
	pytest -q
export:
	python -m deforestation.gee_download --config config.yaml
preprocess:
	python -m deforestation.preprocess --config config.yaml
train:
	python -m deforestation.train --config config.yaml
predict:
	python -m deforestation.predict --config config.yaml
change:
	python -m deforestation.change_detection --config config.yaml
map:
	python -m deforestation.visualize --config config.yaml
# After the Earth Engine exports are downloaded into data/raw/
all: preprocess train predict change map
