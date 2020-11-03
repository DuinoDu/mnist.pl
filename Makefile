
all:

train:
	python scripts/train.py --gpus 0,1,2,3

debug:
	python scripts/train.py --gpus 0,

build:
	python setup.py build

upload:
	python setup.py bdist_wheel upload -r hobot-local

clean:
	@rm -rf build dist src/*.egg-info lightning_logs

test:
	pytest --capture=no

pep8:
	autopep8 src/mnist_pl --recursive -i

lint:
	pylint src/mnist_pl --reports=n

lintfull:
	pylint src/mnist_pl

install:
	python setup.py install

uninstall:
	python setup.py install --record install.log
	cat install.log | xargs rm -rf 
	@rm install.log
