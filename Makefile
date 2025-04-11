# simple makefile to simplify repetitive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python

all: clean

clean-pyc:
	find . -name "*.pyc" | xargs rm -f
	find . -name "__pycache__" | xargs rm -rf

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -fr build
	rm -fr _site

clean: clean-build clean-pyc clean-so

build: clean
	marimo export html-wasm --mode edit poia/poia.py -o _site/notebooks/poia.html

serve:
	python -m http.server --directory _site/notebooks
