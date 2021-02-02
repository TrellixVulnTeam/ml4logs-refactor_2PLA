.PHONY: requirements hdfs1_100k


PROJECT_NAME = ml4logs
export PROJECT_DIR = $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
export PYTHON_INTERPRETER = python

# Change this to your bash interpreter (ex. /bin/bash) or to "sbatch" command
BASH_INTERPRETER = bash


requirements:
	# $(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r "$(PROJECT_DIR)/requirements.txt"


hdfs1_100k:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/hdfs1_100k.batch"
