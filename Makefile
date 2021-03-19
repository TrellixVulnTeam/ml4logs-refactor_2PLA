.PHONY: requirements hdfs1_fasttext_100k hdfs1_drain_100k bgl_100k thunderbird_100k


PROJECT_NAME = ml4logs
export PROJECT_DIR = $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
export PYTHON_INTERPRETER = python

# Change this to your bash interpreter (ex. /bin/bash) or to "sbatch" command
BASH_INTERPRETER = bash


requirements:
	$(PYTHON_INTERPRETER) -m pip install -r "$(PROJECT_DIR)/requirements.txt"


all: hdfs1_fasttext_seq2seq_100k hdfs1_fasttext_loglizer_100k hdfs1_drain_100k bgl_100k

hdfs1_fasttext_seq2seq_100k:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/hdfs1_fasttext_seq2seq_100k.batch"

hdfs1_fasttext_loglizer_100k:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/hdfs1_fasttext_loglizer_100k.batch"

hdfs1_drain_100k:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/hdfs1_drain_100k.batch"

bgl_100k:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/bgl_100k.batch"

thunderbird_100k:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/thunderbird_100k.batch"
