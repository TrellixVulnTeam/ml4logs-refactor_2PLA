.PHONY: requirements hdfs1_fasttext_100k hdfs1_drain_100k bgl_100k thunderbird_100k


PROJECT_NAME = ml4logs
export PROJECT_DIR = $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
export PYTHON_INTERPRETER = python

# Change this to your bash interpreter (ex. /bin/bash) or to "sbatch" command
BASH_INTERPRETER = bash


requirements:
	$(PYTHON_INTERPRETER) -m pip install -r "$(PROJECT_DIR)/requirements.txt"


all: hdfs1_fasttext_seq2seq_100k hdfs1_fasttext_loglizer_100k hdfs1_drain_100k bgl_100k


hdfs1_data_100k:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1/data_100k.batch"

hdfs1_data_all:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1/data.batch"

hdfs1_preprocess:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1/drain_preprocess.batch"
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1/fasttext_preprocess.batch"

hdfs1_train_test:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1/drain_loglizer.batch"
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1/fasttext_loglizer.batch"
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1/fasttext_seq2seq.batch"

bgl_100k:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/bgl_100k.batch"

thunderbird_100k:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/thunderbird_100k.batch"
