PROJECT_NAME = ml4logs
export PROJECT_DIR = $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
export PYTHON_INTERPRETER = python

# Change this to your bash interpreter (ex. /bin/bash) or to "sbatch" command
BASH_INTERPRETER = bash


.PHONY: requirements \
		hdfs1_100k_data hdfs1_100k_preprocess hdfs1_100k_train_test \
		hdfs1_data hdfs1_preprocess hdfs1_train_test \
		bgl_100k thunderbird_100k


requirements:
	$(PYTHON_INTERPRETER) -m pip install -r "$(PROJECT_DIR)/requirements.txt"

hdfs1_100k_data:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1_100k/data.batch"

hdfs1_100k_preprocess:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1_100k/drain_preprocess.batch"
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1_100k/fasttext_preprocess.batch"

hdfs1_100k_train_test:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1_100k/drain_loglizer.batch"
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1_100k/fasttext_loglizer.batch"
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1_100k/fasttext_seq2seq.batch"
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1_100k/fasttext_seq2label.batch"

hdfs1_data:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1/data.batch"

hdfs1_preprocess:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1/drain_preprocess.batch"
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1/fasttext_preprocess.batch"

hdfs1_train_test:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1/drain_loglizer.batch"
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1/fasttext_loglizer.batch"
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1/fasttext_seq2seq.batch"
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/HDFS1_100k/fasttext_seq2label.batch"

bgl_100k:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/bgl_100k.batch"

thunderbird_100k:
	$(BASH_INTERPRETER) "$(PROJECT_DIR)/scripts/thunderbird_100k.batch"
