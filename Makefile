.PHONY: requirements data fasttext mkdirs


PROJECT_NAME = ml4logs
export PROJECT_DIR = $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
export PYTHON_INTERPRETER = python
DOWNLOAD_DATASETS = "BGL" "HDFS_1" "HDFS_2"
EXTRACT_DATASETS = "BGL" #"HDFS_1"
PREPARE_DATASETS = "BGL" #"HDFS_1"
export FASTTEXT_DATASETS = "BGL"
MODELS = "DecisionTree" "LogisticRegression" # "AutoEncoder" "VAE"


requirements:
	# $(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r "$(PROJECT_DIR)/requirements.txt"

data:
	$(PYTHON_INTERPRETER) -m ml4logs.data download $(DOWNLOAD_DATASETS)
	@echo ""
	$(PYTHON_INTERPRETER) -m ml4logs.data extract $(EXTRACT_DATASETS) --mkdir
	@echo ""
	$(PYTHON_INTERPRETER) -m ml4logs.data prepare $(PREPARE_DATASETS)


DUMMY_DATASET = "DUMMY"
export FASTTEXT_DATASETS = "DUMMY"

dummy_fasttext: mkdirs
	# sbatch --output=$(PROJECT_DIR)/logs/fasttext-%j.out "$(PROJECT_DIR)/scripts/train_fasttext.batch"
	"C:\Users\nikol\scoop\apps\msys2\current\usr\bin\bash.exe" "$(PROJECT_DIR)/scripts/train_fasttext.batch"
	# $(PYTHON_INTERPRETER) -m ml4logs.models train_fasttext $(DUMMY_DATASET) --mkdir --model_path models/dummy.bin

mkdirs:
	mkdir -p "$(PROJECT_DIR)/logs"