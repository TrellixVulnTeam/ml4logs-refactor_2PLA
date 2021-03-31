## Usage

1. Activate your virtual environment (conda, venv)
2. Install requirements using `make requirements`
3. Change `BASH_INTERPRETER` variable in `Makefile` (if needed)
4. Run `make {COMMAND_NAME}`

### Run benchmark on HDFS1 (100k lines)

- `make hdfs1_100k_data`
- wait
- `make hdfs1_100k_preprocess`
- wait
- `make hdfs1_100k_train_test`

### Run benchmark on HDFS1

- `make hdfs1_data`
- wait
- `make hdfs1_preprocess`
- wait
- `make hdfs1_train_test`

## Description of scripts and configs

- Each script executes only one config
- Config describes pipeline of actions which are applied to our data

### Scripts/configs

data

- Download archive
- Extract archive
- Prepare dataset

drain_preprocess

- Parse using IBM/Drain3
- Aggregate by block using count vector

fasttext_preprocess

- Train fasttext model
- Get embeddings for all log lines
- Merge with timedeltas
- Aggregate by block using selected method (sum, average, min, max)

drain_loglizer

- Train and test models which are used by loglizer on drain parsed dataset
  - Logistic regression
  - Decision tree
  - Linear SVC
  - LOF
  - One class SVM
  - Isolation forest

fasttext_loglizer

- Train and test models which are used by loglizer on aggregated fasttext embeddings
  - Logistic regression
  - Decision tree
  - Linear SVC
  - LOF
  - One class SVM
  - Isolation forest
  - PCA

fasttext_seq2seq

- Train and test sequence model
- Try to predict next embedding using LSTM based torch model
- Compute threshold on train dataset (assume 5% logs are anomalies)
- Test on different thresholds and save statistics
