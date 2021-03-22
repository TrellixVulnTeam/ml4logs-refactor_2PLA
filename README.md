## Usage

1. Activate your virtual environment (conda, venv)
2. Install requirements using `make requirements`
3. Change `BASH_INTERPRETER` variable in `Makefile` (if needed)
4. Run `make {COMMAND_NAME}`

## Run benchmark on HDFS1

- `make hdfs1_data_all` (or `make hdfs1_data_100k`)
- wait
- `make hdfs1_preprocess`
- wait
- `make hdfs1_train_test`
