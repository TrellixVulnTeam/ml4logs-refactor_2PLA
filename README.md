## How to test package (100k lines version)

1. Activate your virtual environment (conda, venv)
2. Install requirements using `make requirements`
3. Change `BASH_INTERPRETER` variable in `Makefile` (if needed)
4. Run `make {COMMAND_NAME}`

### Possible command names

- `hdfs1_fasttext_100k`
- `hdfs1_drain_100k`
- `bgl_100k`
- `thunderbird_100k`
