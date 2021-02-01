def prepare_hdfs_2(args):
    xdir = ml4logs.DATASETS[name]['xdir']
    log_files = sorted(list(xdir.glob('*.log')))
    logs_path = ml4logs.DATASETS[name]['logs_path']
    logger.info('Start concatenating %d files', len(log_files))
    with logs_path.open('wb') as f_out:
        for in_path in log_files:
            with in_path.open('rb') as f_in:
                shutil.copyfileobj(f_in, f_out)
                f_in.seek(-1, io.SEEK_END)
                if f_in.read() != b'\n':
                    f_out.write(b'\n')
            logger.info('File %s is appended', in_path)
    for in_path in log_files:
        in_path.unlink()
    logger.info('Removed all input files')


LOG_N_LINES = 500 * 10**3


def prepare_bgl(args):
    xdir = ml4logs.DATASETS[name]['xdir']
    in_path = xdir / 'BGL.log'
    labels = []
    logs = []
    logger.info('Start splitting labels and log messages')
    with in_path.open(encoding='utf-8') as f:
        for line in f:
            label, log = tuple(line.split(maxsplit=1))
            labels.append('1' if label != '-' else '0')
            logs.append(log.strip())
            if len(logs) % LOG_N_LINES <= 0:
                logger.info('Processed %d lines', len(logs))
    logger.info('Saving into separate files')
    labels_path = ml4logs.DATASETS[name]['labels_path']
    labels_path.write_text('\n'.join(labels))
    logger.info('Saved labels into %s', labels_path)
    logs_path = ml4logs.DATASETS[name]['logs_path']
    logs_path.write_text('\n'.join(logs), encoding='utf-8')
    logger.info('Saved logs into %s', logs_path)
    in_path.unlink()
    logger.info('Removed input file %s', in_path)
