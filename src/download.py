# Zachary Gray
#
# pull the figshare data tarballs and extract just the 5 task files we need.
#   python -m src.download              # OT only (101 MB)
#   python -m src.download --all        # also fetch the 2.96 GB 'all' tarball
#   python -m src.download --skip-ot    # only 'all'

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

# (label, url, raw_filename, dir_inside_tar)
BUNDLES = {
    'OT':  ('https://ndownloader.figshare.com/files/14546465',
            'tasks_OT_clr.tar.gz',  'tasks_OT_clr'),
    'all': ('https://ndownloader.figshare.com/files/14546420',
            'tasks_all_clr.tar.gz', 'tasks_all_clr'),
}

# (group, task_name); inside-tar filename is f'{geneset}_clr_{group}_{task_name}.h5'
WANTED_TASKS = [
    ('train',    'COAD_stage'),
    ('train',    'KIRC_stage'),
    ('train',    'LGG_grade'),
    ('train',    'GSE65832'),
    ('validate', 'GSE50244'),
]

# small published replication-target CSV (figshare file 14545823)
RESULTS_CSV_URL = 'https://ndownloader.figshare.com/files/14545823'

DATA_RAW = Path('data/raw')
DATA_EXT = Path('data/extracted')


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'  curl -> {dest}')
    # -C - resumes partial downloads, -L follows redirects
    subprocess.run(
        ['curl', '-sSL', '-o', str(dest), '-C', '-', url],
        check=True,
    )


# pull only the 5 task .h5 files we use
def extract_tasks(tar_path, geneset, out_root):
    inner_dir = BUNDLES[geneset][2]
    wanted = {f'{inner_dir}/{geneset}_clr_{group}_{task}.h5'
              for group, task in WANTED_TASKS}
    out_root.mkdir(parents=True, exist_ok=True)
    print(f'  extracting {len(wanted)} files from {tar_path.name}')
    with tarfile.open(tar_path, 'r:gz') as tf:
        members = [m for m in tf.getmembers() if m.name in wanted]
        if len(members) != len(wanted):
            missing = wanted - {m.name for m in members}
            print(f'  WARNING: missing in tarball: {missing}', file=sys.stderr)
        tf.extractall(path=out_root, members=members)
    extracted_dir = out_root / inner_dir
    sizes = sorted(extracted_dir.glob('*.h5'))
    for p in sizes:
        print(f'    {p.name:50s} {p.stat().st_size//1024} KB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true',
                        help='also fetch the 2.96 GB "all" tarball')
    parser.add_argument('--skip-ot', action='store_true',
                        help='skip OT (only useful with --all)')
    args = parser.parse_args()

    bundles_to_fetch = []
    if not args.skip_ot:
        bundles_to_fetch.append('OT')
    if args.all:
        bundles_to_fetch.append('all')
    if not bundles_to_fetch:
        print('nothing to do (use --all or omit --skip-ot)')
        return

    for geneset in bundles_to_fetch:
        url, raw_name, _ = BUNDLES[geneset]
        raw_path = DATA_RAW / raw_name
        if raw_path.exists():
            print(f'[{geneset}] tarball exists at {raw_path}, skipping download')
        else:
            print(f'[{geneset}] downloading {url}')
            download(url, raw_path)
        extract_tasks(raw_path, geneset, DATA_EXT)

    # also pull the small published-results CSV if missing
    results_csv = Path('data/results_table.csv')
    if not results_csv.exists():
        print(f'[results_table] downloading {RESULTS_CSV_URL}')
        download(RESULTS_CSV_URL, results_csv)


if __name__ == '__main__':
    main()
