#!/usr/bin/env python3
import os
import sys
import zipfile
from pathlib import Path

sys.path.append(os.path.abspath('/home/pengfei/Downloads/dynaip'))

import torch
from utils.read_data import read_mvnx, read_xlsx

ROOT = Path('/home/pengfei/Downloads/dynaip')
RAW = ROOT / 'datasets' / 'raw'
OUT = ROOT / 'datasets' / 'extract'


def save_pt(dataset, src_path, data):
    OUT.joinpath(dataset).mkdir(parents=True, exist_ok=True)
    stem = src_path.name
    if stem.endswith('.xsens.mvnx'):
        stem = stem.replace('.xsens.mvnx', '.pt')
    elif stem.endswith('.mvnx'):
        stem = stem.replace('.mvnx', '.pt')
    elif stem.endswith('.xlsx'):
        parent = src_path.parent.name
        stem = stem.replace('.xlsx', f'_{parent}.pt')
    else:
        stem = src_path.stem + '.pt'
    out = OUT / dataset / stem
    torch.save(data, out)
    print(f'[save] {dataset} {out.name}')



def unzip_virginia():
    virginia = RAW / 'virginia'
    extracted = virginia / 'extracted'
    extracted.mkdir(parents=True, exist_ok=True)
    for zip_path in sorted(virginia.glob('*.zip')):
        marker = extracted / (zip_path.stem + '.done')
        if marker.exists():
            continue
        print(f'[unzip] virginia {zip_path.name}')
        target = extracted / zip_path.stem
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        marker.write_text('done\n')

def extract_mvnx_dataset(dataset):
    paths = sorted((RAW / dataset).rglob('*.mvnx'))
    print(f'[dataset] {dataset}: {len(paths)} mvnx')
    ok = fail = 0
    for path in paths:
        out_name = path.name.replace('.xsens.mvnx', '.pt').replace('.mvnx', '.pt')
        if (OUT / dataset / out_name).exists():
            ok += 1
            continue
        try:
            data = read_mvnx(str(path))
            save_pt(dataset, path, data)
            ok += 1
        except Exception as exc:
            fail += 1
            print(f'[skip] {dataset} {path}: {type(exc).__name__}: {exc}')
    return {'ok': ok, 'fail': fail, 'total': len(paths)}


def extract_cip():
    paths = sorted((RAW / 'cip').rglob('*.xlsx'))
    paths = [p for p in paths if p.name.lower() != 'protocol.xlsx']
    print(f'[dataset] cip: {len(paths)} xlsx')
    ok = fail = 0
    for path in paths:
        out_name = path.name.replace('.xlsx', f'_{path.parent.name}.pt')
        if (OUT / 'cip' / out_name).exists():
            ok += 1
            continue
        try:
            data = read_xlsx(str(path))
            save_pt('cip', path, data)
            ok += 1
        except Exception as exc:
            fail += 1
            print(f'[skip] cip {path}: {type(exc).__name__}: {exc}')
    return {'ok': ok, 'fail': fail, 'total': len(paths)}


def main():
    stats = {}
    unzip_virginia()
    for dataset in ['andy', 'emokine', 'unipd', 'virginia']:
        stats[dataset] = extract_mvnx_dataset(dataset)
    stats['cip'] = extract_cip()
    print('[done]', stats)


if __name__ == '__main__':
    main()
