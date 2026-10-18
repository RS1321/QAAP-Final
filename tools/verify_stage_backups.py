"""Verify downloaded stage parts, then optionally join them into ordinary ZIPs.

Usage: python verify_stage_backups.py DOWNLOAD_FOLDER [--join]
Does not execute notebook code or unpickle models. Originals are kept intact.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from zipfile import ZipFile

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--join', action='store_true')
    args = parser.parse_args()
    manifests = sorted(args.folder.glob('QAAP_backup_*.manifest.json'))
    if not manifests:
        raise SystemExit('No stage manifests found in that folder.')
    for manifest in manifests:
        record = json.loads(manifest.read_text())
        parts = []
        for entry in record['parts']:
            name = entry['name']
            if Path(name).name != name:
                raise ValueError('Invalid part filename')
            path = manifest.parent/name
            if not path.is_file() or path.stat().st_size != entry['bytes']:
                raise ValueError(f'Missing or incomplete download: {path}')
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024**2), b''):
                    digest.update(block)
            if digest.hexdigest() != entry['sha256']:
                raise ValueError(f'Checksum mismatch: {path}')
            parts.append(path)
        print('VERIFIED:', record['label'], f'({len(parts)} parts)')
        if args.join:
            destination = manifest.with_name(manifest.name.replace('.manifest.json','.zip'))
            if destination.exists():
                raise FileExistsError(f'Will not overwrite {destination}')
            temporary = destination.with_suffix('.zip.tmp')
            with temporary.open('xb') as output:
                for path in parts:
                    with path.open('rb') as stream:
                        shutil.copyfileobj(stream, output, length=8*1024**2)
            with ZipFile(temporary) as archive:
                bad = archive.testzip()
                if bad is not None:
                    raise ValueError('ZIP CRC failure: '+bad)
            temporary.replace(destination)
            print('JOINED and CRC checked:', destination)

if __name__ == '__main__':
    main()
