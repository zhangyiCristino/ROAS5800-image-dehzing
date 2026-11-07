"""Reuse a completed run only when its actual image files are available."""
import hashlib
import json
from pathlib import Path


def verified_completed_summary(directory):
    directory=Path(directory)
    record_file=directory/'results.json'
    if not record_file.is_file():return None
    record=json.loads(record_file.read_text(encoding='utf-8'))
    rows=record.get('rows',[])
    if not (record.get('complete') and record.get('summary',{}).get('count')==352
            and len(rows)==352 and len({r['id'] for r in rows})==352):
        return None
    missing=False
    for row in rows:
        file=directory/(row['id']+'.png')
        if not file.is_file():
            missing=True
            continue
        if hashlib.sha256(file.read_bytes()).hexdigest()!=row['output_sha256']:
            raise ValueError('Existing output hash mismatch: '+str(file))
    # A fresh clone contains published measurements but intentionally omits
    # full-resolution output files. It must compute these images locally.
    return None if missing else record['summary']
