"""Fetch attributable public source snapshots; never execute downloaded code."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import urllib.error
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    'ipc_dehaze': ('Jiayi-Fu/IPC-Dehaze', 'main'),
    'sfhformer': ('deng-ai-lab/SFHformer', 'main'),
    'sfmn': ('it-hao/SFMN', 'main'),
}
PAPERS = {
    'ipc_dehaze': 'https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_Iterative_Predictor-Critic_Code_Decoding_for_Real-World_Image_Dehazing_CVPR_2025_paper.pdf',
    'sfhformer': 'https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06190.pdf',
    'sfmn': 'https://www.haoshen.site/resources/ShenHao_SFMN_TIP_2025.pdf',
}

def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'ROAS5800-coursework-research/1.0'})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()

def fetch_repo(key: str) -> dict:
    repo, branch = REPOS[key]
    dest = ROOT / 'sources' / key
    metadata = ROOT / 'sources' / (key + '.json')
    expected = json.loads(metadata.read_text(encoding='utf-8')) if metadata.exists() else None
    if metadata.exists() and (dest / 'README.md').exists():
        for file, expected_hash in (expected.get('file_inventory') or {}).items():
            # Authors' archives include obsolete Python 3.6/3.7 bytecode.
            # It is not an executable dependency of this Python 3.12 project.
            if '__pycache__' in Path(file).parts or Path(file).suffix == '.pyc':
                continue
            local = dest / file
            if not local.is_file() or hashlib.sha256(local.read_bytes()).hexdigest() != expected_hash:
                raise ValueError('Source differs from tested official snapshot: ' + str(local))
        return expected
    if expected and expected.get('commit'):
        commit = expected['commit']
    else:
        try:
            commit = json.loads(get(f'https://api.github.com/repos/{repo}/commits/{branch}'))['sha']
        except urllib.error.HTTPError as error:
            if error.code not in (403, 429):
                raise
            commit = None
    revision = commit or ('refs/heads/' + branch)
    url = f'https://codeload.github.com/{repo}/zip/{revision}'
    archive = get(url)
    if expected and hashlib.sha256(archive).hexdigest() != expected['archive_sha256']:
        raise ValueError('Official source archive differs from the tested snapshot: ' + repo)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            relative = Path(*Path(info.filename).parts[1:])
            target = (dest / relative).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError('Unsafe archive path')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))
    record = {'repository': 'https://github.com/' + repo, 'commit': commit,
              'branch': branch, 'revision_note': 'Exact archive preserved by SHA-256; commit API may be rate limited',
              'archive_url': url, 'archive_sha256': hashlib.sha256(archive).hexdigest(),
              'archive_bytes': len(archive)}
    if expected:
        record = expected
    metadata.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record

def fetch_paper(key: str) -> dict:
    dest = ROOT / 'papers' / (key + '.pdf')
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        content = get(PAPERS[key])
        if not content.startswith(b'%PDF'):
            raise ValueError('Not a PDF: ' + PAPERS[key])
        dest.write_bytes(content)
    return {'paper': key, 'url': PAPERS[key], 'bytes': dest.stat().st_size,
            'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repos', action='store_true')
    parser.add_argument('--papers', action='store_true')
    parser.add_argument('--ipc-weights', action='store_true')
    args = parser.parse_args()
    jobs = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        if args.repos:
            jobs.extend((key, pool.submit(fetch_repo, key)) for key in REPOS)
        if args.papers:
            jobs.extend((key, pool.submit(fetch_paper, key)) for key in PAPERS)
        if args.ipc_weights:
            jobs.extend((key, pool.submit(fetch_ipc_weight, key)) for key in ('predictor', 'critic'))
        for key, future in jobs:
            try:
                result = future.result()
                if 'file_inventory' in result:
                    result = {'repository': result['repository'], 'commit': result['commit'],
                              'verified_source_files': sum('__pycache__' not in Path(f).parts and Path(f).suffix != '.pyc'
                                                           for f in result['file_inventory'])}
                print(json.dumps(result), flush=True)
            except Exception as error:
                print(json.dumps({'key': key, 'error': str(error)}), flush=True)
                errors.append(key)
    if errors:
        raise SystemExit(1)

def fetch_ipc_weight(key: str) -> dict:
    import gdown
    ids = {'predictor': '1RIjzFXhqotRwaPjS0hf_BJ9RbiLR4D0F',
           'critic': '1NMK7PEeFY1cN2Eytxlv_UEVlIn4zMsX9'}
    expected_hashes = {'predictor': 'eb6019e01f28c6de2c0c1bc5ece283d9aa730c839cf274091d4d45d376717c53',
                       'critic': '321e98ad60fe1e7ced27904ba81d1cfe4dbe31a334d22b678dfae6f3be0c96d6'}
    dest = ROOT / 'checkpoints' / 'ipc_dehaze' / (key + '.pth')
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        last = [0]
        def progress(current, total):
            if current - last[0] >= 64 * 1024 * 1024:
                print(json.dumps({'download': key, 'MB': round(current / 2**20),
                                  'total_MB': round(total / 2**20) if total else None}), flush=True)
                last[0] = current
        partial = dest.with_suffix('.pth.partial')
        gdown.download(id=ids[key], output=str(partial), use_cookies=False, quiet=True, resume=True,
                       progress=progress, timeout=90, retries=2)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != expected_hashes[key]:
            raise ValueError('Downloaded weight does not match the tested official release: ' + key)
        partial.replace(dest)
    if hashlib.sha256(dest.read_bytes()).hexdigest() != expected_hashes[key]:
        raise ValueError('Existing weight does not match the tested official release: ' + key)
    record = {'file': str(dest.relative_to(ROOT)), 'drive_id': ids[key],
              'folder_provenance': 'https://drive.google.com/drive/folders/1i5DS0jjTlm1nSXSAPd4Qp5_7dLSva_1E',
              'bytes': dest.stat().st_size,
              'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    dest.with_suffix('.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record

if __name__ == '__main__':
    main()
