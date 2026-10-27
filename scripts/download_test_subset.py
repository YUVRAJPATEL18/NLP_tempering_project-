"""Fetch selected ZIP members with HTTP ranges, without downloading the full archive."""
import argparse
import io
import json
import hashlib
from pathlib import Path
import zipfile
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = 'AmAFakePerson123/TrialforGeneratedIDs'
REVISION = 'c1da4db'
URL = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/dataset.zip'


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.session = requests.Session()
        self.downloaded = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, count=-1):
        count = min(self.size-self.position, count if count >= 0 else self.size)
        if count <= 0:
            return b''
        start, end = self.position, self.position + count - 1
        # Distinct URLs prevent an intermediary from reusing another byte range.
        response = self.session.get(self.url + f'?range={start}-{end}',
                                    headers={'Range': f'bytes={start}-{end}'}, timeout=90, stream=True)
        with response:
            response.raise_for_status()
            if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                raise RuntimeError('Server did not honor range; refusing full-archive download')
            data = response.content
        if len(data) != count:
            raise RuntimeError('Truncated range response')
        self.position += len(data)
        self.downloaded += len(data)
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list-only', action='store_true')
    parser.add_argument('--members', nargs='*')
    args = parser.parse_args()
    directory = ROOT / 'datasets' / 'midv2020_subset'
    directory.mkdir(parents=True, exist_ok=True)
    remote = RemoteZip(URL, 3081583683)
    with zipfile.ZipFile(remote) as archive:
        entries = [dict(name=i.filename, size=i.file_size) for i in archive.infolist() if not i.is_dir()]
        (directory / 'archive_index.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')
        if args.list_only:
            print(json.dumps(entries[:35], indent=2))
            print(f'{len(entries)} files; index downloaded using {remote.downloaded} bytes')
            return
        selected = args.members or []
        if not selected:
            selected = [f'dataset/{capture}/images/{kind}/{number:02d}.jpg'
                        for kind in ('alb_id', 'fin_id') for number in range(3)
                        for capture in ('templates', 'scan_upright', 'scan_rotated')]
            selected += [f'dataset/{capture}/annotations/{kind}.json'
                         for kind in ('alb_id', 'fin_id')
                         for capture in ('templates', 'scan_upright', 'scan_rotated')]
        manifest = []

        def download(member):
            info = archive.getinfo(member)
            if info.file_size > 15_000_000:
                raise ValueError('Refusing unexpectedly large image')
            # Flat output filenames avoid archive traversal and preserve original path in manifest.
            filename = member.replace('/', '__').replace('\\', '__').replace(':', '_')
            destination = directory / filename
            transferred = 0
            if not destination.exists():
                reader = RemoteZip(URL, remote.size)
                reader.seek(info.header_offset)
                # One range includes the complete local header and compressed member.
                block = reader.read(info.compress_size + 131102)
                if block[:4] != b'PK\x03\x04':
                    raise RuntimeError('Invalid local ZIP header')
                name_length, extra_length = struct.unpack_from('<HH', block, 26)
                start = 30 + name_length + extra_length
                compressed = block[start:start + info.compress_size]
                if info.compress_type == zipfile.ZIP_DEFLATED:
                    payload = zlib.decompress(compressed, -15)
                elif info.compress_type == zipfile.ZIP_STORED:
                    payload = compressed
                else:
                    raise RuntimeError('Unsupported ZIP compression')
                if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
                    raise RuntimeError('ZIP member integrity check failed')
                destination.write_bytes(payload)
                transferred = reader.downloaded
            return dict(source_member=member, file=filename, bytes=destination.stat().st_size,
                        sha256=hashlib.sha256(destination.read_bytes()).hexdigest()), transferred

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(download, member) for member in selected]
            for future in as_completed(futures):
                entry, transferred = future.result()
                manifest.append(entry)
                remote.downloaded += transferred
                print(f'Saved {entry["file"]}', flush=True)
        (directory / 'download_manifest.json').write_text(json.dumps(dict(
            source=URL, publisher='Community mirror; provenance not independently authenticated',
            license_declared_by_mirror='cc-by-sa-3.0', files=manifest,
            transferred_bytes=remote.downloaded), indent=2), encoding='utf-8')
    print(f'Transferred {remote.downloaded} bytes; full archive was not downloaded')


if __name__ == '__main__':
    main()
