"""Recover exact frozen COCO image payloads from the official storage bucket."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen

from public.workloads.datasets.identity import load_tsv, sha256

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--limit",type=int)
    args = parser.parse_args()
    records = json.loads((ROOT/"data/manifests/index.json").read_text())["records"]
    jobs = []
    for key,split in (("coco_evaluation_5k","val2017"),("coco_calibration_2k","train2017")):
        record = records[key]
        path = ROOT/record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError("frozen COCO list changed")
        rows = sorted(load_tsv(path),key=lambda row:row["sha256"])
        if args.limit:
            rows = rows[:args.limit]
        for row in rows:
            destination = ROOT/"data/raw"/record["logical_payload_root"]/row["file_name"]
            if not destination.resolve().is_relative_to((ROOT/"data/raw").resolve()):
                raise ValueError("invalid image destination")
            jobs.append((row,split,destination))

    def download(job):
        row,split,path = job
        if path.exists():
            if sha256(path) != row["sha256"]:
                raise ValueError(f"existing frozen image changed: {path}")
            return
        url = f"https://s3.amazonaws.com/images.cocodataset.org/{split}/{row['file_name']}"
        for attempt in range(4):
            try:
                with urlopen(url,timeout=45) as stream:
                    content = stream.read()
                if hashlib.sha256(content).hexdigest() != row["sha256"]:
                    raise ValueError(f"official image does not match frozen hash: {row['file_name']}")
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary = path.with_suffix(".partial")
                temporary.write_bytes(content)
                temporary.replace(path)
                return
            except (OSError,TimeoutError):
                if attempt == 3:
                    raise
                time.sleep(attempt+1)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download,job) for job in jobs]
        for count,future in enumerate(as_completed(futures),1):
            future.result()
            if count % 100 == 0 or count == len(jobs):
                print(f"verified {count}/{len(jobs)} frozen COCO images",flush=True)


if __name__ == "__main__":
    main()
