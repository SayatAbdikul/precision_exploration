"""Recover frozen ImageNet images from the public official ILSVRC archives.

Only matching frozen image payloads are written. Training uses byte ranges
for each nested synset tar, and stops a stream once its required images have
been verified. Neither complete archive is stored on disk.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import tarfile
import time
import re
import os

import requests
from urllib3.exceptions import HTTPError as UrllibHTTPError

from public.workloads.datasets.identity import load_tsv,sha256

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_"
SIZES = {"val":6744924160,"train":147897477120}
INDEX_DIR = ROOT/"artifacts/dataset_indexes"
VALIDATION_CHUNK_BYTES = 32*1024*1024


def payload_name(row):
    # The HF packaging appends the class WNID to the original archive name.
    name = row["source_name"]
    return name.rsplit("_",1)[0]+".JPEG"


def target_rows(split):
    index = json.loads((ROOT/"data/manifests/index.json").read_text())["records"]
    record = index["imagenet_calibration_2k" if split == "train" else "imagenet_evaluation_10k"]
    path = ROOT/record["path"]
    if sha256(path) != record["sha256"]:
        raise ValueError("frozen ImageNet list hash mismatch")
    rows = load_tsv(path)
    if len(rows) != record["count"]:
        raise ValueError("frozen ImageNet list count mismatch")
    root = ROOT/"data/raw"/record["logical_payload_root"]
    selected = {}
    for row in rows:
        destination = root/row["relative_path"]
        if not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError("image destination escapes frozen payload root")
        if destination.exists():
            if sha256(destination) != row["sha256"]:
                raise ValueError(f"existing image hash mismatch: {destination}")
        else:
            selected[payload_name(row)] = row
    return selected,root


def get_range(session,split,start,end):
    response = session.get(BASE+split+".tar",headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"},stream=True,timeout=(20,120))
    response.raise_for_status()
    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZES[split]}":
        response.close()
        raise ValueError("official archive server did not honor the requested byte range")
    return response


def recover_stream(split,start,size,selected,root):
    remaining = dict(selected)
    restored = 0
    for attempt in range(4):
        if not remaining:
            return restored
        try:
            with requests.Session() as session, get_range(session,split,start,start+size-1) as response:
                with tarfile.open(fileobj=response.raw,mode="r|") as archive:
                    for member in archive:
                        name = Path(member.name).name
                        if name not in remaining:
                            continue
                        if not member.isfile():
                            raise ValueError("frozen image points to a non-file tar member")
                        row = remaining[name]
                        data = archive.extractfile(member).read()
                        if hashlib.sha256(data).hexdigest() != row["sha256"]:
                            raise ValueError(f"official archive image differs from frozen bytes: {name}")
                        destination = root/row["relative_path"]
                        destination.parent.mkdir(parents=True,exist_ok=True)
                        temporary = destination.with_suffix(".partial")
                        temporary.write_bytes(data)
                        temporary.replace(destination)
                        del remaining[name]
                        restored += 1
                        if split == "val" and restored%100 == 0:
                            print(f"restored {restored}/{len(selected)} frozen validation images",flush=True)
                        if not remaining:
                            return restored
                raise ValueError(f"archive ended before frozen images were found: {sorted(remaining)[:3]}")
        except (requests.RequestException,OSError,tarfile.ReadError) as error:
            if attempt == 3:
                raise
            print(f"retrying {split} range {start}: {type(error).__name__}",flush=True)
            time.sleep(attempt+1)
    return restored


def training_ranges():
    cache = ROOT/"artifacts/dataset_indexes/imagenet-train-tar-index.json"
    records = json.loads(cache.read_text()) if cache.exists() else []
    for record in records:
        yield record
    position = 0 if not records else records[-1]["start"]+((records[-1]["size"]+511)//512)*512
    with requests.Session() as session:
        while position+512 <= SIZES["train"]:
            with get_range(session,"train",position,position+511) as response:
                header = response.content
            if header == bytes(512):
                break
            member = tarfile.TarInfo.frombuf(header,"utf-8","surrogateescape")
            if not member.isfile() or not member.name.endswith(".tar"):
                raise ValueError("unexpected training archive member")
            record = {"name":member.name,"start":position+512,"size":member.size}
            records.append(record)
            position = record["start"]+((member.size+511)//512)*512
            cache.parent.mkdir(parents=True,exist_ok=True)
            temporary = cache.with_suffix(".partial")
            temporary.write_text(json.dumps(records)+"\n")
            temporary.replace(cache)
            yield record


class PrefixReader:
    def __init__(self,prefix,stream):
        self.prefix,self.stream = prefix,stream

    def read(self,size=-1):
        if size < 0:
            result,self.prefix = self.prefix,b""
            return result+self.stream.read()
        prefix,self.prefix = self.prefix[:size],self.prefix[size:]
        return prefix+self.stream.read(size-len(prefix))


def validation_partition(number,workers,selected,root):
    """Start at the first real tar header after a partition boundary.

    All boundaries are block aligned. Header names AND tar checksums must
    validate, and only headers starting in this partition are consumed.
    Streams may read one member past a byte boundary, so images are not split.
    """
    blocks = SIZES["val"]//512
    start,stop = (blocks*number//workers)*512,(blocks*(number+1)//workers)*512
    index_path = ROOT/f"artifacts/dataset_indexes/imagenet-val-part-{number}-of-{workers}.jsonl"
    index_path.parent.mkdir(parents=True,exist_ok=True)
    restored = 0
    with requests.Session() as session,get_range(session,"val",start,SIZES["val"]-1) as response:
        position = start
        while position < stop:
            header = response.raw.read(512)
            if len(header) != 512:
                raise ValueError("truncated validation archive header")
            name = header[:100].rstrip(b"\0")
            if re.fullmatch(rb"ILSVRC2012_val_[0-9]{8}\.JPEG",name):
                try:
                    tarfile.TarInfo.frombuf(header,"utf-8","surrogateescape")
                    break
                except tarfile.HeaderError:
                    pass
            position += 512
        else:
            return 0
        with index_path.open("w") as index_stream,tarfile.open(fileobj=PrefixReader(header,response.raw),mode="r|") as archive:
            for member in archive:
                header_position = position+member.offset
                if header_position >= stop:
                    break
                name = Path(member.name).name
                index_stream.write(json.dumps({"name":name,"start":position+member.offset_data,"size":member.size})+"\n")
                if name not in selected:
                    continue
                row = selected[name]
                data = archive.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError(f"official validation image differs from frozen bytes: {name}")
                destination = root/row["relative_path"]
                destination.parent.mkdir(parents=True,exist_ok=True)
                temporary = destination.with_suffix(".partial")
                temporary.write_bytes(data)
                temporary.replace(destination)
                restored += 1
                if restored%100 == 0:
                    print(f"validation stream {number+1}/{workers}: {restored} frozen images restored",flush=True)
    return restored


def validation_index():
    """Read durable tar offsets without trusting them as payload identity.

    Old partition indexes and newer small-region indexes can overlap after an
    interruption. A final torn JSON line is ignored; conflicting or invalid
    records fail closed. Every recovered image still requires its frozen hash.
    """
    by_start = {}
    for path in sorted(INDEX_DIR.glob("imagenet-val-*.jsonl")):
        lines = path.read_text().splitlines()
        for number,line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if number == len(lines)-1:
                    break
                raise ValueError(f"invalid validation index: {path}")
            if (set(record) != {"name","start","size"}
                    or not isinstance(record["name"],str)
                    or not re.fullmatch(r"ILSVRC2012_val_[0-9]{8}\.JPEG",record["name"])
                    or type(record["start"]) is not int
                    or type(record["size"]) is not int
                    or record["start"] < 512 or record["start"]%512
                    or record["size"] <= 0
                    or record["start"]+record["size"] > SIZES["val"]):
                raise ValueError(f"invalid validation index record: {path}")
            previous = by_start.setdefault(record["start"],record)
            if previous != record:
                raise ValueError(f"conflicting validation index record: {path}")
    records = sorted(by_start.values(),key=lambda record:record["start"])
    previous_end = 0
    for record in records:
        if record["start"]-512 < previous_end:
            raise ValueError("overlapping validation index records")
        previous_end = record["start"]+((record["size"]+511)//512)*512
    return records


def validation_regions(records,chunk_bytes=VALIDATION_CHUNK_BYTES):
    """Schedule only unindexed bytes in small, balanced, block-aligned jobs."""
    if chunk_bytes < 512 or chunk_bytes%512:
        raise ValueError("validation chunk size must be a positive multiple of 512")
    position = 0
    for record in records+[None]:
        stop = SIZES["val"] if record is None else record["start"]-512
        while position < stop:
            end = min(position+chunk_bytes,stop)
            yield position,end
            position = end
        if record is not None:
            position = record["start"]+((record["size"]+511)//512)*512


def write_validation_payload(data,row,root,name):
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError(f"official validation image differs from frozen bytes: {name}")
    destination = root/row["relative_path"]
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = destination.with_suffix(".partial")
    temporary.write_bytes(data)
    temporary.replace(destination)


def recover_indexed_validation(record,row,root):
    for attempt in range(4):
        try:
            with requests.Session() as session,get_range(session,"val",record["start"],record["start"]+record["size"]-1) as response:
                write_validation_payload(response.content,row,root,record["name"])
            return 1
        except (requests.RequestException,UrllibHTTPError,OSError) as error:
            if attempt == 3:
                raise
            print(f"retrying indexed validation image {record['name']}: {type(error).__name__}",flush=True)
            time.sleep(attempt+1)


def recover_validation_region(start,stop,selected,root):
    """Index an unvisited region and resume at a verified header on failure."""
    INDEX_DIR.mkdir(parents=True,exist_ok=True)
    index_path = INDEX_DIR/f"imagenet-val-region-{start}-{stop}.jsonl"
    resume = start
    restored = 0
    for attempt in range(4):
        if resume >= stop:
            return restored
        try:
            with requests.Session() as session,get_range(session,"val",resume,SIZES["val"]-1) as response:
                position = resume
                while position < stop:
                    header = response.raw.read(512)
                    if len(header) != 512:
                        raise tarfile.ReadError("truncated validation archive header")
                    name = header[:100].rstrip(b"\0")
                    if re.fullmatch(rb"ILSVRC2012_val_[0-9]{8}\.JPEG",name):
                        try:
                            tarfile.TarInfo.frombuf(header,"utf-8","surrogateescape")
                            break
                        except tarfile.HeaderError:
                            pass
                    position += 512
                else:
                    return restored
                with index_path.open("a",buffering=1) as index_stream,tarfile.open(fileobj=PrefixReader(header,response.raw),mode="r|") as archive:
                    for member in archive:
                        header_position = position+member.offset
                        if header_position >= stop:
                            return restored
                        name = Path(member.name).name
                        if not member.isfile() or not re.fullmatch(r"ILSVRC2012_val_[0-9]{8}\.JPEG",name):
                            raise ValueError("unexpected validation archive member")
                        record = {"name":name,"start":position+member.offset_data,"size":member.size}
                        if name in selected:
                            write_validation_payload(archive.extractfile(member).read(),selected[name],root,name)
                            restored += 1
                        index_stream.write(json.dumps(record)+"\n")
                        resume = record["start"]+((member.size+511)//512)*512
                return restored
        except (requests.RequestException,UrllibHTTPError,OSError,tarfile.ReadError) as error:
            if attempt == 3:
                raise
            print(f"retrying validation region at {resume}: {type(error).__name__}",flush=True)
            time.sleep(attempt+1)
    return restored


def recover_validation(selected,root,workers):
    records = validation_index()
    known = [record for record in records if record["name"] in selected]
    regions = list(validation_regions(records))
    print(f"resuming {len(known)} indexed images and {len(regions)} unindexed regions ({sum(stop-start for start,stop in regions)/1024**2:.1f} MiB)",flush=True)
    restored = 0
    with ThreadPoolExecutor(workers) as executor:
        futures = [executor.submit(recover_indexed_validation,record,selected[record["name"]],root) for record in known]
        futures.extend(executor.submit(recover_validation_region,start,stop,selected,root) for start,stop in regions)
        for number,future in enumerate(as_completed(futures),1):
            restored += future.result()
            print(f"validation jobs {number}/{len(futures)}: {restored}/{len(selected)} missing frozen images restored",flush=True)
            link_screening_subset()
    missing,_ = target_rows("val")
    if missing:
        raise ValueError(f"validation extraction missed {len(missing)} frozen images")
    print(f"linked {link_screening_subset()} frozen screening images",flush=True)


def link_screening_subset():
    records = json.loads((ROOT/"data/manifests/index.json").read_text())["records"]
    screen,evaluation = records["imagenet_screen_1k"],records["imagenet_evaluation_10k"]
    if sha256(ROOT/screen["path"]) != screen["sha256"]:
        raise ValueError("frozen ImageNet screening list mismatch")
    source_root = ROOT/"data/raw"/evaluation["logical_payload_root"]
    target_root = ROOT/"data/raw"/screen["logical_payload_root"]
    count = 0
    for row in load_tsv(ROOT/screen["path"]):
        source,target = source_root/row["relative_path"],target_root/row["relative_path"]
        if not source.exists():
            continue
        if sha256(source) != row["sha256"]:
            raise ValueError("screening payload hash mismatch")
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            os.link(source,target)
        elif sha256(target) != row["sha256"]:
            raise ValueError("existing screening payload hash mismatch")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",required=True,choices=("val","train"))
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--probe-first-class",action="store_true",help="Verify the first training class before a full restoration")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("use between one and eight archive streams")
    selected,root = target_rows(args.split)
    if not selected:
        print("all frozen payloads already verified")
        if args.split == "val":
            print(f"linked {link_screening_subset()} frozen screening images",flush=True)
        return
    if args.split == "val":
        recover_validation(selected,root,args.workers)
    else:
        grouped = {}
        for name,row in selected.items():
            grouped.setdefault(name.split("_")[0]+".tar",{})[name] = row
        total = 0
        with ThreadPoolExecutor(args.workers) as executor:
            futures = []
            for record in training_ranges():
                if record["name"] not in grouped:
                    continue
                futures.append(executor.submit(recover_stream,"train",record["start"],record["size"],grouped.pop(record["name"]),root))
                if args.probe_first_class:
                    break
            for future in as_completed(futures):
                total += future.result()
                print(f"restored {total}/{len(selected)} frozen training images",flush=True)
        if grouped and not args.probe_first_class:
            raise ValueError("frozen classes missing from official training archive")
    print("requested official-archive restoration finished",flush=True)


if __name__ == "__main__":
    main()
