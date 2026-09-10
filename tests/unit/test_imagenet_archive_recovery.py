import hashlib
import io
import json
import tarfile

import pytest

from tools.setup import restore_phase2_imagenet_archives as recovery


@pytest.fixture
def archive_source(tmp_path,monkeypatch):
    stream = io.BytesIO()
    selected = {}
    with tarfile.open(fileobj=stream,mode="w") as archive:
        for number,size in enumerate((2400,6500,3700),1):
            name = f"ILSVRC2012_val_{number:08}.JPEG"
            data = bytes([number])*size
            member = tarfile.TarInfo(name)
            member.size = size
            archive.addfile(member,io.BytesIO(data))
            selected[name] = {"relative_path":f"0000/{number}.JPEG","sha256":hashlib.sha256(data).hexdigest()}
    payload = stream.getvalue()
    records = []
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive:
            records.append({"name":member.name,"start":member.offset_data,"size":member.size})
    calls = []

    class Response:
        def __init__(self,data):
            self.raw = io.BytesIO(data)
            self.content = data

        def __enter__(self):
            return self

        def __exit__(self,*args):
            self.raw.close()

    def get_range(session,split,start,end):
        assert split == "val"
        calls.append((start,end))
        return Response(payload[start:end+1])

    monkeypatch.setattr(recovery,"INDEX_DIR",tmp_path/"index")
    monkeypatch.setattr(recovery,"SIZES",{"val":len(payload)})
    monkeypatch.setattr(recovery,"get_range",get_range)
    monkeypatch.setattr(recovery.time,"sleep",lambda _:None)
    return selected,records,calls,tmp_path/"images"


def test_interrupted_region_restarts_at_next_unwritten_member(archive_source,monkeypatch):
    selected,records,calls,root = archive_source
    original = recovery.write_validation_payload
    failed = False

    def write(data,row,destination,name):
        nonlocal failed
        if name == records[1]["name"] and not failed:
            failed = True
            raise OSError("injected connection interruption")
        original(data,row,destination,name)

    monkeypatch.setattr(recovery,"write_validation_payload",write)
    assert recovery.recover_validation_region(0,recovery.SIZES["val"],selected,root) == 3
    assert calls[1][0] == records[1]["start"]-512
    assert recovery.validation_index() == records
    for row in selected.values():
        assert recovery.sha256(root/row["relative_path"]) == row["sha256"]


def test_recovery_skips_indexed_bytes_and_handles_boundaries_inside_members(archive_source):
    selected,records,calls,root = archive_source
    recovery.INDEX_DIR.mkdir()
    (recovery.INDEX_DIR/"imagenet-val-old.jsonl").write_text(json.dumps(records[0])+"\n")
    indexed = recovery.validation_index()
    regions = list(recovery.validation_regions(indexed,chunk_bytes=2048))
    assert min(start for start,_ in regions) == records[1]["start"]-512
    count = recovery.recover_indexed_validation(records[0],selected[records[0]["name"]],root)
    for start,stop in regions:
        count += recovery.recover_validation_region(start,stop,selected,root)
    assert count == 3
    assert recovery.validation_index() == records
    assert calls[0] == (records[0]["start"],records[0]["start"]+records[0]["size"]-1)


def test_hash_mismatch_never_publishes_payload(archive_source):
    selected,records,calls,root = archive_source
    row = dict(selected[records[0]["name"]],sha256="0"*64)
    with pytest.raises(ValueError,match="differs from frozen bytes"):
        recovery.recover_indexed_validation(records[0],row,root)
    assert not root.exists()


def test_index_tolerates_torn_tail_but_rejects_conflicting_offsets(archive_source):
    _,records,_,_ = archive_source
    recovery.INDEX_DIR.mkdir()
    first = recovery.INDEX_DIR/"imagenet-val-old.jsonl"
    first.write_text(json.dumps(records[0])+"\n{\"name\":")
    assert recovery.validation_index() == records[:1]
    second = recovery.INDEX_DIR/"imagenet-val-new.jsonl"
    second.write_text(json.dumps(dict(records[0],size=records[0]["size"]+1))+"\n")
    with pytest.raises(ValueError,match="conflicting"):
        recovery.validation_index()
