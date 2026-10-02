"""Mapping logic of tools/analysis/evaluation_history_audit.py on a tiny synthetic tree."""
import hashlib
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools", "analysis"))

import evaluation_history_audit as eha  # noqa: E402


def h(tag):
    return hashlib.sha256(tag.encode()).hexdigest()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def writebin(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def imagenet_rows(tags, start):
    rows = ["relative_path\tlabel\tsource_name\tstream_position\tsha256"]
    for i, t in enumerate(tags):
        s = h(t)
        rows.append("%04d/%s.JPEG\t%d\tILSVRC2012_val_%08d_n%08d.JPEG\t%d\t%s" % (i, s[:20], i, start + i, i, i, s))
    return "\n".join(rows) + "\n"


def coco_rows(ids, base):
    rows = ["file_name\timage_id\twidth\theight\tsha256"]
    for i in ids:
        rows.append("%012d.jpg\t%d\t640\t480\t%s" % (i, i, h("coco%d" % i)))
    return "\n".join(rows) + "\n"


def build(root):
    m = os.path.join(root, "data", "manifests")
    write(os.path.join(m, "calibration", "imagenet1k_train_2k.tsv"), imagenet_rows(["c0", "c1"], 100))
    write(os.path.join(m, "evaluation", "imagenet1k_val_1k.tsv"), imagenet_rows(["s0", "s1"], 200))
    write(os.path.join(m, "evaluation", "imagenet1k_val_10k.tsv"), imagenet_rows(["s0", "s1", "r0", "r1", "r2"], 200))
    write(os.path.join(m, "calibration", "coco2017_train_2k.tsv"), coco_rows([900, 901], 0))
    write(os.path.join(m, "evaluation", "coco2017_val_1k.tsv"), coco_rows([1, 2], 0))
    write(os.path.join(m, "evaluation", "coco2017_val_5k.tsv"), coco_rows([1, 2, 3, 4], 0))
    a = os.path.join(root, "artifacts")
    # stem evidence: screen image s0 and remainder image r0 in an evaluation tree
    write(os.path.join(a, "exp", "predictions", "cfgA", h("s0") + ".json"), "{}")
    write(os.path.join(a, "exp", "predictions", "cfgA", h("r0") + ".json"), "{}")
    # field evidence: relative_path of s1 (screen) and an unknown path
    rel_s1 = "%04d/%s.JPEG" % (1, h("s1")[:20])
    write(os.path.join(a, "other", "run.json"),
          json.dumps({"sample": {"relative_path": rel_s1}, "x": {"relative_path": "9999/" + "a" * 20 + ".JPEG"}}))
    # COCO predictions by image_id: 1 (screen), 3 (remainder), 77777 (outside)
    write(os.path.join(a, "per_image_predictions", "yolov8n_p.json"),
          json.dumps([{"image_id": 1}, {"image_id": 3}, {"image_id": 77777}]))
    # calibration tree touching a calibration image (ok) and an evaluation image (crossing)
    write(os.path.join(a, "calibration_obs", "m", h("c0") + ".json"), "{}")
    write(os.path.join(a, "calibration_obs", "m", h("s1") + ".json"), "{}")
    # evaluation tree touching a calibration image (crossing)
    write(os.path.join(a, "exp2", "predictions", "cfgB", h("c1") + ".json"), "{}")
    # mention-only evidence: r1 named in a log
    write(os.path.join(a, "logs_tree", "note.log"), "plan lists " + h("r1"))
    # formerly skipped tree: now read; coco id 4 is the last remainder image
    write(os.path.join(a, "agent_orchestration", "x.json"), json.dumps({"image_id": 4}))
    # binary payload named by an image hash, with a val name (no synset suffix) inside the bytes
    writebin(os.path.join(a, "binary", h("r2") + ".npz"), b"\x00\xff" * 10 + b"ILSVRC2012_val_%08d.JPEG" % 204 + b"\x00")
    # no size limit: identity in the tail of a file bigger than one chunk
    writebin(os.path.join(a, "big", "blob.bin"), b"\x00" * (eha.CHUNK - 10) + b'"image_id": 2')
    # full path with a leading slash, from another machine
    write(os.path.join(a, "conf", "t.xml"), "FileNotFoundError: /home/x/val_evaluation_10k/%s" % rel_s1)
    # source name without synset suffix (screen image s1 is val number 201), and one outside every list
    write(os.path.join(a, "names", "n.log"), "ILSVRC2012_val_00000201.JPEG ILSVRC2012_val_00099999.JPEG")
    # run record: per-image outcome keys next to the identity; non-run: manifest copy and archive index
    write(os.path.join(a, "runs", "r.json"), json.dumps({"sample": {"relative_path": "%04d/%s.JPEG" % (2, h("r0")[:20])}, "top1": 5}))
    write(os.path.join(a, "mcopy", "list.tsv"), "relative_path\tlabel\tsource_name\n%04d/%s.JPEG\t1\tx\n" % (3, h("r1")[:20]))
    write(os.path.join(a, "dataset_indexes", "idx.jsonl"), json.dumps({"name": "ILSVRC2012_val_00000202_n00000002.JPEG"}))
    # the audit's own output is not read
    os.makedirs(os.path.join(root, "results", "summaries"))
    write(os.path.join(root, "results", "summaries", "evaluation-history-audit-v1.json"), json.dumps({"image_id": 3}))
    write(os.path.join(root, "results", "top_level.txt"), '"image_id": 1')


def count(summary, tree, role, key, ds, cat):
    return summary["trees"][tree]["roles"][role][key][ds][cat]


def test_mapping_roles_and_crossings(tmp_path):
    root = str(tmp_path)
    build(root)
    s = eha.run(root, 1, False)
    ro = lambda t, r, c: s["trees"][t]["roles"][r].get("by_record_class", {}).get(c)

    assert s["calibration_vs_evaluation_list_overlap"] == {"imagenet": 0, "coco": 0}
    assert s["list_sizes"]["imagenet_evaluation_10k"] == 5

    # stems: one screen, one remainder image in an evaluation tree
    assert count(s, "exp", "evaluation", "strict_union", "imagenet", "screen_1k") == 1
    assert count(s, "exp", "evaluation", "strict_union", "imagenet", "remainder_9k") == 1
    # field evidence maps relative_path to the screen; unknown path is outside every list
    assert count(s, "other", "evaluation", "field", "imagenet", "screen_1k") == 1
    assert s["trees"]["other"]["outside_every_list"] == {"relative_path": 1}
    # COCO image_id mapping and an id outside every list
    assert count(s, "per_image_predictions", "evaluation", "field", "coco", "screen_1k") == 1
    assert count(s, "per_image_predictions", "evaluation", "field", "coco", "remainder_4k") == 1
    assert s["trees"]["per_image_predictions"]["outside_every_list"] == {"image_id": 1}
    # mention-only evidence is kept apart from strict evidence
    assert count(s, "logs_tree", "evaluation", "strict_union", "imagenet", "remainder_9k") == 0
    assert count(s, "logs_tree", "evaluation", "mention_only_extra", "imagenet", "remainder_9k") == 1
    # nothing skipped; binary payloads are read as bytes
    assert "skipped" not in s["trees"]["agent_orchestration"]
    assert count(s, "agent_orchestration", "evaluation", "field", "coco", "remainder_4k") == 1
    assert s["trees"]["binary"]["files"]["files"] == 1
    assert count(s, "binary", "evaluation", "field", "imagenet", "remainder_9k") == 1   # r2 = val 204 by name, no suffix
    # no size limit, identity in the tail of a file longer than one chunk
    assert s["trees"]["big"]["files"]["bytes_read"] > eha.CHUNK
    assert count(s, "big", "evaluation", "field", "coco", "screen_1k") == 1
    # files directly under a root are scanned; the audit's own output file is not
    assert "results/(root files)" in s["trees"]
    assert count(s, "results/(root files)", "evaluation", "field", "coco", "screen_1k") == 1
    assert "results/summaries" in s["trees"] and s["trees"]["results/summaries"]["files"].get("files", 0) == 0
    assert "results/summaries/evaluation-history-audit-v1.json" in s["scan"]["excluded_files"]
    # widened patterns: full path with a leading slash; val name without the synset suffix
    assert count(s, "conf", "evaluation", "field", "imagenet", "screen_1k") == 1
    assert count(s, "names", "evaluation", "field", "imagenet", "screen_1k") == 1
    assert s["trees"]["names"]["outside_every_list"] == {"source_name": 1}
    # run record versus non-run mention
    assert ro("runs", "evaluation", "run_record")["strict_union"]["imagenet"]["remainder_9k"] == 1
    assert ro("mcopy", "evaluation", "non_run")["strict_union"]["imagenet"]["remainder_9k"] == 1
    assert ro("mcopy", "evaluation", "run_record") is None
    assert ro("dataset_indexes", "index", "non_run")["strict_union"]["imagenet"]["remainder_9k"] == 1
    assert ro("names", "evaluation", "other_mention")["strict_union"]["imagenet"]["screen_1k"] == 1
    assert ro("exp", "evaluation", "run_record")["strict_union"]["imagenet"]["remainder_9k"] == 1
    rr = s["touched_remainder_sha256"]["evaluation_run_record_strict:imagenet/remainder_9k"]
    assert rr == sorted([h("r0"), h("r2")]) or rr == [h("r0")]
    assert h("r1") not in rr     # r1 only in a log mention and a manifest copy
    # crossings in both directions
    assert "calibration_obs (calibration role)" in s["crossing_trees"]
    assert s["crossing_trees"]["calibration_obs (calibration role)"]["imagenet"]["screen_1k"] == 1
    assert s["crossing_trees"]["exp2 (evaluation role)"]["imagenet"]["calibration_2k"] == 1
    # global touched remainder: r0 (stem, run record), r2 (stem + name in binary), r1 (manifest copy field)
    strict = s["touched_remainder_sha256"]["evaluation_strict:imagenet/remainder_9k"]
    both = s["touched_remainder_sha256"]["evaluation_with_mentions:imagenet/remainder_9k"]
    assert strict == sorted([h("r0"), h("r1"), h("r2")])
    assert both == strict
    # the sha mention of r1 in a log must not count as a run record
    assert h("r1") not in s["touched_remainder_sha256"]["evaluation_run_record_strict:imagenet/remainder_9k"]
    assert s["touched"]["evaluation_strict"]["coco/remainder_4k"]["count"] == 2   # ids 3 and 4


def test_digest_is_order_independent():
    assert eha.digest([h("a"), h("b")]) == eha.digest([h("b"), h("a")])
    assert eha.digest([]) == hashlib.sha256(b"").hexdigest()
