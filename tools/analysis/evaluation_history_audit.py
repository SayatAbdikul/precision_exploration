#!/usr/bin/env python3
"""Read-only audit: which frozen-list images has this project ever run a model on?

Reads EVERY file under artifacts/ and results/ (all trees, files directly under
the two roots, binary files, sqlite databases, no size limit, no skipped tree) as
bytes, in chunks.  The only file left out is the script's own output file.
Image identities found in file names and file bytes are mapped to the frozen lists
in data/manifests (ImageNet 2k calibration / 1k screen / 10k evaluation; COCO 2k
calibration / 1k screen / 5k evaluation).

Identity evidence kinds (kept separate in the output):
  stem     a manifest sha256 appears in the file NAME (e.g. predictions/<cfg>/<image sha>.json)
  field    a record field names the image: ImageNet relative_path / source_name,
           COCO file_name, COCO image_id (only these key forms, not bare numbers)
  mention  a manifest sha256 appears anywhere in the file text (weakest: a plan
           or log may name an image it never ran)
  sqlite   a sample_id / image_id value in a sqlite table (per_image_results etc.)

Record class of a file (kept separate in the output; decided per file):
  non_run        archive offset index (tree dataset_indexes) or a manifest copy (path
                 part contains "manifest", or the file starts with a manifest header)
  run_record     the identity is the file name (stem), a sqlite per-image row, or the
                 same file carries per-image outcome keys (top1, logits, prediction,
                 detections, bbox, score, category_id, output, correct, loss ...)
  other_mention  an identity occurs in a file that is neither of the above (plan,
                 log, summary, binary file ...); weaker than run_record, stronger than
                 nothing, never discounted in the verdict.

Nothing is written except the JSON summary (and optionally stdout).  Git state is
not touched.  Typical use:

    .venv/bin/python tools/analysis/evaluation_history_audit.py \
        --out results/summaries/evaluation-history-audit-v1.json

Rerun it just before any confirmation run (about 10 minutes for 21 GB, parallel)
and compare `touched_digest` per list with the previous summary.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import multiprocessing as mp
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MANIFESTS = {
    # name: (path, dataset, role)
    "imagenet_calibration_2k": ("data/manifests/calibration/imagenet1k_train_2k.tsv", "imagenet", "calibration"),
    "imagenet_screen_1k": ("data/manifests/evaluation/imagenet1k_val_1k.tsv", "imagenet", "screen"),
    "imagenet_evaluation_10k": ("data/manifests/evaluation/imagenet1k_val_10k.tsv", "imagenet", "evaluation"),
    "coco_calibration_2k": ("data/manifests/calibration/coco2017_train_2k.tsv", "coco", "calibration"),
    "coco_screen_1k": ("data/manifests/evaluation/coco2017_val_1k.tsv", "coco", "screen"),
    "coco_evaluation_5k": ("data/manifests/evaluation/coco2017_val_5k.tsv", "coco", "evaluation"),
}

# Categories reported per dataset.
CATEGORIES = {
    "imagenet": ["screen_1k", "remainder_9k", "calibration_2k"],
    "coco": ["screen_1k", "remainder_4k", "calibration_2k"],
}

DEFAULT_OUT = os.path.join("results", "summaries", "evaluation-history-audit-v1.json")
CHUNK = 32 * 1024 * 1024
OVERLAP = 1024          # longer than any identity pattern, so no match is split by a chunk edge
SQLITE_SUFFIXES = (".sqlite", ".sqlite3", ".db")

HEX64 = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
STEM64 = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
# full paths ("/x/val_evaluation_10k/0072/<sha20>.JPEG") are matched too: only a letter or digit may not precede
RELPATH = re.compile(rb"(?<![0-9A-Za-z])(\d{4}/[0-9a-f]{20}\.JPEG)")
# source names with or without the synset suffix
SRCNAME = re.compile(rb"(ILSVRC2012_val_\d{8})(?:_n\d{8})?\.JPEG|(n\d{8}_\d+_n\d{8}\.JPEG)")
COCONAME = re.compile(rb"(?<![0-9])(\d{12}\.jpg)")
COCOID = re.compile(rb'"image_id"\s*:\s*(\d+)')
RUN_MARKER = re.compile(
    rb'"(?:top1|top5|top_k|logits?|predictions?|predicted|pred|detections?|boxes|bbox|scores?|'
    rb'category_id|outputs?|output_sha256|correct|is_correct|loss|margin|metrics)"\s*:')
MANIFEST_HEADS = (b"relative_path\tlabel\tsource_name", b"file_name\timage_id", b"sha256\t")
MODEL_RE = re.compile(r"(resnet18|mobilenet_v2|mobilenet_v3_large|yolov8n)")


# ----------------------------------------------------------------------------
# manifests

class Frozen:
    """Frozen lists and identity maps (module-global so forked workers share it)."""

    def __init__(self, root):
        self.root = root
        self.sets = {}          # list name -> set(sha)
        self.rel_to_sha = {}
        self.src_to_sha = {}
        self.val_to_sha = {}    # "ILSVRC2012_val_00023515" -> sha (the number is unique over the 50k set)
        self.file_to_sha = {}
        self.id_to_sha = {}
        self.category = {}      # sha -> (dataset, category)
        self.dataset_of = {}    # sha -> dataset
        self.manifest_sha = {}
        for name, (path, dataset, role) in MANIFESTS.items():
            full = os.path.join(root, path)
            with open(full, "rb") as fh:
                self.manifest_sha[name] = hashlib.sha256(fh.read()).hexdigest()
            rows = list(csv.DictReader(open(full, newline=""), delimiter="\t"))
            self.sets[name] = {r["sha256"] for r in rows}
            for r in rows:
                s = r["sha256"]
                self.dataset_of[s] = dataset
                if dataset == "imagenet":
                    self.rel_to_sha[r["relative_path"]] = s
                    self.src_to_sha[r["source_name"]] = s
                    if r["source_name"].startswith("ILSVRC2012_val_"):
                        self.val_to_sha[r["source_name"][:23]] = s
                else:
                    self.file_to_sha[r["file_name"]] = s
                    self.id_to_sha[r["image_id"]] = s
        for ds, cal, scr, ev, rem in [
            ("imagenet", "imagenet_calibration_2k", "imagenet_screen_1k", "imagenet_evaluation_10k", "remainder_9k"),
            ("coco", "coco_calibration_2k", "coco_screen_1k", "coco_evaluation_5k", "remainder_4k"),
        ]:
            for s in self.sets[ev]:
                self.category[s] = (ds, "screen_1k" if s in self.sets[scr] else rem)
            for s in self.sets[cal]:
                # calibration and evaluation lists are disjoint (checked in main)
                self.category.setdefault(s, (ds, "calibration_2k"))

    def sha_from_relpath(self, rel):
        return self.rel_to_sha.get(rel)


FROZEN: Frozen | None = None
CFG_MODEL: dict = {}


# Trees whose records carry no model name; the hint says where the model comes from.
TREE_MODEL_HINT = {
    "scaled_bridge_v1": "resnet18(inferred)",
    "scaled_bridge_gap_v1": "resnet18(inferred)",
}


def role_of(parts):
    """Role of a file from its path parts: calibration | index | evaluation."""
    for p in parts:
        if "calibration" in p:
            return "calibration"
    if parts and parts[0] == "dataset_indexes":
        return "index"
    return "evaluation"


def model_of(rel_parts, tree):
    if tree == "per_image_predictions":
        return "file:" + "/".join(rel_parts[1:])
    s = "/".join(rel_parts)
    m = MODEL_RE.search(s)
    if m:
        return m.group(1)
    for p in rel_parts:
        if p in CFG_MODEL:
            return CFG_MODEL[p]
    if tree == "per_image_predictions":
        return "file:" + "/".join(rel_parts[1:])
    if tree in ("workload_runs",) or tree.startswith("results"):
        return "file:" + (rel_parts[-1] if tree != "workload_runs" else rel_parts[1][:12])
    return TREE_MODEL_HINT.get(tree, "unattributed")


# ----------------------------------------------------------------------------
# per-file extraction (runs in worker processes)

def classify_head(rel_parts, tree, head):
    """Definite non-run files: archive offset indexes and manifest copies."""
    if tree == "dataset_indexes":
        return "non_run"
    if any("manifest" in p.lower() for p in rel_parts):
        return "non_run"
    if head.startswith(MANIFEST_HEADS):
        return "non_run"
    return None


def extract(data, F, ev, outside):
    for m in RELPATH.findall(data):
        rel = m.decode()
        s = F.rel_to_sha.get(rel)
        (ev["field"].add(s) if s else outside["relative_path"].add(rel))
    for m_val, m_train in SRCNAME.findall(data):
        if m_val:
            nm = m_val.decode()
            s = F.val_to_sha.get(nm)
            if s:
                ev["field"].add(s)
            else:
                outside["source_name"].add(nm)
        else:
            nm = m_train.decode()
            s = F.src_to_sha.get(nm)
            if s:
                ev["field"].add(s)
    for m in COCONAME.findall(data):
        nm = m.decode()
        s = F.file_to_sha.get(nm)
        if s:
            ev["field"].add(s)
        else:
            outside["file_name"].add(nm)
    for m in COCOID.findall(data):
        s = F.id_to_sha.get(m.decode())
        if s:
            ev["field"].add(s)
        else:
            outside["image_id"].add(m.decode())
    for m in set(HEX64.findall(data)):
        h = m.decode()
        if h in F.category:
            ev["mention"].add(h)


def scan_file(task):
    """Read one file completely as bytes.  Returns (tree, role, model, status, bytes,
    record_class, evidence, outside) with evidence = {kind: set(sha)} and
    outside = {kind: set(identifiers not in any frozen list)}."""
    path, tree, rel_parts = task
    F = FROZEN
    ev = defaultdict(set)
    outside = defaultdict(set)
    status = "read"
    nbytes = 0
    has_marker = False
    cls = None
    for part in rel_parts:
        for h in STEM64.findall(part):
            if h in F.category:
                ev["stem"].add(h)
    try:
        with open(path, "rb") as fh:
            tail = b""
            first = True
            while True:
                block = fh.read(CHUNK)
                if not block:
                    break
                nbytes += len(block)
                if first:
                    cls = classify_head(rel_parts, tree, block[:256])
                    first = False
                data = tail + block
                extract(data, F, ev, outside)
                if not has_marker and RUN_MARKER.search(data):
                    has_marker = True
                tail = data[-OVERLAP:]
        if first:   # empty file
            cls = classify_head(rel_parts, tree, b"")
    except OSError as exc:
        status = "unreadable:" + type(exc).__name__
    if cls is None:
        cls = "run_record" if (ev.get("stem") or has_marker) else "other_mention"
    role = role_of(rel_parts)
    model = model_of(rel_parts, tree)
    return tree, role, model, status, nbytes, cls, {k: v for k, v in ev.items()}, {k: v for k, v in outside.items()}


# ----------------------------------------------------------------------------

def tree_name(top, name):
    return name if top == "artifacts" else "results/" + name


def discover(root):
    """Evidence trees: every top-level directory of artifacts/ and results/, plus the
    files lying directly in each of the two roots (as the pseudo-tree '(root files)')."""
    trees = []
    for top in ("artifacts", "results"):
        base = os.path.join(root, top)
        if not os.path.isdir(base):
            continue
        names = sorted(os.listdir(base))
        for name in names:
            full = os.path.join(base, name)
            if os.path.isdir(full) and not os.path.islink(full):
                trees.append((top, name, full))
        if any(not (os.path.isdir(os.path.join(base, n)) and not os.path.islink(os.path.join(base, n))) for n in names):
            trees.append((top, "(root files)", base))
    return trees


def load_cfg_models(root):
    """config sha -> model, from configuration records of the b-trees and phase3."""
    out = {}
    for tree in ("experiment_b", "experiment_b_ext", "experiment_b2"):
        d = os.path.join(root, "artifacts", tree, "configurations")
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.endswith(".json"):
                continue
            try:
                j = json.load(open(os.path.join(d, f)))
                p = j.get("payload", j)
                m = p.get("model_context", {}).get("model")
                if m:
                    out[f[:-5]] = m
            except Exception:
                pass
    d = os.path.join(root, "artifacts", "phase3", "configurations")
    if os.path.isdir(d):
        for h in os.listdir(d):
            f = os.path.join(d, h, "configuration.json")
            if os.path.isfile(f):
                try:
                    m = json.load(open(f)).get("model")
                    if m:
                        out[h] = m
                except Exception:
                    pass
    return out


def iter_tasks(root, trees, excluded):
    """Every file under the trees; `excluded` = set of real paths that are not read."""
    for top, name, full in trees:
        tree = tree_name(top, name)
        base = os.path.join(root, top)
        if name == "(root files)":
            walker = [(full, [], [f for f in sorted(os.listdir(full)) if not (os.path.isdir(os.path.join(full, f)) and not os.path.islink(os.path.join(full, f)))])]
        else:
            walker = os.walk(full)
        for dp, dn, fn in walker:
            for f in fn:
                p = os.path.join(dp, f)
                if os.path.realpath(p) in excluded:
                    continue
                parts = tuple(os.path.relpath(p, base).split(os.sep))
                yield (p, tree, parts)


def sqlite_evidence(root, trees, excluded):
    """Per-image rows of every sqlite database under the trees (opened read-only,
    immutable): every table that has a sample_id or image_id column."""
    out = []
    for top, name, full in trees:
        base = os.path.join(root, top)
        for dp, dn, fn in os.walk(full):
            for f in sorted(fn):
                if not f.lower().endswith(SQLITE_SUFFIXES):
                    continue
                p = os.path.join(dp, f)
                if os.path.realpath(p) in excluded or (name == "(root files)" and dp != full):
                    continue
                rel = os.path.relpath(p, base)
                entry = {"tree": tree_name(top, name), "db": rel, "ids": [], "cfg": {}, "tables": []}
                try:
                    c = sqlite3.connect("file:%s?mode=ro&immutable=1" % p, uri=True)
                    tables = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
                    if "configurations" in tables:
                        try:
                            for eid, canon in c.execute("select experiment_id, canonical_json from configurations"):
                                try:
                                    entry["cfg"][eid] = json.loads(canon)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    for t in tables:
                        cols = [r[1] for r in c.execute('pragma table_info("%s")' % t.replace('"', '""'))]
                        for col in ("sample_id", "image_id"):
                            if col in cols:
                                ecol = "experiment_id" if "experiment_id" in cols else "NULL"
                                rows = c.execute('select %s, "%s" from "%s"' % (ecol, col, t.replace('"', '""'))).fetchall()
                                entry["tables"].append("%s.%s (%d rows)" % (t, col, len(rows)))
                                entry["ids"].extend(rows)
                    c.close()
                except Exception as exc:
                    entry["error"] = "%s: %s" % (type(exc).__name__, exc)
                out.append(entry)
    return out


def digest(shas):
    return hashlib.sha256("\n".join(sorted(shas)).encode()).hexdigest()


def cat_counts(F, shas):
    c = {"imagenet": defaultdict(int), "coco": defaultdict(int)}
    for s in shas:
        ds, cat = F.category[s]
        c[ds][cat] += 1
    return {ds: {k: c[ds].get(k, 0) for k in CATEGORIES[ds]} for ds in c}


CLASSES = ("run_record", "non_run", "other_mention")


def run(root, workers, include_mentions_in_union=False, out_path=None):
    global FROZEN, CFG_MODEL
    t0 = time.time()
    started = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    FROZEN = F = Frozen(root)
    CFG_MODEL = load_cfg_models(root)
    cal_eval_overlap = {
        "imagenet": len(F.sets["imagenet_calibration_2k"] & F.sets["imagenet_evaluation_10k"]),
        "coco": len(F.sets["coco_calibration_2k"] & F.sets["coco_evaluation_5k"]),
    }
    excluded = {os.path.realpath(os.path.join(root, DEFAULT_OUT))}
    if out_path:
        excluded.add(os.path.realpath(out_path))
    trees = discover(root)
    tasks = list(iter_tasks(root, trees, excluded))

    # tree -> role -> class -> kind -> set ; plus (tree, model, role) -> kind -> set
    by_tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
    by_model = defaultdict(lambda: defaultdict(set))
    outside = defaultdict(lambda: defaultdict(set))
    files = defaultdict(lambda: defaultdict(int))
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for tree, role, model, status, nbytes, cls, ev, out in pool.imap_unordered(scan_file, tasks, chunksize=16):
            files[tree]["files"] += 1
            files[tree]["bytes_read"] += nbytes
            if status != "read":
                files[tree][status] += 1
            files[tree]["class_" + cls] += 1
            for k, v in ev.items():
                by_tree[tree][role][cls][k] |= v
                by_model[(tree, model, role)][k] |= v
            for k, v in out.items():
                outside[tree][k] |= v
    sqlite_dbs = []
    for entry in sqlite_evidence(root, trees, excluded):
        tree = entry["tree"]
        files[tree]["sqlite_files_queried"] += 1
        sqlite_dbs.append({k: entry[k] for k in ("tree", "db", "tables")} | ({"error": entry["error"]} if entry.get("error") else {}))
        if entry.get("error"):
            files[tree]["sqlite_query_errors"] += 1
            continue
        for eid, sid in entry["ids"]:
            s = F.rel_to_sha.get(sid) or F.id_to_sha.get(str(sid))
            model = (entry["cfg"].get(eid) or {}).get("model") or "unattributed"
            role = role_of(tuple(entry["db"].split(os.sep)))
            if s:
                by_tree[tree][role]["run_record"]["sqlite"].add(s)
                by_model[(tree, entry["db"] + ":" + model, role)]["sqlite"].add(s)
            else:
                outside[tree]["sqlite_sample_id"].add(str(sid))

    strict_kinds = ("stem", "field", "sqlite")

    def pick(kinds, with_mention):
        u = set()
        for k, v in kinds.items():
            if k in strict_kinds or (with_mention and k == "mention"):
                u |= v
        return u

    def merged(role_classes, classes=CLASSES):
        m = defaultdict(set)
        for c in classes:
            for k, v in role_classes.get(c, {}).items():
                m[k] |= v
        return m

    tree_rows = {}
    for top, name, full in trees:
        tree = tree_name(top, name)
        row = {"path": ("artifacts/" if top == "artifacts" else "results/") + name}
        row["files"] = dict(files.get(tree, {}))
        row["roles"] = {}
        for role, classes in by_tree.get(tree, {}).items():
            kinds = merged(classes)
            r = {}
            for k, v in kinds.items():
                r[k] = cat_counts(F, v)
            strict = pick(kinds, False)
            r["strict_union"] = cat_counts(F, strict)
            r["mention_only_extra"] = cat_counts(F, pick(kinds, True) - strict)
            r["by_record_class"] = {}
            for c in CLASSES:
                ck = merged(classes, (c,))
                cs = pick(ck, False)
                if ck:
                    r["by_record_class"][c] = {"strict_union": cat_counts(F, cs),
                                               "mention_only_extra": cat_counts(F, pick(ck, True) - cs)}
            row["roles"][role] = r
        if outside.get(tree):
            row["outside_every_list"] = {k: len(v) for k, v in outside[tree].items()}
            row["outside_examples"] = {k: sorted(v)[:3] for k, v in outside[tree].items()}
        tree_rows[tree] = row

    models = {}
    for (tree, model, role), kinds in sorted(by_model.items()):
        strict = pick(kinds, False)
        if not strict and not kinds.get("mention"):
            continue
        models["%s | %s | %s" % (tree, model, role)] = {
            "strict_union": cat_counts(F, strict),
            "mention_only_extra": cat_counts(F, pick(kinds, True) - strict),
        }

    # global unions
    union = defaultdict(set)
    for tree, roles in by_tree.items():
        for role, classes in roles.items():
            kinds = merged(classes)
            if role == "evaluation":
                union["evaluation_strict"] |= pick(kinds, False)
                union["evaluation_with_mentions"] |= pick(kinds, True)
                for c in CLASSES:
                    union["evaluation_%s_with_mentions" % c] |= pick(merged(classes, (c,)), True)
                union["evaluation_run_record_strict"] |= pick(merged(classes, ("run_record",)), False)
            elif role == "calibration":
                union["calibration_strict"] |= pick(kinds, False)
                union["calibration_with_mentions"] |= pick(kinds, True)
            else:
                union["index_mentions"] |= pick(kinds, True)
            union["non_run_mentions_any_role"] |= pick(merged(classes, ("non_run",)), True)
    for k in ("evaluation_strict", "evaluation_with_mentions", "evaluation_run_record_strict",
              "evaluation_run_record_with_mentions", "evaluation_non_run_with_mentions",
              "evaluation_other_mention_with_mentions", "calibration_strict",
              "calibration_with_mentions", "index_mentions", "non_run_mentions_any_role"):
        union.setdefault(k, set())

    def split_by_category(shas):
        out = {}
        for ds in ("imagenet", "coco"):
            for cat in CATEGORIES[ds]:
                sub = sorted(x for x in shas if F.category[x] == (ds, cat))
                out["%s/%s" % (ds, cat)] = {"count": len(sub), "touched_digest": digest(sub)}
        return out

    touched = {k: split_by_category(v) for k, v in sorted(union.items())}
    remainder_lists = {}
    for k in ("evaluation_strict", "evaluation_with_mentions", "evaluation_run_record_strict",
              "evaluation_run_record_with_mentions"):
        for ds, cat in (("imagenet", "remainder_9k"), ("coco", "remainder_4k")):
            remainder_lists["%s:%s/%s" % (k, ds, cat)] = sorted(x for x in union[k] if F.category[x] == (ds, cat))

    cross = {
        "evaluation_role_trees_touching_calibration_list_images": cat_counts(F, {x for x in union["evaluation_strict"] if F.category[x][1] == "calibration_2k"}),
        "calibration_role_trees_touching_evaluation_list_images": cat_counts(F, {x for x in union["calibration_strict"] if F.category[x][1] != "calibration_2k"}),
    }
    cross_trees = {}
    for tree, roles in by_tree.items():
        for role, classes in roles.items():
            s = pick(merged(classes), True)
            bad = {x for x in s if (role == "evaluation" and F.category[x][1] == "calibration_2k")
                   or (role == "calibration" and F.category[x][1] != "calibration_2k")}
            if bad:
                cross_trees["%s (%s role)" % (tree, role)] = cat_counts(F, bad)

    list_sizes = {n: len(s) for n, s in F.sets.items()}
    total_files = sum(v.get("files", 0) for v in files.values())
    summary = {
        "schema": "evaluation-history-audit-2",
        "scan_started_utc": started,
        "scan_seconds": round(time.time() - t0, 1),
        "repo": root,
        "git_head_note": "git state not read by this script; record `git rev-parse HEAD` alongside the summary if needed",
        "manifest_sha256": F.manifest_sha,
        "list_sizes": list_sizes,
        "calibration_vs_evaluation_list_overlap": cal_eval_overlap,
        "evidence_kinds": {
            "stem": "manifest sha256 in a path part (file or directory name)",
            "field": "relative_path (also inside a full path) / ILSVRC2012_val_NNNNNNNN with or without synset suffix / COCO file_name / \"image_id\" in file bytes",
            "sqlite": "sample_id or image_id column of any sqlite table",
            "mention": "manifest sha256 anywhere in file bytes (weak)",
        },
        "record_classes": {
            "non_run": "archive offset index (tree dataset_indexes) or manifest copy (path part 'manifest', or file starts with a manifest header)",
            "run_record": "identity is the file name, a sqlite per-image row, or the same file has per-image outcome keys (top1, logits, prediction(s), detections, bbox, score, category_id, output, correct, loss, margin, metrics)",
            "other_mention": "identity in any other file (plan, log, summary, test report, binary); counted with the evaluation evidence, not discounted",
        },
        "scan": {
            "reads": "every file under artifacts/ and results/ as bytes, no size limit, no skipped tree, binary and sqlite files included, files directly under the two roots included",
            "chunk_bytes": CHUNK,
            "chunk_overlap_bytes": OVERLAP,
            "excluded_files": sorted(os.path.relpath(x, root) for x in excluded),
            "workers": workers,
            "total_bytes_read": sum(v.get("bytes_read", 0) for v in files.values()),
            "unreadable_files": sum(n for v in files.values() for k, n in v.items() if k.startswith("unreadable")),
            "sqlite_databases": sqlite_dbs,
        },
        "trees": tree_rows,
        "model_attribution": models,
        "touched": touched,
        "touched_remainder_sha256": remainder_lists,
        "crossings": cross,
        "crossing_trees": cross_trees,
        "total_files_seen": total_files,
        "not_covered": [
            "a model run that left no record under artifacts/ or results/ is invisible",
            "records of other machines are not here",
            "COCO images named only by a bare number outside an image_id key or a sqlite sample_id/image_id column",
            "backups/, cache/, build/, data/, git history are outside this scan",
        ],
    }
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=REPO)
    ap.add_argument("--out", help="write the JSON summary here (default: stdout)")
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    a = ap.parse_args(argv)
    summary = run(os.path.abspath(a.root), a.workers, False, os.path.abspath(a.out) if a.out else None)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
        print("wrote %s (%.1f s, %d files, %.1f GB)" % (a.out, summary["scan_seconds"], summary["total_files_seen"],
                                                         summary["scan"]["total_bytes_read"] / 1e9))
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
