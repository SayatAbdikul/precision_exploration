# Useful-quality E1 prospective protocol v1

Frozen before new native outcomes. Protocol file SHA256: `0315ab8af441761d7ad2e58c8bbdecc47dc40819f9766a52173c0b8300cb91de`. The protocol includes the actual ordered 128-image manifest; 32 is the initial quality panel, and the first eight images are the paired native admission panel.

The selected B maxabs configurations motivate this candidate set. They do not translate to canonical A: A uses necessary MSE integer scales or unscaled floating formats, explicit residual alignment and accumulator-domain bias. B uses optional external maxabs scales and folded framework QDQ. Their scores are kept under separate identities.

| Case | Exact graph file SHA256 | A scale policy | Source archive |
| --- | --- | --- | --- |
| resnet18/int8 | `891369e4ad8f3777642d931ff0494add4dd67057e806afa3800c42a778dba0c1` | necessary INT MSE scales | `2e5a519effcd24697846b03803d44e94597121693813e92d7802352a7859cb13` |
| mobilenet_v2/int8 | `175f3d8f41dc592560b231f2b598fbc199f44d2c2a2f0768ae73739a9304b567` | necessary INT MSE scales | `2e5a519effcd24697846b03803d44e94597121693813e92d7802352a7859cb13` |
| resnet18/fp6_e2m3 | `c6b632bb01f2f3d90455523265e09dd46d2e7897d70e1fde44f821b2b2d20d52` | direct unscaled FP format | `2e5a519effcd24697846b03803d44e94597121693813e92d7802352a7859cb13` |
| resnet18/fp6_e3m2 | `fbaeb10ee7599c69537fec7ececcbb0c20ab25c12604030cb25bfd11b1f81c04` | direct unscaled FP format | `2e5a519effcd24697846b03803d44e94597121693813e92d7802352a7859cb13` |
| resnet18/fp7_e3m3 | `765adddcbb54031e6cc988f48a331347876b80058eb47c0329f5a4365bea645d` | direct unscaled FP format | `2e5a519effcd24697846b03803d44e94597121693813e92d7802352a7859cb13` |

Each sealed mapping contains checkpoint/preprocessing/deployment hashes, calibration manifest/context, every deployed node and attribute, weight-code hashes, all scales/biases, format manifest hashes and the distinct B configuration reference.

Strict and control arms share graph, codes, scales, bias/store policy, image order and sequential reduction order. The control changes MAC reduction to correctly rounded sequential FP32 FMA, residual addition to a single FP32 rounding, and average-pool summation to sequential FP32. Integer operations use their original unscaled code domain. The final MAC state is cast into the original accumulator before the original bias, activation and output store. ReLU/ReLU6, maxpool and flatten stay identical.

The two INT8 strict arms reuse verified accepted image records with original execution provenance. The three floating graphs must reproduce their local dyadic MAC/store and complete non-MAC proofs, then pass eight fresh CPP/CUDA images. Every control requires rational-oracle native conformance and eight paired CPP/CUDA images. The first newly executed floating exact image also compares the prepared engine with the uncompiled exact executor, including diagnostic-event signatures. No historical acceptance is modified.

Promotion is fixed before viewing new outcomes: extend to 128 when either arithmetic arm reaches at least 50% top-1 with at most 20 percentage points loss to paired FP32, or reaches at least 40% with observed layer/output divergence. Both arms at or below 10% never promote. Useful integer anchors use the same 128-image panel as promoted floating cases. All five remain in the ledger. This is exploratory development, not independent confirmation.

The resource plan starts with two processes and four threads per process. It permits three only after pilot RSS is below 2 GiB and available memory is at least 12 GiB; it falls to one below 6 GiB available. Native/controller locks exclude competing native jobs. The planned new-inference budget is 24 aggregate worker-hours, with at most 12 allocated to extensions. Extension estimates use 1.5 times the measured median CUDA per-image cost, including input and diagnostic preparation, and allocate in frozen candidate order. Retained exact records consume no new inference time.

Resume: `.venv/bin/python -m tools.run.useful_quality run`

Status: `.venv/bin/python -m tools.run.useful_quality status`

Audit and figures after completion: `.venv/bin/python -m tools.analysis.useful_quality_e1`
