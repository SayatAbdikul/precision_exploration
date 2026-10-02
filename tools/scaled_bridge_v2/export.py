"""Recipe exports: the quantization recipe as data.

Schema `scaled-bridge-export-2` (one JSON document plus one array archive):

  codebooks   id -> {id, bits, shift, units[], codes[], choose_upper_tie[]}
              ascending integer units on a 2^-shift grid; signedness is the
              range of `units` (an unsigned activation range is a codebook
              whose units are all >= 0).
  nodes[]     execution order. Each node: name, op, inputs, attrs and
              store: null | {codebook, scale}
                  null on identity/flatten: the code state passes through.
                  null on an arithmetic node: the boundary does not quantize
                  and the binary64 value is passed on (fused boundary).
              mac (conv/linear only): {weight_codebook, weight_units,
                  weight_scales[], bias} where weight_units and bias name
                  arrays in constants.npz (int64 grid integers; binary64 bias).
  provenance  where the recipe came from; free-form for the engine.

The engine reads only `schema`, `case`, `model`, `codebooks`, `nodes` and the
arrays. This module holds the adapter for the original B recipes (family b1):
B's own code produces the weights and scales, and the adapter verifies that
its integer codes reproduce B's FP32 reconstruction bit for bit. Other recipe
families (B2) write the same schema from their own adapters.
"""
from __future__ import annotations
import operator
import numpy as np
from .common import (ROOT, BASE, EXPORT_SCHEMA, digest, file_hash, unseal, reference, checked,
                     immutable, array_hash, save_npz, exporter_sources, case_id)
from .codebooks import Codebook, scalar_codebook, quantize

LEDGER = ROOT / 'results/summaries/b-stage-paired-1k-v2/analysis.json'


def b_anchor(model, fmt, recipe):
    """Verify and load the retained B evidence for one configuration (read-only)."""
    from tools.experiment_b import common as b
    ledger = unseal(LEDGER)
    entry = next(x for x in ledger['configurations'] if (x['model'], x['format'], x['recipe']) == (model, fmt, recipe))
    if entry['component'] != 'experiment_b':
        raise ValueError('not a scalar B configuration; shared-exponent formats use the block adapter')
    config = unseal(checked(entry['configuration']))
    summary = unseal(checked(entry['summary']))
    ident = b.digest(config)
    if ident != entry['configuration_sha256'] or config['source_sha256'] != b.source_identity():
        raise ValueError('original B configuration/source drift')
    if config['protocol'] != b.PROTOCOL or config['model_context'] != b.frozen_inputs(model):
        raise ValueError('B frozen context/protocol drift')
    images = int(summary['panel_images'])
    rows = b.dataset('imagenet_screen_1k')[1]
    predictions = {}
    for key, identity, expected in (('B', ident, summary['prediction_digest']),
                                    ('FP32', config['baseline_sha256'], summary['baseline_prediction_digest'])):
        records = [unseal(b.BASE / 'predictions' / identity / (r['sha256'] + '.json')) for r in rows[:images]]
        if any(r['sample'] != s or r['configuration_sha256'] != identity or r['batch_start'] != 8 * (i // 8)
               or r['batch_images'] != 8 for i, (r, s) in enumerate(zip(records, rows))):
            raise ValueError('retained B batch membership/sample identity mismatch')
        if b.digest(records) != expected:
            raise ValueError('B prediction digest drift')
        predictions[key] = records
    def percent(records, top):
        return 100 * sum((int(r['sample']['label']) in r['top5'][:top]) for r in records) / images
    historical = {'images': images, 'B_top1': percent(predictions['B'], 1), 'B_top5': percent(predictions['B'], 5),
                  'FP32_top1': percent(predictions['FP32'], 1)}
    if abs(historical['B_top1'] - summary['metrics']['top1_percent']) > 1e-9:
        raise ValueError('historical B metrics mismatch')
    return {'rows': rows, 'entry': entry, 'config': config, 'retained': predictions,
            'historical': historical, 'ledger': reference(LEDGER)}


def b_calibration(model, config, need_samples):
    """Reload B's retained calibration evidence exactly as B combined it."""
    from tools.experiment_b import common as b
    folder = b.BASE / 'calibration' / model / config['calibration']['identity']
    summary = unseal(folder / 'summary.json')
    provenance = unseal(folder / 'provenance.json')
    if b.digest(summary) != config['calibration']['summary_sha256'] or b.digest(provenance) != config['calibration']['identity']:
        raise ValueError('calibration summary/provenance drift')
    rows = b.dataset('imagenet_calibration_2k')[1]
    maxima, chunks = {}, {}
    for batch in summary['batch_checkpoints']:
        start = batch['start']
        path = folder / f'{start:05d}.json'
        meta = unseal(path)
        data = path.with_suffix('.npz')
        if file_hash(path) != batch['sha256'] or file_hash(data) != meta['arrays_sha256'] or meta['samples'] != rows[start:start + 8]:
            raise ValueError('calibration checkpoint drift')
        if meta['calibration_sha256'] != config['calibration']['identity']:
            raise ValueError('calibration identity changed')
        for k, v in meta['maxima'].items():
            maxima[k] = max(maxima.get(k, 0), v)
        if need_samples:
            with np.load(data, allow_pickle=False) as saved:
                for k in saved.files:
                    chunks.setdefault(k, []).append(saved[k].ravel())
    arrays = ({k: np.concatenate(v) for k, v in chunks.items()} if need_samples
              else {k: np.empty(0) for k in maxima})
    return arrays, maxima, {'summary': reference(folder / 'summary.json'),
                            'provenance': reference(folder / 'provenance.json'),
                            'batches': len(summary['batch_checkpoints'])}


def topology(graph):
    """Folded FX graph as operator rows. Unknown operators fail closed."""
    import torch
    from torch.nn import functional as F
    nn = torch.nn
    result = []
    for node in graph.graph.nodes:
        row = {'name': node.name, 'fx_op': node.op, 'target': str(node.target),
               'inputs': [x.name for x in node.all_input_nodes], 'attrs': {}}
        if node.op == 'placeholder':
            row['op'] = 'input'
        elif node.op == 'output':
            row['op'] = 'output'
        elif node.op == 'call_function' and node.target == operator.add:
            row['op'] = 'add'
        elif node.op == 'call_function' and node.target == operator.mul:
            row['op'] = 'mul'
        elif node.op == 'call_function' and node.target == torch.flatten:
            row['op'] = 'flatten'
        elif node.op == 'call_function' and node.target == F.adaptive_avg_pool2d:
            if node.args[1] not in (1, (1, 1)):
                raise ValueError('only global average pooling is exported')
            row['op'] = 'avgpool'
        elif node.op == 'call_module':
            mod = graph.get_submodule(node.target)
            def pair(value):
                return [value, value] if isinstance(value, int) else list(value)
            if isinstance(mod, nn.Conv2d):
                if mod.padding_mode != 'zeros' or isinstance(mod.padding, str):
                    raise ValueError('unsupported convolution padding')
                row['op'] = 'conv'
                row['attrs'] = {k: list(getattr(mod, k)) for k in ('stride', 'padding', 'dilation')}
                row['attrs']['groups'] = mod.groups
            elif isinstance(mod, nn.Linear):
                row['op'] = 'linear'
            elif isinstance(mod, nn.ReLU):
                row['op'] = 'relu'
            elif isinstance(mod, nn.ReLU6):
                row['op'] = 'relu6'
            elif isinstance(mod, nn.Hardswish):
                row['op'] = 'hardswish'
            elif isinstance(mod, nn.Hardsigmoid):
                row['op'] = 'hardsigmoid'
            elif isinstance(mod, nn.MaxPool2d):
                row['op'] = 'maxpool'
                row['attrs'] = {k: pair(getattr(mod, k)) for k in ('kernel_size', 'stride', 'padding', 'dilation')}
                if mod.ceil_mode:
                    raise ValueError('ceil mode unsupported')
            elif isinstance(mod, nn.AdaptiveAvgPool2d) and mod.output_size in (1, (1, 1)):
                row['op'] = 'avgpool'
            elif isinstance(mod, nn.Flatten):
                row['op'] = 'flatten'
            elif isinstance(mod, (nn.Identity, nn.Dropout)):
                row['op'] = 'identity'
            else:
                raise ValueError(f'unsupported exported module {type(mod)}')
        else:
            raise ValueError(f'unsupported FX node {node}')
        result.append(row)
    return result


def export_path(case):
    return BASE / 'exports' / digest(exporter_sources()) / case / 'export.json'


def export_b1(model, fmt, recipe, device='cuda'):
    """Export one original-B configuration. Needs B's retained CUDA runtime."""
    import torch
    from tools.experiment_b.classifier import configure, load_model, prepare_qdq, quantized_node
    from tools.experiment_b.runner import runtime
    case = case_id(model, fmt, recipe)
    path = export_path(case)
    if path.exists():
        return reference(path)
    book_data = scalar_codebook(fmt)
    book = Codebook(book_data)
    configure(device)
    anchor = b_anchor(model, fmt, recipe)
    config = anchor['config']
    if runtime(device) != config['runtime']:
        raise ValueError('B runtime differs from the retained batch-8 anchor')
    graph, transform, original = load_model(model, device)
    del original
    arrays_cal, maxima, cal = b_calibration(model, config, need_samples=recipe != 'maxabs')
    with torch.inference_mode():
        engine, scales = prepare_qdq(graph, fmt, recipe, arrays_cal, maxima, device)
    if scales != config['scales']:
        raise ValueError('B weight/activation scales are not reproduced')
    nodes, arrays, reconstruction = [], {}, {}
    fx = {n.name: n for n in graph.graph.nodes}
    for row in topology(graph):
        key = row['name']
        stored = quantized_node(graph, fx[key])
        if stored != (row['op'] not in ('identity', 'flatten', 'output')):
            raise ValueError('B boundary rule and exported operator class disagree')
        row['store'] = {'codebook': book.id, 'scale': scales['activation_scales'][key]} if stored else None
        if row['op'] in ('conv', 'linear'):
            mod = graph.get_submodule(row['target'])
            w = mod.weight.detach().cpu().numpy()
            ws = np.array(scales['weight_scales'][key], dtype=np.float32).reshape((-1,) + (1,) * (w.ndim - 1))
            codes, units, _ = quantize(w, ws, book, b_weight=True)
            rebuilt = np.ldexp(units.astype(np.float32), -book.shift) * ws
            rebuilt = np.where((units == 0) & np.signbit(w), np.float32(-0.), rebuilt).astype(np.float32)
            b_weight = engine.module.get_submodule(row['target']).weight.detach().cpu().numpy()
            if not np.array_equal(rebuilt.view('u4'), b_weight.view('u4')):
                raise ValueError('exported weight codes do not reproduce B FP32 reconstruction')
            bias = (mod.bias.detach().cpu().numpy() if mod.bias is not None
                    else np.zeros(w.shape[0], dtype=np.float32))
            arrays[key + '__weight_units'] = units.astype(np.int64)
            arrays[key + '__weight_codes'] = codes
            arrays[key + '__bias'] = bias.astype(np.float64)
            reconstruction[key] = array_hash(b_weight)
            row['mac'] = {'weight_codebook': book.id, 'weight_units': key + '__weight_units',
                          'weight_scales': [float(s) for s in scales['weight_scales'][key]],
                          'bias': key + '__bias', 'bias_origin': 'folded FP32 bias promoted exactly to binary64'}
        nodes.append(row)
    keep = min(anchor['historical']['images'], 128)
    logical = {
        'schema': EXPORT_SCHEMA, 'case': case, 'model': model, 'format': fmt, 'recipe': recipe,
        'recipe_family': 'b1 (original Experiment B scalar QDQ)',
        'codebooks': {book.id: book_data}, 'nodes': nodes,
        'array_identities': {k: array_hash(v) for k, v in arrays.items()},
        'provenance': {
            'B_configuration': anchor['entry']['configuration'], 'B_summary': anchor['entry']['summary'],
            'B_source_sha256': config['source_sha256'], 'B_runtime': config['runtime'],
            'B_analysis_ledger': anchor['ledger'], 'calibration': cal,
            'model_context': config['model_context'], 'input_transform': repr(transform),
            'folded_FX_topology': str(graph.graph), 'B_reconstructed_weight_sha256': reconstruction,
            'weight_check': 'integer codes reproduce B FP32 reconstructed weights bit for bit, including signed zero',
            'exporter_sources': exporter_sources()},
        'retained': {'images': anchor['historical']['images'], 'historical': anchor['historical'],
                     'B_prefix': anchor['retained']['B'][:keep], 'FP32_prefix': anchor['retained']['FP32'][:keep],
                     'ordered_samples_sha256': digest(anchor['rows'])},
    }
    constants = save_npz(path.with_name('constants.npz'), arrays)
    immutable(path, {**logical, 'constants': constants})
    return reference(path)


def load_export(case, path=None):
    path = path or export_path(case)
    data = unseal(path)
    if data['schema'] != EXPORT_SCHEMA or data['case'] != case:
        raise ValueError('unexpected bridge export')
    with np.load(checked(data['constants']), allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    if {k: array_hash(v) for k, v in arrays.items()} != data['array_identities']:
        raise ValueError('export array drift')
    return data, arrays, reference(path)
