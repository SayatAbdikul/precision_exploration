"""Recompute the 21 locally complete classifier graph proofs before pilots.

This verifier is intentionally outside the running production acceptance path.
It never creates an acceptance artifact or authorizes screening by itself.
"""
from public.inference.conformance_job import source_identity
from tools.phase3.common import ROOT, campaign, checked, digest, file_hash, read
from development.acceptance_proofs import dyadic_fp_mac_store, posit_mac_store, q1_6_mac_store


SOURCES = {
    'q1_6': ('q1_6-mac-store-v1', q1_6_mac_store),
    **{name: ('dyadic-fp-mac-store-v1', dyadic_fp_mac_store) for name in dyadic_fp_mac_store.FORMATS},
    'posit4_es0': ('posit-mac-store-v1', posit_mac_store),
}


def verify_static(prepared, root=ROOT):
    config = read(checked(prepared['configuration'], root))
    graph = read(checked(prepared['graph'], root))
    key = config['model'] + '/' + config['formats']['activation']['name']
    name = config['formats']['activation']['name']
    if config['model'] == 'yolov8n' or name not in SOURCES:
        raise ValueError('configuration lacks a complete local classifier proof')
    source, campaign_id = source_identity(), digest(campaign(root))
    if (config['runtime']['source_sha256'] != source or config['campaign_sha256'] != campaign_id or
            digest(config) != prepared['configuration_sha256']):
        raise ValueError('prepared classifier identity mismatch')
    version, module = SOURCES[name]
    directory = root / 'artifacts/phase3/proof-development' / version
    summary = read(directory / 'summary.json')
    manifest = read(checked(summary['manifest'], directory))
    if (manifest['source_sha256'] != source or manifest['campaign_sha256'] != campaign_id or
            file_hash(module.__file__) != manifest['implementation_sha256']):
        raise ValueError('local proof source or frozen identity changed')
    matching = []
    for ref in summary['records']:
        record = read(checked(ref, directory))
        if record['configuration'] == key:
            matching.append(record)
    if len(matching) != 1:
        raise ValueError('configuration has no unique local proof')
    record = matching[0]
    if (record['graph'] != prepared['graph'] or record['configuration_artifact'] != prepared['configuration'] or
            record['source_sha256'] != source or record['campaign_sha256'] != campaign_id or
            record['proof_implementation_sha256'] != manifest['implementation_sha256'] or
            record['graph_acceptance'] is not False or record['pending_nonmac_nodes'] or
            record['proof']['status'] != 'conditional_output_codes_proven'):
        raise ValueError('local MAC/output or non-MAC graph proof remains pending')
    actual = module.prove_graph(graph) if name == 'q1_6' else module.prove_graph(graph, name)
    if actual != record['proof'] or actual['pending_channels']:
        raise ValueError('retained MAC/output proof does not reproduce')
    nonmac_name = 'phase3-fixed-nonmac' if name.startswith('posit') else 'phase3-finite-fp64-nonmac'
    nonmac = read(root / 'results/summaries' / (nonmac_name + '.json'))
    if nonmac['source_sha256'] != source or nonmac['campaign_sha256'] != campaign_id:
        raise ValueError('non-MAC report identity mismatch')
    rows = [row for row in nonmac['records'] if row['configuration'] == key]
    if len(rows) != 1 or rows[0]['graph'] != prepared['graph'] or rows[0]['pending_nodes']:
        raise ValueError('non-MAC report is missing or pending')
    shapes_summary = read(root / 'results/summaries/phase3-shapes.json')
    if shapes_summary['campaign_sha256'] != campaign_id:
        raise ValueError('retained shape inventory differs from campaign')
    shape_ref = shapes_summary['models'][config['model']]
    shapes = read(checked(shape_ref, root))['shapes']
    if name.startswith('posit'):
        from public.analysis.phase3.fixed_nonmac import prove_nonmac
    else:
        from public.analysis.phase3.finite_fp64_nonmac import prove_nonmac
    nonmac_actual = prove_nonmac(graph, shapes)
    if (nonmac_actual['records'] != rows[0]['records'] or nonmac_actual['pending_nodes'] or
            len(nonmac_actual['records']) != len(graph['nodes']) - actual['mac_nodes']):
        raise ValueError('retained non-MAC proof does not cover the graph')
    return {'configuration': key, 'status': 'static_graph_arithmetic_reproduced_native_pilot_required',
            'configuration_sha256': digest(config), 'graph': prepared['graph'],
            'mac_nodes': actual['mac_nodes'], 'nonmac_nodes': len(nonmac_actual['records']),
            'shape_evidence': shape_ref, 'source_sha256': source,
            'proof_implementation_sha256': manifest['implementation_sha256']}


def verify_with_native_pilot(prepared, pilot_ref, root=ROOT):
    static = verify_static(prepared, root)
    from tools.phase3.acceptance import verify_pilot
    graph, _, _, records = verify_pilot(prepared, pilot_ref, root)
    expected = read(checked(static['shape_evidence'], root))['shapes']
    if set(expected) != {node['name'] for node in graph['nodes']}:
        raise ValueError('shape evidence omits a classifier node')
    for record in records:
        for backend in ('cpp', 'cuda'):
            shapes = {name: row['shape'] for name, row in record['backends'][backend]['layers'].items()}
            if shapes != expected:
                raise ValueError('native pilot shape differs from retained FP32 graph')
    return {**static, 'status': 'family_graph_and_native_pilot_verified',
            'pilot': pilot_ref, 'images': len(records), 'backends': ['cpp', 'cuda']}
