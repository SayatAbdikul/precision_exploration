"""Study-scoped family admission from reproduced proofs and paired pilots."""
from tools.phase3.common import ROOT, checked, read, reference
from tools.phase3.acceptance import verify_pilot
from development.acceptance_proofs.family_static import verify_static


def verify_shapes(graph, observed, records):
    names = {node['name'] for node in graph['nodes']}
    if not names <= set(observed):
        raise ValueError('shape evidence omits a deployed graph node')
    # The FP32 trace additionally contains input, batch-normalization and
    # dropout nodes removed by folding. Check every deployed node explicitly;
    # unrelated FP32 trace entries cannot make complete coverage fail.
    expected = {name: observed[name] for name in names}
    for record in records:
        for backend in ('cpp', 'cuda'):
            actual = {name: row['shape'] for name, row in record['backends'][backend]['layers'].items()}
            if actual != expected:
                raise ValueError('native shapes do not match every deployed graph node')


def verify(prepared, pilot_ref):
    static = verify_static(prepared)
    graph, _, _, records = verify_pilot(prepared, pilot_ref)
    observed = read(checked(static['shape_evidence']))['shapes']
    verify_shapes(graph, observed, records)
    return {'status': 'study_family_graph_and_native_pilot_verified', 'static': static, 'pilot': pilot_ref,
            'images': 8, 'scope': 'prospective E1 only; historical A acceptance registry unchanged',
            'implementation': reference(__file__),
            'static_verifier': reference(ROOT / 'development/acceptance_proofs/family_static.py')}
