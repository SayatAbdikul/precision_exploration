"""Retain scaled binary/ternary MAC output-code evidence without acceptance."""
from pathlib import Path

from public.inference.conformance_job import source_identity
from tools.phase3.common import ROOT, campaign, checked, digest, file_hash, read, reference, write
from development.acceptance_proofs.mapped_mac_store import FORMATS, prove_graph


OUTPUT = ROOT / 'artifacts/phase3/proof-development/mapped-mac-store-v1'


def run(root=ROOT, output=OUTPUT):
    plan, source = campaign(root), source_identity()
    identity = digest(plan)
    inventory_path = root / 'results/summaries/phase3-gate-inventory.json'
    nonmac_path = root / 'results/summaries/phase3-finite-fp64-nonmac.json'
    inventory, nonmac = read(inventory_path), read(nonmac_path)
    if (inventory['campaign_sha256'] != identity or inventory['source_sha256'] != source or
            nonmac['campaign_sha256'] != identity or nonmac['source_sha256'] != source):
        raise ValueError('retained inputs do not match frozen campaign and numerical engine')
    implementation = Path(__file__).with_name('mapped_mac_store.py')
    sources = {}
    for path in (Path(__file__), implementation, Path(__file__).with_name('test_mapped_mac_store.py')):
        target = output / 'sources' / file_hash(path) / path.name
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError('proof source archive collision')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        sources[path.name] = reference(target, output)
    manifest = {'schema_version': 'phase3-mapped-mac-store-manifest-1.0.0',
                'campaign_sha256': identity, 'source_sha256': source,
                'sources': sources, 'implementation_sha256': file_hash(implementation),
                'gate_inventory': reference(inventory_path, root), 'nonmac_report': reference(nonmac_path, root)}
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError('proof inputs or implementation changed; use a new versioned output directory')
    write(manifest_path, manifest)
    nonmac_rows = {row['configuration']: row for row in nonmac['records']}
    rows = [row for row in inventory['records'] if row['configuration'].split('/')[1] in FORMATS]
    expected = {model + '/' + name for model in plan['models'] for name in FORMATS}
    if len(rows) != len(expected) or {row['configuration'] for row in rows} != expected:
        raise ValueError('expected exactly eight prepared mapped graphs')
    records = []
    for row in rows:
        key, name = row['configuration'], row['configuration'].split('/')[1]
        graph = read(checked(row['graph'], root))
        config = read(checked(row['configuration_artifact'], root))
        if config['runtime']['source_sha256'] != source or config['campaign_sha256'] != identity:
            raise ValueError('prepared mapped configuration identity mismatch')
        local = nonmac_rows[key]
        if local['graph'] != row['graph'] or local['configuration_artifact'] != row['configuration_artifact']:
            raise ValueError('non-MAC report belongs to another graph')
        proof = prove_graph(graph, name)
        if proof['mac_nodes'] != row['mac_nodes']:
            raise ValueError('mapped MAC proof omitted a node')
        record = {'schema_version': 'phase3-mapped-mac-store-record-1.0.0',
                  'configuration': key, 'campaign_sha256': identity, 'source_sha256': source,
                  'configuration_artifact': row['configuration_artifact'], 'graph': row['graph'],
                  'proof_implementation_sha256': file_hash(implementation), 'proof': proof,
                  'pending_nonmac_nodes': local['pending_nodes'], 'graph_acceptance': False,
                  'native_pilot_required': True}
        path = output / 'configurations' / (digest(record) + '.json')
        if path.exists() and read(path) != record:
            raise ValueError('proof record content collision')
        write(path, record)
        records.append(reference(path, output))
        print(key, proof['status'], proof['mac_nodes'], 'MACs;', proof['pending_channels'],
              'sensitive channels;', len(local['pending_nodes']), 'pending non-MAC', flush=True)
    if source_identity() != source:
        raise ValueError('numerical source changed while proving mapped graphs')
    documents = [read(checked(ref, output)) for ref in records]
    summary = {'schema_version': 'phase3-mapped-mac-store-summary-1.0.0',
               'campaign_sha256': identity, 'source_sha256': source,
               'configuration_count': len(records),
               'mac_output_code_proven_configurations': sum(r['proof']['status'] == 'conditional_output_codes_proven' for r in documents),
               'mac_nodes': sum(r['proof']['mac_nodes'] for r in documents),
               'channels': sum(r['proof']['channels'] for r in documents),
               'pending_sensitive_channels': sum(r['proof']['pending_channels'] for r in documents),
               'pending_nonmac_nodes': sum(len(r['pending_nonmac_nodes']) for r in documents),
               'graph_acceptance': False, 'manifest': reference(manifest_path, output),
               'records': records, 'status': 'conditional_local_output_code_evidence_not_graph_acceptance'}
    write(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    result = run()
    print({key: value for key, value in result.items() if key != 'records'}, flush=True)
