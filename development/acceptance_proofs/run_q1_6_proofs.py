"""Retain Q1.6 MAC output-code evidence without issuing graph acceptance."""
from pathlib import Path

from public.inference.conformance_job import source_identity
from tools.phase3.common import ROOT, campaign, checked, digest, file_hash, read, reference, write
from development.acceptance_proofs.q1_6_mac_store import prove_graph


OUTPUT = ROOT / 'artifacts/phase3/proof-development/q1_6-mac-store-v1'


def run(root=ROOT, output=OUTPUT):
    plan, source = campaign(root), source_identity()
    identity = digest(plan)
    inventory_path = root / 'results/summaries/phase3-gate-inventory.json'
    nonmac_path = root / 'results/summaries/phase3-finite-fp64-nonmac.json'
    inventory, nonmac = read(inventory_path), read(nonmac_path)
    if (inventory['campaign_sha256'] != identity or inventory['source_sha256'] != source or
            nonmac['campaign_sha256'] != identity or nonmac['source_sha256'] != source):
        raise ValueError('retained inputs do not match frozen campaign and numerical engine')
    implementation = Path(__file__).with_name('q1_6_mac_store.py')
    sources = {}
    for path in (Path(__file__), implementation, Path(__file__).with_name('test_q1_6_mac_store.py')):
        target = output / 'sources' / file_hash(path) / path.name
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError('proof source archive collision')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        sources[path.name] = reference(target, output)
    manifest = {'schema_version': 'phase3-q1_6-mac-store-manifest-1.0.0',
                'campaign_sha256': identity, 'source_sha256': source,
                'sources': sources, 'implementation_sha256': file_hash(implementation),
                'gate_inventory': reference(inventory_path, root), 'nonmac_report': reference(nonmac_path, root)}
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError('proof inputs or implementation changed; use a new versioned output directory')
    write(manifest_path, manifest)
    nonmac_rows = {row['configuration']: row for row in nonmac['records']}
    rows = [row for row in inventory['records'] if row['configuration'].endswith('/q1_6')]
    if len(rows) != 4 or {row['configuration'] for row in rows} != {model + '/q1_6' for model in plan['models']}:
        raise ValueError('expected exactly four prepared Q1.6 graphs')
    records = []
    for row in rows:
        key = row['configuration']
        graph = read(checked(row['graph'], root))
        config = read(checked(row['configuration_artifact'], root))
        if config['runtime']['source_sha256'] != source or config['campaign_sha256'] != identity:
            raise ValueError('prepared Q1.6 configuration identity mismatch')
        local = nonmac_rows[key]
        if local['graph'] != row['graph'] or local['configuration_artifact'] != row['configuration_artifact']:
            raise ValueError('non-MAC report belongs to another graph')
        proof = prove_graph(graph)
        if proof['mac_nodes'] != row['mac_nodes'] or proof['pending_channels'] or proof['status'] != 'conditional_output_codes_proven':
            raise ValueError('Q1.6 MAC/output proof is incomplete')
        record = {'schema_version': 'phase3-q1_6-mac-store-record-1.0.0',
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
        print(key, proof['mac_nodes'], 'MACs', proof['channels'], 'channels; pending non-MAC:', len(local['pending_nodes']), flush=True)
    if source_identity() != source:
        raise ValueError('numerical source changed while proving Q1.6 graphs')
    summary = {'schema_version': 'phase3-q1_6-mac-store-summary-1.0.0',
               'campaign_sha256': identity, 'source_sha256': source,
               'configuration_count': len(records), 'mac_nodes': sum(read(checked(ref, output))['proof']['mac_nodes'] for ref in records),
               'channels': sum(read(checked(ref, output))['proof']['channels'] for ref in records),
               'pending_nonmac_nodes': sum(len(read(checked(ref, output))['pending_nonmac_nodes']) for ref in records),
               'graph_acceptance': False, 'manifest': reference(manifest_path, output), 'records': records,
               'status': 'conditional_q1_6_mac_output_codes_proven_not_graph_acceptance'}
    write(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    result = run()
    print({key: value for key, value in result.items() if key != 'records'}, flush=True)
