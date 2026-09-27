"""Retain the 21 reproducible static classifier candidates; no acceptance."""
from pathlib import Path

from public.inference.conformance_job import source_identity
from tools.phase3.common import ROOT, campaign, checked, digest, file_hash, read, reference, write
from tools.phase3.execution_guard import verify_guard, controller_identity
from tools.phase3.preparation_inventory import preparation_records
from development.acceptance_proofs.family_static import verify_static
from development.acceptance_proofs.pilot_batch import pilot_inventory


OUTPUT = ROOT / 'artifacts/phase3/proof-development/family-static-v2'


def run(root=ROOT, output=OUTPUT):
    source, campaign_id = source_identity(), digest(campaign(root))
    execution = (verify_guard(root), controller_identity(root))
    _, candidates = pilot_inventory(root)
    implementation = Path(__file__).with_name('family_static.py')
    sources = {}
    for path in (Path(__file__), implementation, Path(__file__).with_name('test_family_static.py')):
        target = output / 'sources' / file_hash(path) / path.name
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError('static candidate source archive collision')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        sources[path.name] = reference(target, output)
    manifest = {'schema_version': 'phase3-family-static-manifest-1.0.0',
                'campaign_sha256': campaign_id, 'source_sha256': source,
                'guard_sha256': execution[0], 'controller_sha256': execution[1],
                'implementation_sha256': file_hash(implementation), 'sources': sources}
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError('static candidate implementation changed; choose a new output version')
    write(manifest_path, manifest)
    prepared = preparation_records(root)
    refs = []
    for key in sorted(candidates):
        row = prepared[key]
        item = read((root / row['configuration']['path']).parent / 'prepared.json')
        verified = verify_static(item, root)
        record = {'schema_version': 'phase3-family-static-record-1.0.0',
                  'configuration': key, 'campaign_sha256': campaign_id, 'source_sha256': source,
                  'implementation_sha256': file_hash(implementation),
                  'verification': verified, 'graph_acceptance': False,
                  'remaining_gate': 'eight matching native C++/CUDA pilot images and family-aware production acceptance dispatch'}
        path = output / 'configurations' / (digest(record) + '.json')
        if path.exists() and read(path) != record:
            raise ValueError('static candidate record collision')
        write(path, record)
        refs.append(reference(path, output))
        print(key, verified['mac_nodes'], 'MACs;', verified['nonmac_nodes'], 'non-MAC nodes', flush=True)
    if source_identity() != source or (verify_guard(root), controller_identity(root)) != execution:
        raise ValueError('live execution identity changed during static checks')
    documents = [read(checked(ref, output)) for ref in refs]
    summary = {'schema_version': 'phase3-family-static-summary-1.0.0',
               'campaign_sha256': campaign_id, 'source_sha256': source,
               'configuration_count': len(documents), 'mac_nodes': sum(r['verification']['mac_nodes'] for r in documents),
               'nonmac_nodes': sum(r['verification']['nonmac_nodes'] for r in documents),
               'graph_acceptance': False, 'manifest': reference(manifest_path, output),
               'records': refs, 'status': 'static_classifier_arithmetic_reproduced_native_pilots_required'}
    write(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    result = run()
    print({key: value for key, value in result.items() if key != 'records'}, flush=True)
