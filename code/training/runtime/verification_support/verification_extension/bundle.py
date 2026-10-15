"""Export completed verification evidence without duplicating waveform caches."""
from pathlib import Path
import os
from zipfile import ZipFile, ZIP_DEFLATED

from .protocol import file_hash, read_json, write_json


def bundle_results(output):
    output = Path(output).resolve()
    statistics = read_json(output / 'corrected_summary/statistics.json')
    if not statistics.get('score_assets_ready'):
        raise RuntimeError('A verification return package needs the complete declared score suite')
    archive = output / 'verification_results_for_paper.zip'
    # Large replay arrays are transient inputs. Keep their manifests and hashes.
    files = [path for path in output.rglob('*') if path.is_file()
             and not any(part.startswith('.') for part in path.relative_to(output).parts)
             and path.suffix not in ('.zip', '.tmp')
             and not ('observation_cache' in path.relative_to(output).parts and path.suffix == '.npz')
             and path.name != 'return_package_manifest.json']
    write_json(output / 'return_package_manifest.json', {
        'schema_version': 1, 'suite': statistics['suite'],
        'paper_ready': statistics['paper_ready'], 'pending': statistics['pending'],
        'training_performed': False,
        'waveform_arrays_included': False,
        'files': {path.relative_to(output).as_posix(): file_hash(path) for path in sorted(files)}})
    files.append(output / 'return_package_manifest.json')
    temporary = archive.with_suffix('.zip.tmp')
    with ZipFile(temporary, 'w', compression=ZIP_DEFLATED, compresslevel=6) as package:
        for path in sorted(files):
            package.write(path, 'dopnet_verification_results/' + path.relative_to(output).as_posix())
    os.replace(temporary, archive)
    print('Return package: ' + str(archive), flush=True)
    return archive
