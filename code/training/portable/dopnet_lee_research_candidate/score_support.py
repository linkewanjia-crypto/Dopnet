"""Accepted scientific score inputs; no administrative parent authorization."""
import hashlib
import copy
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from .common import require, digest, file_hash, sign, read_signed
WIMAE_ARCHIVE_SHA = 'b4220c699f92d82b147365444e59684fc2a61139d5056295a496a6e04fddea13'
PORTABLE_PARENT_ARCHIVE_SHA = '3ec0b2adff81c7ce89581cf68bc135736f0e02e79a7bec5a91763cc85d090f0c'
SCIENTIFIC_PROJECTION_SHA = 'e95a7a20a2ac245f3d1ec4a96537b716a9e0936861bc53fd09b96ae67394f935'
ORIGINAL_DELIVERY_SIGNATURE = '567889565e14fdd98b4e07e3414d5fcfb5d3d8d588e4cfc7b80b09d9fc8388ea'
ORIGINAL_EVALUATION_SIGNATURE = 'daf7b2c4c2ca6215b797d8df33fc04bc67285b734081bf8afa4ab35542918196'
SEEDS = (42,43,44)
NEW_METHOD = 'lee_alternative_meta_task_adapted'
CLAIM = 'fulltext_guided_original_algorithm_equal_frame_budget_task_adaptation'
LABELS = {'dopnet':'DOP-Net','dopnet_lite':'DOP-Net-Lite','lee3ddual':'Dual3D (supervised adapter)',
          'wimae_ce':'WiMAE (supervised adapter)','fortitran':'FortiTran (common recipe)',
          'fortitran_budget_matched':'FortiTran (equal budget)','wimae_author':'WiMAE (20 SSL +20 CE)',
          NEW_METHOD:'Lee A-ML (1 meta +39 joint)'}
CORE=('errors_diagonal','errors_native','bits','native_residual_max','native_direct_fallback')
NOISE=('errors_noise_minus3db','errors_noise_plus3db')
signed = read_signed

def safe_name(name):
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name and p.as_posix() == name

def require_routes(routes, inherited):
    require(len(inherited) == 21 and len(routes) == 24 and routes[:21] == inherited, 'All21 inherited routes must be reused exactly')
    require(routes[21:] == [dict(method=NEW_METHOD, training_seed=s, trained=True) for s in SEEDS], 'Lee routes/seeds differ')
    require(list(dict.fromkeys(r['method'] for r in routes)) == list(LABELS), 'Expected eight method order differs')
    for method in LABELS:
        rr = [r for r in routes if r['method'] == method]
        require([r['training_seed'] for r in rr] == list(SEEDS) and all(r['trained'] is True for r in rr), 'Incomplete three-seed route')

def read_parent_archive(archive):
    """Verify this namespace's scientific input view, never a runtime permission.

    Original scalar score signatures identify the unchanged NPZ payloads only.
    The view has its own integrity signature and is not the whole historical
    WiMAE delivery. No source, training, nested archive or admin gate is run.
    """
    archive = Path(archive)
    require(file_hash(archive) == PORTABLE_PARENT_ARCHIVE_SHA, 'Wrong portable scientific parent view bytes')
    with ZipFile(archive) as z:
        names = [x.filename for x in z.infolist() if not x.is_dir()]
        require(len(names) == len(set(names)) and all(safe_name(n) for n in names), 'Unsafe/duplicate parent members')
        require(z.testzip() is None, 'Parent ZIP CRC failed')
        m = json.loads(z.read('DELIVERY_MANIFEST.json'))
        require(m.get('signature') == digest({k:v for k,v in m.items() if k != 'signature'})
                and m.get('schema') == 'dopnet_lee_portable_parent_view_v1'
                and m.get('scientific_projection_complete') is True
                and m.get('original_whole_archive_checker_claimed_for_new_view') is False
                and m.get('source_archive_sha256') == WIMAE_ARCHIVE_SHA
                and m.get('science_projection_sha256') == SCIENTIFIC_PROJECTION_SHA,
                'Incomplete or foreign scientific view identity')
        require(set(names) == set(m['files'])|{'DELIVERY_MANIFEST.json'}, 'Parent membership differs')
        for name, record in m['files'].items():
            raw = z.read(name)
            require(len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'],
                    'Parent view member bytes differ: '+name)
        projection = json.loads(z.read('SCIENTIFIC_PROJECTION.json'))
        require(projection.get('signature') == digest({k:v for k,v in projection.items() if k != 'signature'})
                and projection.get('schema') == 'dopnet_lee_portable_parent_scientific_projection_v1'
                and projection['source_archive']['sha256'] == WIMAE_ARCHIVE_SHA
                and projection.get('arrays_recomputed') is False
                and projection.get('administrative_authorization_included_or_reused') is False,
                'Wrong scientific origin projection')
        origin = json.loads(z.read('origin/DELIVERY_IDENTITY.json'))
        require(origin.get('schema') == 'dopnet_historical_parent_delivery_identity_v1'
                and origin['original_archive_sha256'] == WIMAE_ARCHIVE_SHA
                and origin['historical_original_delivery_signature'] == ORIGINAL_DELIVERY_SIGNATURE
                and origin.get('original_metadata_payload_included') is False,
                'Wrong historical origin identity')
        ev = json.loads(z.read('evaluation/evaluation.json'))
        require(ev.get('signature') == digest({k:v for k,v in ev.items() if k != 'signature'})
                and ev.get('schema') == 'dopnet_lee_portable_parent_evaluation_v1'
                and len(ev['cells']) == 269 and len(ev['routes']) == 21
                and ev['samples_per_cell'] == 512 and ev['batch_size'] == 8,
                'Parent frozen scientific suite differs')
        require(ev['origin']['historical_original_evaluation_signature'] == ORIGINAL_EVALUATION_SIGNATURE
                and projection['original_evaluation']['historical_signature'] == ORIGINAL_EVALUATION_SIGNATURE,
                'Wrong historical score signature')
        science = copy.deepcopy(ev)
        for key in ('signature','schema','origin'):
            science.pop(key,None)
        science['forti_delivery']['archive'] = 'origin/fortitran_parent_archive_not_in_scientific_view'
        require(digest(science) == SCIENTIFIC_PROJECTION_SHA == projection['science_projection_sha256'],
                'Scientific metadata projection changed')
        cell_names = {f'evaluation/cells/{cell["cell_id"]}.npz' for cell in ev['cells']}
        expected = cell_names|{'DELIVERY_MANIFEST.json','SCIENTIFIC_PROJECTION.json',
                               'origin/DELIVERY_IDENTITY.json','evaluation/evaluation.json','README.txt'}
        require(len(cell_names) == 269 and set(names) == expected, 'Unexpected scientific view payload')
        retained = projection['retained_scientific_members']
        require(len(retained) == 269 and {r['member'] for r in retained} == cell_names,
                'Incomplete original-to-view score closure')
        for record in retained:
            require(record['original_bytes_equal_copy'] is True
                    and record['original_sha256'] == record['copy_sha256'] == m['files'][record['member']]['sha256']
                    and record['bytes'] == m['files'][record['member']]['bytes'],
                    'Retained score is not byte-identical to its scientific origin')
    # The unchanged scoring loop compares NPZ scalar signatures against their
    # historical score origin. This temporary dictionary is never signed/written
    # as a new receipt; current input integrity remains m['signature'] above.
    score_metadata = dict(ev, view_integrity_signature=ev['signature'])
    score_metadata['signature'] = ORIGINAL_EVALUATION_SIGNATURE
    manifest = dict(m, files={name:row['sha256'] for name,row in m['files'].items()})
    return score_metadata, manifest

def receiver_keys(cell,available):
    """Exact original receiver scope, including all33 BER denominators."""
    if not cell['ber']:
        require(not set(CORE+NOISE).intersection(available),'NMSE-only cell has receiver data')
        return ()
    require(set(CORE)<=set(available),'Mandatory BER/native fields absent')
    sweep=cell['modulation']=='16qam' and cell['snr_db']==20 and cell['doppler_hz']==1200
    require(set(NOISE).intersection(available)==(set(NOISE) if sweep else set()),'Parent receiver noise fields differ from frozen original scope')
    return CORE+(NOISE if sweep else ())

