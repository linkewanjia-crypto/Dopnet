"""Fresh manifests, source identities and original scientific configuration checks."""
import ast
import copy
import json
from pathlib import Path
from .common import digest, file_hash, read_signed, sign, write_json, require
HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
VALIDATION_SHA256 = '83d0e03524628b617671c42a73d3c458cc3395c31965a4e22295e8389f56e597'
RUNTIME_SHA256 = '0af0222908c8e1bd8543e29a2a78f8dc53ede503750464e22445f773d56cb181'
RECIPE = 'lee_alternative_meta_equal40_v0_3_1'

def runtime_sources(project):
    project = Path(project).resolve()
    tree = ast.parse((project / 'experiment_protocol.py').read_text(encoding='utf-8-sig'))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'code_hash']
    assignments = [n for n in funcs[0].body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == 'runtime' for t in n.targets)] if len(funcs) == 1 else []
    if len(assignments) != 1:
        raise ValueError('Missing literal frozen runtime source set')
    names = ast.literal_eval(assignments[0].value)
    if not isinstance(names, set) or any(not isinstance(n, str) or Path(n).name != n or
                                       '/' in n or '\\' in n or not n.endswith('.py') for n in names):
        raise ValueError('Unsafe runtime source set')
    files = [p for p in project.rglob('*.py') if
             (p.parent == project and p.name in names) or
             (p.parent.parent == project and p.parent.name in ('data', 'models', 'utils'))]
    result = {p.relative_to(project).as_posix(): file_hash(p) for p in sorted(files)}
    if len(result) != 31 or digest(result) != RUNTIME_SHA256:
        raise ValueError('Use the original frozen31 physical/runtime sources')
    return result

def implementation_sources():
    return {p.name: file_hash(p) for p in sorted(HERE.iterdir())
            if p.is_file() and p.suffix in ('.py', '.json')}

def build_config(archived, seed, microbatch):
    if type(seed) is not int or seed not in SEEDS or type(microbatch) is not int or microbatch not in (1, 2, 4, 8):
        raise ValueError('Unsupported seed or microbatch')
    cfg = copy.deepcopy(archived)
    expected = {'n_users': 2, 'n_ant': 8, 'n_subcarriers': 128, 'frame_num_symbols': 25,
                'threegpp_link_batch_size': 8}
    if any(cfg['system'].get(k) != v for k, v in expected.items()):
        raise ValueError('Physical geometry differs')
    if (cfg['data_pipeline'].get('generation_batch_size') != 32 or
            cfg['data_pipeline'].get('shuffle_buffer') is not False or
            cfg['train'].get('train_samples') != 12000 or cfg['train'].get('val_samples') != 2000 or
            cfg['train'].get('validation_seed') != 610000001):
        raise ValueError('Frozen training/validation stream differs')
    cfg['seed'], cfg['device'] = seed, 'cuda'
    cfg['train'].update(epochs=40, batch_size=32, gradient_microbatch_size=microbatch,
                        validation_microbatch_size=microbatch, amp='none', tf32=False,
                        cudnn_benchmark=False, early_stopping=False, optimizer='adam',
                        lr=0.001, weight_decay=0, betas=[0.9, 0.999], eps=1e-8,
                        scheduler='none', selection_metric='val_mse')
    cfg['lee_meta_reproduction'] = {'identifier': RECIPE, 'microbatch': microbatch,
                                    'recipe_sha256': file_hash(HERE / 'recipe.json'),
                                    'training_seed_offset': 10001, 'test_adaptation_steps': 0}
    # The archived estimator entry remains solely for generator/cache identity.
    # The trainer creates the explicit parallel model from this package.
    if cfg['system'] != archived['system'] or cfg['data_pipeline'] != archived['data_pipeline']:
        raise AssertionError('Physical source contract changed')
    return cfg


def support_sources(root):
    root = Path(root).resolve()
    require((root / 'fixed_validation.py').is_file() and (root / 'reference_replay.py').is_file()
            and (root / 'verification_extension' / 'protocol.py').is_file(), 'Incomplete verification support')
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}

def validate_manifest(path, project, seed, verification_root=None):
    path = Path(path).resolve()
    manifest = read_signed(path)
    require(manifest.get('schema') == 'dopnet_lee_research_manifest_v1'
            and manifest.get('recipe') == RECIPE and manifest.get('seeds') == list(SEEDS),
            'Use a fresh independent Lee prepared manifest')
    require(runtime_sources(project) == manifest['runtime_sources_sha256'], 'Runtime sources changed')
    require(implementation_sources() == manifest['implementation_sha256'], 'Candidate sources changed')
    support = Path(verification_root or manifest['verification_root']).resolve()
    require(support_sources(support) == manifest['verification_support_sha256'], 'Verification sources changed')
    record = manifest['configs'][f'lee_meta_seed{seed}']
    cfg_path = path.parent / record['file']
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    require(file_hash(cfg_path) == record['sha256'] and digest(cfg) == record['config_sha256'], 'Config changed')
    require(cfg['seed'] == seed and cfg['lee_meta_reproduction']['recipe_sha256'] == file_hash(HERE/'recipe.json'),
            'Wrong scientific recipe/config identity')
    return manifest, cfg, f'lee_meta_seed{seed}'
