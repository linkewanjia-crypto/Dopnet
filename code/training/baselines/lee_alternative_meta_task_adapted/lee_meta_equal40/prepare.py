"""Prepare signed Lee A-ML task-adapted configs; never import a tensor runtime."""
import argparse
import ast
import copy
import json
from pathlib import Path
from .artifacts import digest, file_hash, read_signed, sign, write_json

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


def validate_manifest(path, project, seed):
    path = Path(path).resolve()
    manifest = read_signed(path)
    if manifest.get('recipe') != RECIPE or manifest.get('seeds') != list(SEEDS):
        raise ValueError('Wrong experiment identity')
    if runtime_sources(project) != manifest['runtime_sources_sha256']:
        raise ValueError('Runtime sources changed')
    if implementation_sources() != manifest['implementation_sha256']:
        raise ValueError('Lee candidate sources changed')
    record = manifest['configs'][f'lee_meta_seed{seed}']
    cfg = json.loads((path.parent / record['file']).read_text(encoding='utf-8'))
    if file_hash(path.parent / record['file']) != record['sha256'] or digest(cfg) != record['config_sha256']:
        raise ValueError('Prepared config changed')
    return manifest, cfg, f'lee_meta_seed{seed}'


def prepare(project, reference, validation_cache, output, microbatch):
    project, reference, output = [Path(p).resolve() for p in (project, reference, output)]
    if output == reference or reference in output.parents or output == project or project in output.parents:
        raise ValueError('Use independent output outside archived scientific directories')
    experiment, evaluation = read_signed(reference / 'experiment.json'), read_signed(reference / 'evaluation.json')
    sources = runtime_sources(project)
    if (experiment['code_sha256'] != RUNTIME_SHA256 or
            evaluation['experiment_signature'] != experiment['signature'] or
            experiment['validation_cache_sha256'] != VALIDATION_SHA256):
        raise ValueError('Original signed experiment provenance differs')
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for seed in SEEDS:
        cfg = build_config(experiment['configs'][f'lee3ddual_seed{seed}'], seed, microbatch)
        relative = f'configs/lee_meta_seed{seed}.json'
        path = output / relative
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != cfg:
            raise ValueError('Refuse to overwrite another config')
        if not path.exists():
            write_json(path, cfg)
        records[f'lee_meta_seed{seed}'] = {'file': relative, 'sha256': file_hash(path), 'config_sha256': digest(cfg)}
    manifest = sign({'schema': 'lee_equal40_manifest_v1', 'recipe': RECIPE, 'seeds': list(SEEDS),
                     'reference_experiment_signature': experiment['signature'],
                     'reference_evaluation_signature': evaluation['signature'],
                     'runtime_sources_sha256': sources, 'implementation_sha256': implementation_sources(),
                     'validation_cache_sha256': VALIDATION_SHA256,
                     'validation': {'path': str(Path(validation_cache).resolve()), 'bytes': 4682930065,
                                    'frames': 2000, 'optimizer_use': False}, 'configs': records})
    path = output / 'lee_manifest.json'
    if path.exists() and read_signed(path) != manifest:
        raise ValueError('Refuse to overwrite another manifest')
    if not path.exists():
        write_json(path, manifest)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'reference-results', 'validation-cache', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--microbatch', type=int, choices=(1, 2, 4, 8), required=True)
    args = parser.parse_args()
    print(prepare(args.project, args.reference_results, args.validation_cache, args.output, args.microbatch))


if __name__ == '__main__':
    main()
