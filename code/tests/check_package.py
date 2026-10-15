"""Stdlib-only packaging checks; no scientific execution or tensor loading."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    root=args.root.resolve()
    facade=root/'code/scripts/tmlcn.py'
    records=[]
    def command(values):
        if '--execute' in values:
            raise RuntimeError('Scientific execution prohibited in packaging checks')
        result=subprocess.run([sys.executable,'-B',str(facade)]+values,
            cwd=root,capture_output=True,text=True,encoding='utf-8')
        if result.returncode:
            raise RuntimeError('CLI failed: '+str(values)+' '+result.stderr)
        records.append(dict(arguments=values,returncode=result.returncode))
        return result.stdout
    command(['--help'])
    listing=json.loads(command(['list']))
    routes=listing['recipes']
    assert len(routes)==42 and listing['main_routes']==33 and listing['independent_routes']==9
    assert all(r['config_present'] and (root/r['config']).is_file() for r in routes)
    output=root.parent/'DOPNet_TMLCN_packaging_check_future_output'
    for values in (
        ['plan','--method','dopnet','--seed','42','--output',str(output)],
        ['generate','--method','dopnet','--seed','42','--epoch','1','--output',str(output)],
        ['evaluate','--method','main','--output',str(output)],
        ['prepare','--method','fortitran_budget_matched','--output',str(output)],
        ['prepare','--method','wimae_author','--output',str(output)],
        ['prepare','--method','lee_meta','--output',str(output)],
        ['plot','--output',str(output)],
        ['extract-reference','--output',str(output)],
        ['verify-validation','--output',str(output)],
        ['mechanism','--output',str(output)],
    ):
        plan=json.loads(command(values))
        assert plan['execute'] is False
        for field in ('immutable_reference','immutable_validation','source','identity_manifest'):
            if field in plan:
                assert (root/plan[field]).exists(),(field,plan[field])
        if 'command' in plan:
            assert Path(plan['cwd']).is_dir()
            for item in plan.get('pythonpath',[]):
                assert Path(item).is_dir()
            for item in plan['command']:
                if item.endswith('.py'):
                    assert Path(item).is_file(),item
    parsed=0
    for path in (root/'code').rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        parsed+=1
    config=json.loads((root/'code/config/TMLCN_RECIPES.json').read_text(encoding='utf-8-sig'))
    indexed=config['rows']
    assert len(indexed)==42
    assert {r['method'] for r in routes}=={r['method'] for r in indexed}
    assert (root/'data/results/base_scores/original_retained/REFERENCE_VIEW.json').is_file()
    assert (root/'data/weights/selected').is_dir()
    print(json.dumps(dict(status='PASS_STATIC_PACKAGING_ONLY',python_AST_files=parsed,
        recipe_slots=len(routes),CLI_checks=records,scientific_execution=False,
        torch_imported=False,checkpoint_or_tensor_deserialized=False,
        end_to_end_reproduction_verified=False),indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
