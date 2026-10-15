"""Compatibility guard for historical NMSE-only score cells (no fake bits)."""
import ast
from contextlib import contextmanager
import inspect
from pathlib import Path


ORIGINAL_WORKFLOW_SHA256 = 'b53c0d348d3617b1cb06a714818da48d35446e52b5c6ff21744ca918c9e453e3'


def guarded_function(original):
    """Guard exactly the historical bits load, with all BER logic unchanged."""
    tree = ast.parse(inspect.getsource(original))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == 'old_bits']
    if len(assignments) != 1:
        raise RuntimeError('Historical score adapter does not recognize this evaluator')
    node = assignments[0]
    def is_original_bits(item):
        if not (isinstance(item, ast.Subscript)
                and isinstance(item.value, ast.Name) and item.value.id == 'original'):
            return False
        key = item.slice.value if isinstance(item.slice, ast.Index) else item.slice
        try:
            return ast.literal_eval(key) == 'bits'
        except (ValueError, TypeError, SyntaxError):
            return False
    if not any(is_original_bits(item) for item in ast.walk(node.value)):
        raise RuntimeError('Historical BER load changed; schema adapter rejected')
    node.value = ast.IfExp(test=ast.parse("cell['ber']", mode='eval').body,
                           body=node.value, orelse=ast.parse('None', mode='eval').body)
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, str(Path(__file__).resolve()) + ':NMSE_only_guard', 'exec'),
         original.__globals__, namespace)
    return namespace[original.__name__]


@contextmanager
def installed():
    from verification_extension import workflow as W, protocol as P
    if P.file_hash(W.__file__) != ORIGINAL_WORKFLOW_SHA256:
        raise RuntimeError('Unrecognized calibrated evaluator; schema adapter rejected')
    previous = W.evaluate
    W.evaluate = guarded_function(previous)
    try:
        yield
    finally:
        W.evaluate = previous
