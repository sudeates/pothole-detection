"""Statically inspect PyTorch zip checkpoints; never unpickle or import them.

This is a conservative allowlist check, not a proof that a pickle is harmless.
Unknown or dynamically resolved globals are reported as suspicious.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickletools
import zipfile


UNKNOWN = object()
MARK = object()
ALLOWED_BUILTINS = {'set', 'slice', 'getattr', 'frozenset'}
DENIED_EXACT = {'torch.serialization._load_from_bytes', 'torch.load',
                'torch.hub.load', 'torch.package.PackageImporter'}
PUSH_UNKNOWN = {
    'NONE', 'NEWTRUE', 'NEWFALSE', 'BININT', 'BININT1', 'BININT2', 'INT', 'LONG',
    'LONG1', 'LONG4', 'FLOAT', 'BINFLOAT', 'SHORT_BINBYTES', 'BINBYTES',
    'BINBYTES8', 'BYTEARRAY8', 'NEXT_BUFFER', 'EMPTY_LIST', 'EMPTY_DICT',
    'EMPTY_SET', 'SHORT_BINSTRING', 'BINSTRING', 'STRING',
}
PUSH_STRING = {'SHORT_BINUNICODE', 'BINUNICODE', 'BINUNICODE8', 'UNICODE'}


def allowed(reference: str) -> bool:
    if reference in DENIED_EXACT:
        return False
    if reference == 'collections.OrderedDict':
        return True
    module, _, name = reference.rpartition('.')
    if module in ('builtins', '__builtin__'):
        return name in ALLOWED_BUILTINS
    return reference.startswith(('torch.', 'ultralytics.nn.',
                                 'ultralytics.utils.', 'numpy.'))


def static_globals(payload: bytes) -> dict:
    stack = []
    memo = {}
    references = Counter()
    reducers = Counter()
    unresolved = []

    def pop():
        if not stack:
            unresolved.append('stack_underflow')
            return UNKNOWN
        return stack.pop()

    def pop_mark():
        items = []
        while stack and stack[-1] is not MARK:
            items.append(stack.pop())
        if not stack:
            unresolved.append('missing_MARK')
        else:
            stack.pop()
        return items[::-1]

    for opcode, arg, position in pickletools.genops(payload):
        name = opcode.name
        if name in PUSH_STRING:
            stack.append(str(arg))
        elif name in PUSH_UNKNOWN:
            stack.append(UNKNOWN)
        elif name == 'GLOBAL':
            module, symbol = str(arg).split(' ', 1)
            reference = f'{module}.{symbol}'
            references[reference] += 1
            stack.append(('global', reference))
        elif name == 'STACK_GLOBAL':
            symbol, module = pop(), pop()
            if isinstance(module, str) and isinstance(symbol, str):
                reference = f'{module}.{symbol}'
                references[reference] += 1
                stack.append(('global', reference))
            else:
                unresolved.append(f'STACK_GLOBAL@{position}')
                stack.append(UNKNOWN)
        elif name == 'MARK':
            stack.append(MARK)
        elif name == 'EMPTY_TUPLE':
            stack.append(())
        elif name in ('TUPLE', 'LIST', 'DICT', 'FROZENSET'):
            values = pop_mark()
            stack.append(tuple(values) if name == 'TUPLE' else UNKNOWN)
        elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
            count = int(name[-1])
            values = [pop() for _ in range(count)][::-1]
            stack.append(tuple(values))
        elif name in ('BINPUT', 'LONG_BINPUT', 'PUT'):
            memo[int(arg)] = stack[-1] if stack else UNKNOWN
        elif name == 'MEMOIZE':
            memo[len(memo)] = stack[-1] if stack else UNKNOWN
        elif name in ('BINGET', 'LONG_BINGET', 'GET'):
            stack.append(memo.get(int(arg), UNKNOWN))
        elif name == 'REDUCE':
            args = pop()
            func = pop()
            if isinstance(func, tuple) and len(func) == 2 and func[0] == 'global':
                reducers[func[1]] += 1
                if func[1] in ('builtins.getattr', '__builtin__.getattr'):
                    if not isinstance(args, tuple) or len(args) != 2 or not isinstance(args[1], str):
                        unresolved.append(f'dynamic_getattr@{position}')
                    elif args[1].startswith('__'):
                        unresolved.append(f'dunder_getattr:{args[1]}@{position}')
            else:
                unresolved.append(f'dynamic_REDUCE@{position}')
            stack.append(UNKNOWN)
        elif name == 'NEWOBJ':
            pop()  # args
            pop()  # class
            stack.append(UNKNOWN)
        elif name == 'NEWOBJ_EX':
            pop()  # kwargs
            pop()  # args
            pop()  # class
            stack.append(UNKNOWN)
        elif name == 'BUILD':
            pop()  # state; object remains
        elif name in ('BINPERSID',):
            pop()
            stack.append(UNKNOWN)
        elif name == 'PERSID':
            stack.append(UNKNOWN)
        elif name == 'APPEND':
            pop()
        elif name == 'SETITEM':
            pop()
            pop()
        elif name in ('APPENDS', 'SETITEMS', 'ADDITEMS'):
            pop_mark()  # parent container remains
        elif name == 'POP':
            pop()
        elif name == 'POP_MARK':
            pop_mark()
        elif name == 'DUP':
            stack.append(stack[-1] if stack else UNKNOWN)
        elif name in ('PROTO', 'FRAME', 'STOP', 'READONLY_BUFFER'):
            pass
        elif name in ('EXT1', 'EXT2', 'EXT4', 'INST', 'OBJ'):
            unresolved.append(f'{name}@{position}')
            stack.append(UNKNOWN)
        else:
            unresolved.append(f'unhandled_{name}@{position}')

    suspicious = sorted(ref for ref in references if not allowed(ref))
    return dict(references=dict(sorted(references.items())),
                reducers=dict(sorted(reducers.items())),
                suspicious_references=suspicious, unresolved=unresolved,
                status='clean_allowlist' if not suspicious and not unresolved else 'suspicious')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scan(path: Path) -> dict:
    row = dict(path=str(path.resolve()), size_bytes=path.stat().st_size,
               sha256=sha256(path), status='suspicious', entries=[])
    with path.open('rb') as stream:
        header = stream.read(200)
    if header.startswith(b'version https://git-lfs.github.com/spec/v1'):
        row.update(status='lfs_pointer', reason='Git LFS pointer, not a checkpoint')
        return row
    if not zipfile.is_zipfile(path):
        row.update(reason='Not a PyTorch zip checkpoint; legacy pickle not loaded')
        return row
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith('.pkl')]
        if not any(name.endswith('/data.pkl') or name == 'data.pkl' for name in members):
            row.update(reason='No data.pkl in zip')
            return row
        for name in members:
            try:
                result = static_globals(archive.read(name))
            except Exception as exc:
                result = dict(status='suspicious', error=f'{type(exc).__name__}: {exc}')
            row['entries'].append(dict(name=name, **result))
    row['status'] = ('clean_allowlist' if all(entry['status'] == 'clean_allowlist'
                                              for entry in row['entries']) else 'suspicious')
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = [scan(path) for path in args.paths]
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()
