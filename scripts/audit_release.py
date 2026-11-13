#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Scan distributable files for machine paths, network identities and secret patterns."""
import argparse
import ast
import json
from pathlib import Path
import re

SKIP={'.git','node_modules','.venv','__pycache__','runs','outputs'}
PATTERNS={
 'personal_path':re.compile(r'(?i)(?:/home/[^/\s]+|/Users/[^/\s]+|[A-Z]:[\\/]Users[\\/][^\\/\s]+)'),
 'ipv4':re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])'),
 'email':re.compile(r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'),
 'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
 'service_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{15,}|AKIA[A-Z0-9]{16})\b'),
 'secret_assignment':re.compile(r'''(?i)\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*["'][^"'\s]{8,}["']'''),
}


def audit(root):
    hits=[];files=[];imports=[]
    for path in sorted(root.rglob('*')):
        rel=path.relative_to(root)
        if any(p in SKIP or p.endswith('.egg-info') for p in rel.parts):continue
        if not path.is_file():continue
        if path.is_symlink():hits.append({'path':str(rel),'kind':'symlink'});continue
        files.append(str(rel))
        if path.suffix in ('.pkl','.bin','.npz','.log') or path.name.startswith('.env'):
            hits.append({'path':str(rel),'kind':'excluded_artifact'});continue
        try:text=path.read_text()
        except UnicodeDecodeError:hits.append({'path':str(rel),'kind':'binary'});continue
        # The scanner itself necessarily spells some detection patterns.
        for kind,pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                if kind=='ipv4' and match.group()=='0.12.0.1' and ('jax' in text or 'JAX' in text):continue  # pinned library version
                # The literal detector path is generic code, never a personal path.
                if rel.as_posix()=='scripts/audit_release.py' and kind=='personal_path':continue
                hits.append({'path':str(rel),'line':text[:match.start()].count('\n')+1,'kind':kind})
        if path.suffix=='.py':
            tree=ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
                if isinstance(node,ast.ImportFrom):imports.append(node.module or '')
        if path.suffix in ('.ts','.py') and rel.as_posix()!='scripts/audit_release.py':
            for pattern in (r'src/server',r'engine\.jax_battle',r'engine\.jax_fast',r'data/private',r'foul[-_]play',r'poke[-_]engine'):
                if re.search(pattern,text,re.I):hits.append({'path':str(rel),'kind':'excluded_dependency'})
    return {'passed':not hits,'files':len(files),'inventory':files,'hits':hits,'python_imports':sorted(set(imports))}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path)
    a=p.parse_args();result=audit(a.root)
    raw=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.write_text(raw)
    print(raw,end='');raise SystemExit(0 if result['passed'] else 1)
