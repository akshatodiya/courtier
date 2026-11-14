"""Durable, identity-bound checkpoint publication; independent of tensor format.

The caller supplies relative artifact names after all ranks finish writing and
a distributed barrier confirms success. Filesystem crash guarantees depend on
the underlying filesystem and must be verified on PARAM.
This module neither saves nor restores FSDP, optimizer or RNG state.
"""
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _file(root, relative):
    name=Path(relative)
    if name.is_absolute() or not name.parts or any(p in ('','.','..') for p in relative.split('/')):
        raise ValueError('Unsafe checkpoint artifact path')
    path=root/name
    current=root
    for part in name.parts:
        current=current/part
        if current.is_symlink(): raise ValueError('Symlinks are not checkpoint artifacts')
    if not path.is_file() or path.stat().st_size==0:
        raise ValueError('Missing or empty checkpoint artifact: '+relative)
    return path


def _record(root, artifacts, identity, position):
    if not isinstance(identity,str) or len(identity)!=64 or any(c not in '0123456789abcdef' for c in identity):
        raise ValueError('Invalid run identity')
    if not isinstance(position,dict) or set(position)!={'epoch','step_in_epoch','global_step'}:
        raise ValueError('Invalid training position')
    if any(type(v) is not int or v<0 for v in position.values()):
        raise ValueError('Position values must be nonnegative integers')
    if not artifacts or len(set(artifacts))!=len(artifacts):
        raise ValueError('Artifact list must be nonempty and unique')
    records=[]
    for name in sorted(artifacts):
        if not isinstance(name,str) or name in ('COMPLETE.json','COMPLETE.json.tmp'):
            raise ValueError('Invalid checkpoint artifact name')
        p=_file(root,name)
        records.append({'path':name,'size':p.stat().st_size,'sha256':sha256(p)})
    return {'schema':'aya32-checkpoint-v1','run_identity':identity,'position':position,'files':records}


def publish(root, artifacts, identity, position):
    root=Path(root)
    if not root.is_dir() or root.is_symlink(): raise ValueError('Checkpoint directory missing or unsafe')
    marker=root/'COMPLETE.json'; temporary=root/'COMPLETE.json.tmp'
    if marker.exists() or temporary.exists(): raise FileExistsError('Checkpoint already published or incomplete')
    record=_record(root,artifacts,identity,position)
    # Flush artifact contents and their directory entries before the marker.
    # Writers must already be closed and no rank may mutate files after barrier.
    directories={root}
    for name in artifacts:
        path=_file(root,name)
        with path.open('rb') as artifact:
            os.fsync(artifact.fileno())
        parent=path.parent
        while parent!=root:
            directories.add(parent)
            parent=parent.parent
    for directory in sorted(directories,key=lambda p:len(p.parts),reverse=True):
        directory_fd=os.open(directory,os.O_RDONLY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    with temporary.open('x',encoding='utf-8') as f:
        json.dump(record,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,marker)
    directory_fd=os.open(root,os.O_RDONLY)
    try: os.fsync(directory_fd)
    finally: os.close(directory_fd)
    return record


def validate(root, expected_identity, required_artifacts):
    root=Path(root)
    if root.is_symlink() or (root/'COMPLETE.json').is_symlink(): raise ValueError('Unsafe marker')
    marker=json.loads((root/'COMPLETE.json').read_text())
    if marker.get('schema')!='aya32-checkpoint-v1' or marker.get('run_identity')!=expected_identity:
        raise ValueError('Checkpoint identity or schema mismatch')
    records=marker.get('files')
    if not isinstance(records,list) or any(not isinstance(x,dict) or not isinstance(x.get('path'),str) for x in records):
        raise ValueError('Invalid artifact manifest')
    names=[x['path'] for x in records]
    if set(names)!=set(required_artifacts) or len(names)!=len(required_artifacts):
        raise ValueError('Checkpoint artifact inventory mismatch')
    current=_record(root,names,expected_identity,marker.get('position'))
    if current!=marker: raise ValueError('Checkpoint contents do not match marker')
    return marker
