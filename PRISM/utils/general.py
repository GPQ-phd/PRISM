from pathlib import Path
import yaml

def increment_path(path,exist_ok=False):
    path=Path(path)
    if exist_ok or not path.exists():return path
    for i in range(2,10000):
        candidate=Path(f'{path}{i}')
        if not candidate.exists():return candidate
    raise RuntimeError('No output directory available')

def check_dataset(path):
    yaml_path=Path(path).resolve()
    data=yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
    root=Path(data.get('path',yaml_path.parent))
    if not root.is_absolute():root=(yaml_path.parent/root).resolve()
    for split in ('train','val','test'):
        if split not in data:continue
        items=data[split] if isinstance(data[split],list) else [data[split]]
        data[split]=[str((root/item).resolve()) if not Path(item).is_absolute() else item for item in items]
    metadata=data.get('metadata')
    if isinstance(metadata,dict):
        data['metadata']={k:str((root/v).resolve()) if not Path(v).is_absolute() else v for k,v in metadata.items()}
    elif metadata:
        data['metadata']=str((root/metadata).resolve()) if not Path(metadata).is_absolute() else metadata
    for split in ('train','val','test'):
        key=f'{split}_metadata'
        if data.get(key):
            value=data[key];data[key]=str((root/value).resolve()) if not Path(value).is_absolute() else value
    if isinstance(data.get('names'),dict):data['names']=[data['names'][k] for k in sorted(data['names'])]
    return data
