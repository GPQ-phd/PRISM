"""YOLO normalized xywh labels with optional per-image flight metadata."""
import csv
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SUFFIXES={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

class YOLODataset(Dataset):
    def __init__(self,images,labels=None,size=640,augment=False,hyp=None,single_cls=False,metadata=None):
        self.images=[]
        sources=images if isinstance(images,(list,tuple)) else [images]
        for source in sources:
            source=Path(source)
            if source.is_file() and source.suffix.lower()=='.txt':
                entries=[Path(p.strip()) for p in source.read_text().splitlines() if p.strip()]
                entries=[p if p.is_absolute() else source.parent/p for p in entries]
                self.images.extend(entries)
            else:self.images.extend(p for p in source.rglob('*') if p.suffix.lower() in SUFFIXES)
        self.images=sorted(set(self.images));self.labels=Path(labels) if labels else None
        self.size=size;self.augment=augment;self.hyp=hyp or {};self.single_cls=single_cls
        self.metadata=self._load_metadata(metadata)
        if not self.images:raise ValueError(f'No images found: {images}')
    @staticmethod
    def _load_metadata(source):
        if source is None:return {}
        if isinstance(source,dict):return source
        path=Path(source)
        if not path.exists():raise FileNotFoundError(f'Metadata CSV not found: {path}')
        records={}
        with path.open(newline='',encoding='utf-8-sig') as stream:
            for raw in csv.DictReader(stream):
                row={str(k).strip().lower():v for k,v in raw.items()}
                key=next((row.get(k) for k in ('image','image_path','path','filename','file','name') if row.get(k)),None)
                height=next((row.get(k) for k in ('altitude','height','flight_altitude','flight_height') if row.get(k) not in (None,'')),None)
                angle=next((row.get(k) for k in ('viewing_angle','view_angle','angle','camera_angle') if row.get(k) not in (None,'')),None)
                if key is None or height is None or angle is None:continue
                try:record={'height':float(height),'angle':float(angle)}
                except ValueError:continue
                item=Path(key)
                for alias in dict.fromkeys((str(item),item.name,item.stem)):
                    if alias in records:
                        raise ValueError(f'Ambiguous duplicate metadata key: {alias}')
                    records[alias]=record
        return records
    def image_metadata(self,path):
        record=next((self.metadata[k] for k in (str(path),path.name,path.stem) if k in self.metadata),None)
        if record is None:return float('nan'),float('nan'),False
        return float(record['height']),float(record['angle']),True
    def __len__(self):return len(self.images)
    def label_file(self,path):
        if self.labels is not None:return self.labels/(path.stem+'.txt')
        parts=list(path.parts)
        if 'images' in parts:
            i=len(parts)-1-parts[::-1].index('images');parts[i]='labels';return Path(*parts).with_suffix('.txt')
        return path.with_suffix('.txt')
    def __getitem__(self,i):
        path=self.images[i]
        with Image.open(path) as image:
            image=image.convert('RGB').resize((self.size,self.size)); array=np.asarray(image).copy()
        boxes=[];classes=[];labelfile=self.label_file(path)
        if labelfile.exists():
            for line in labelfile.read_text().splitlines():
                v=line.strip().split()
                if len(v)<5:continue
                c,cx,cy,w,h=map(float,v[:5]);classes.append(0 if self.single_cls else int(c))
                boxes.append([max(0,cx-w/2),max(0,cy-h/2),min(1,cx+w/2),min(1,cy+h/2)])
        boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
        labels=torch.tensor(classes,dtype=torch.long)
        if self.augment and random.random()<self.hyp.get('fliplr',0.5):
            array=np.ascontiguousarray(array[:,::-1]); boxes[:,[0,2]]=1-boxes[:,[2,0]]
        if self.augment and random.random()<self.hyp.get('flipud',0):
            array=np.ascontiguousarray(array[::-1]);boxes[:,[1,3]]=1-boxes[:,[3,1]]
        x=torch.from_numpy(array.copy()).permute(2,0,1).float().div_(255)
        height,angle,has_metadata=self.image_metadata(path)
        return x,{'boxes':boxes,'labels':labels,'path':str(path),
                  'height':torch.tensor(height,dtype=torch.float32),
                  'angle':torch.tensor(angle,dtype=torch.float32),
                  'has_metadata':torch.tensor(has_metadata,dtype=torch.bool)}

def collate(batch):return torch.stack([a for a,_ in batch]),[b for _,b in batch]

def split_metadata(data,split):
    metadata=data.get('metadata')
    if isinstance(metadata,dict):return metadata.get(split)
    return data.get(f'{split}_metadata',metadata)
