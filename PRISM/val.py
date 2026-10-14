import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from models.prism import PRISM
from utils.datasets import YOLODataset,collate,split_metadata
from utils.general import check_dataset
from utils.metrics import calculate_map
from utils.postprocess import postprocess

@torch.no_grad()
def validate(model,loader,device,nc,conf_thres=.001,iou_thres=.6):
    model.eval();records=[]
    for images,targets in loader:
        height=torch.stack([t['height'] for t in targets]).to(device)
        angle=torch.stack([t['angle'] for t in targets]).to(device)
        has_metadata=torch.stack([t['has_metadata'] for t in targets]).to(device)
        outputs=model(images.to(device),height=height,angle=angle,has_metadata=has_metadata)
        detections=postprocess(outputs,confidence=conf_thres,iou=iou_thres)
        for det,tar in zip(detections,targets):
            records.append(({k:v.detach().cpu() for k,v in det.items()},tar))
    return calculate_map(records,nc)

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--data',required=True)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--img-size',type=int,default=640)
    p.add_argument('--device',default='0');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--conf-thres',type=float,default=.001);p.add_argument('--iou-thres',type=float,default=.6)
    p.add_argument('--metadata-free',action='store_true')
    a=p.parse_args();device=torch.device('cpu' if a.device=='cpu' or not torch.cuda.is_available() else 'cuda:'+a.device.split(',')[0])
    ckpt=torch.load(a.weights,map_location='cpu',weights_only=False);nc=ckpt['nc'];model=PRISM(nc,**ckpt.get('model_cfg',{})).to(device)
    model.load_state_dict(ckpt.get('ema') or ckpt['model']);data=check_dataset(a.data)
    metadata=None if a.metadata_free else split_metadata(data,'val')
    loader=DataLoader(YOLODataset(data['val'],size=a.img_size,metadata=metadata),batch_size=a.batch_size,num_workers=a.workers,collate_fn=collate)
    print(validate(model,loader,device,nc,a.conf_thres,a.iou_thres))
if __name__=='__main__':main()
