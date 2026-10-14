import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from models.prism import PRISM
from utils.postprocess import postprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--source',required=True)
    p.add_argument('--img-size',type=int,default=640);p.add_argument('--conf-thres',type=float,default=.25)
    p.add_argument('--iou-thres',type=float,default=.5);p.add_argument('--device',default='0')
    p.add_argument('--height',type=float,default=None,help='optional flight altitude')
    p.add_argument('--angle',type=float,default=None,help='optional viewing angle in degrees')
    p.add_argument('--project',default='runs/detect');p.add_argument('--name',default='exp');a=p.parse_args()
    device=torch.device('cpu' if a.device=='cpu' or not torch.cuda.is_available() else 'cuda:'+a.device.split(',')[0]);ckpt=torch.load(a.weights,map_location='cpu',weights_only=False)
    model=PRISM(ckpt['nc'],**ckpt.get('model_cfg',{})).to(device).eval();model.load_state_dict(ckpt.get('ema') or ckpt['model'])
    source=Path(a.source);files=[source] if source.is_file() else sorted(p for p in source.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'))
    out=Path(a.project)/a.name;out.mkdir(parents=True,exist_ok=True)
    for path in files:
        image=Image.open(path).convert('RGB');x=torch.from_numpy(np.asarray(image.resize((a.img_size,a.img_size))).copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255
        with torch.no_grad():det=postprocess(model(x,height=a.height,angle=a.angle),confidence=a.conf_thres,iou=a.iou_thres)[0]
        draw=ImageDraw.Draw(image);w,h=image.size
        for box,score,label in zip(det['boxes'].cpu(),det['scores'].cpu(),det['labels'].cpu()):
            l,t,r,b=box.tolist();draw.rectangle((l*w,t*h,r*w,b*h),outline='red',width=2);draw.text((l*w,t*h),f'{int(label)} {float(score):.2f}',fill='red')
        dest=out/path.name;image.save(dest);print(dest,len(det['boxes']))
if __name__=='__main__':main()
