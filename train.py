"""PRISM training, adapted from train_SeaDroneSee.py CLI and YOLOv5-style runs.
RGB-only PRISM: this file intentionally does not call create_dataloader_rgb_ir
or YOLOv5's anchor-specific ComputeLoss.
"""
import argparse,csv,math,random,time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from models.prism import PRISM
from utils.datasets import YOLODataset,collate,split_metadata
from utils.general import check_dataset,increment_path
from utils.loss import prism_loss
from val import validate

class ModelEMA:
    def __init__(self,model,decay=.9999):
        self.ema=deepcopy(model).eval();self.updates=0;self.decay=decay
        for p in self.ema.parameters():p.requires_grad_(False)
    @torch.no_grad()
    def update(self,model):
        self.updates+=1;d=self.decay*(1-math.exp(-self.updates/2000))
        ms=model.state_dict();es=self.ema.state_dict()
        for k,v in es.items():
            if v.is_floating_point():v.mul_(d).add_(ms[k].detach(),alpha=1-d)
            else:v.copy_(ms[k])

def parse_opt():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=str,default='',help='PRISM checkpoint')
    p.add_argument('--cfg',default='models/prism.yaml');p.add_argument('--data',default='data/multispectral/SeaDroneSee.yaml')
    p.add_argument('--hyp',default='data/hyp.scratch.yaml')
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--img-size',nargs='+',type=int,default=[640,640]);p.add_argument('--device',default='0')
    p.add_argument('--workers',type=int,default=8);p.add_argument('--project',default='runs/train')
    p.add_argument('--name',default='PRISM_SDS');p.add_argument('--exist-ok',action='store_true')
    p.add_argument('--resume',nargs='?',const=True,default=False);p.add_argument('--single-cls',action='store_true')
    p.add_argument('--adam',action='store_true',help='deprecated: AdamW is already the default')
    p.add_argument('--sgd',action='store_true');p.add_argument('--linear-lr',action='store_true')
    p.add_argument('--nosave',action='store_true');p.add_argument('--notest',action='store_true')
    p.add_argument('--save-period',type=int,default=-1);p.add_argument('--accumulate',type=int,default=0)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--conf-thres',type=float,default=.001)
    p.add_argument('--iou-thres',type=float,default=.6)
    p.add_argument('--amp',dest='amp',action='store_true');p.add_argument('--no-amp',dest='amp',action='store_false')
    p.set_defaults(amp=True)
    p.add_argument('--metadata-free',action='store_true',help='ignore metadata and use the learned default embedding')
    p.add_argument('--cache-images',action='store_true',help='reserved; not implemented')
    p.add_argument('--rect',action='store_true',help='reserved; square resizing used')
    p.add_argument('--multi-scale',action='store_true',help='not supported (fixed pyramid geometry)')
    p.add_argument('--noautoanchor',action='store_true',help='accepted; PRISM is anchor-free')
    return p.parse_args()

def main(opt=None):
    opt=opt or parse_opt();random.seed(opt.seed);np.random.seed(opt.seed);torch.manual_seed(opt.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(opt.seed)
    if opt.multi_scale:raise ValueError('PRISM uses a fixed 4-level pyramid; multi-scale training is not implemented')
    if opt.img_size[0]%16:raise ValueError('--img-size must be divisible by 16')
    device=torch.device('cpu' if opt.device=='cpu' or not torch.cuda.is_available() else 'cuda:'+opt.device.split(',')[0]);cuda=device.type=='cuda'
    cfg=yaml.safe_load(Path(opt.cfg).read_text());hyp=yaml.safe_load(Path(opt.hyp).read_text());data=check_dataset(opt.data)
    nc=1 if opt.single_cls else int(data['nc']);names=['item'] if opt.single_cls else data['names'];assert len(names)==nc
    save_dir=increment_path(Path(opt.project)/opt.name,opt.exist_ok);save_dir.mkdir(parents=True,exist_ok=True)
    wdir=save_dir/'weights';wdir.mkdir(exist_ok=True);last=wdir/'last.pt';best=wdir/'best.pt'
    (save_dir/'opt.yaml').write_text(yaml.safe_dump(vars(opt),allow_unicode=True));(save_dir/'hyp.yaml').write_text(yaml.safe_dump(hyp))
    train_metadata=None if opt.metadata_free else split_metadata(data,'train')
    val_metadata=None if opt.metadata_free else split_metadata(data,'val')
    train_set=YOLODataset(data['train'],size=opt.img_size[0],augment=True,hyp=hyp,single_cls=opt.single_cls,metadata=train_metadata)
    val_set=YOLODataset(data['val'],size=opt.img_size[-1],single_cls=opt.single_cls,metadata=val_metadata)
    train_loader=DataLoader(train_set,batch_size=opt.batch_size,shuffle=True,num_workers=opt.workers,collate_fn=collate,pin_memory=cuda)
    val_loader=DataLoader(val_set,batch_size=max(1,opt.batch_size),shuffle=False,num_workers=opt.workers,collate_fn=collate,pin_memory=cuda)
    model=PRISM(nc,**cfg).to(device)
    decay=float(hyp.get('weight_decay',.01));lr=float(hyp.get('lr0',1e-4))
    norm=[];other=[]
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        (norm if name.endswith('.bias') or 'norm' in name or '.n1.' in name or '.n2.' in name else other).append(param)
    parameters=[{'params':other,'weight_decay':decay},{'params':norm,'weight_decay':0.}]
    optimizer=(torch.optim.SGD(parameters,lr=lr,momentum=hyp.get('momentum',.9),nesterov=True)
               if opt.sgd else torch.optim.AdamW(parameters,lr=lr,betas=(hyp.get('beta1',.9),.999)))
    ema=ModelEMA(model);start_epoch=0;best_fitness=-1.
    ckpt_path=None
    if opt.resume:
        ckpt_path=Path(opt.weights) if isinstance(opt.resume,bool) and opt.weights else (last if isinstance(opt.resume,bool) else Path(opt.resume))
    elif opt.weights:ckpt_path=Path(opt.weights)
    if ckpt_path:
        checkpoint=torch.load(ckpt_path,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model'],strict=True)
        if opt.resume:
            optimizer.load_state_dict(checkpoint['optimizer']);start_epoch=checkpoint['epoch']+1;best_fitness=checkpoint.get('best_fitness',-1.)
            if checkpoint.get('ema'):ema.ema.load_state_dict(checkpoint['ema']);ema.updates=checkpoint.get('updates',0)
        else:ema.ema.load_state_dict(model.state_dict())
    if hasattr(torch,'amp') and hasattr(torch.amp,'GradScaler'):
        try:scaler=torch.amp.GradScaler('cuda',enabled=(cuda and opt.amp))
        except TypeError:scaler=torch.cuda.amp.GradScaler(enabled=(cuda and opt.amp))
    else:scaler=torch.cuda.amp.GradScaler(enabled=(cuda and opt.amp))
    nb=len(train_loader)
    # Preserve the paper's effective batch size of 8 on memory-limited GPUs.
    accumulate=opt.accumulate or max(round(8/opt.batch_size),1)
    warmup_steps=max(1,round(float(hyp.get('warmup_epochs',3))*nb))
    def factor(epoch):
        if opt.linear_lr:return 1-(1-float(hyp.get('lrf',.01)))*epoch/max(1,opt.epochs-1)
        return .1 if epoch>=int(hyp.get('lr_decay_epoch',80)) else 1.
    csv_path=save_dir/'results.csv';fields=['epoch','train/box_loss','train/obj_loss','train/cls_loss','train/wav_loss','train/prior_loss','train/total_loss','metrics/precision','metrics/recall','metrics/mAP_0.5','metrics/mAP_0.5:0.95','x/lr0']
    if not csv_path.exists():
        with csv_path.open('w',newline='') as f:csv.writer(f).writerow(fields)
    print(f'PRISM: {nc} classes, {len(train_set)} training, {len(val_set)} val, device={device}, save_dir={save_dir}',flush=True)
    for epoch in range(start_epoch,opt.epochs):
        model.train();averages=np.zeros(6);optimizer.zero_grad(set_to_none=True)
        pbar=tqdm(enumerate(train_loader),total=nb,desc=f'Epoch {epoch+1}/{opt.epochs}')
        for i,(images,targets) in pbar:
            step=epoch*nb+i; warmup=min(1.,(step+1)/warmup_steps)
            current_lr=lr*factor(epoch)*warmup
            for group in optimizer.param_groups:group['lr']=current_lr
            images=images.to(device,non_blocking=True)
            height=torch.stack([t['height'] for t in targets]).to(device)
            angle=torch.stack([t['angle'] for t in targets]).to(device)
            has_metadata=torch.stack([t['has_metadata'] for t in targets]).to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(cuda and opt.amp)):
                aux = model(images,height=height,angle=angle,has_metadata=has_metadata,return_aux=True)
                losses = prism_loss(aux, targets, images.shape[-2:])
                loss = losses['total'] / accumulate
            scaler.scale(loss).backward()
            if (i+1)%accumulate==0 or i==nb-1:
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);ema.update(model)
            values=np.array([float(losses[k].detach()) for k in ('box','obj','cls','wav','prior','total')]);averages+=values
            pbar.set_postfix(loss=f'{values[-1]:.3f}',lr=f'{current_lr:.2g}')
        averages/=nb
        metrics={'precision':0.,'recall':0.,'map50':0.,'map50_95':0.}
        if not opt.notest or epoch==opt.epochs-1:metrics=validate(ema.ema,val_loader,device,nc,opt.conf_thres,opt.iou_thres)
        fitness=metrics['map50_95'];improved=fitness>best_fitness;best_fitness=max(best_fitness,fitness)
        row=[epoch,*averages.tolist(),metrics['precision'],metrics['recall'],metrics['map50'],metrics['map50_95'],current_lr]
        with csv_path.open('a',newline='') as f:csv.writer(f).writerow(row)
        print(f'Epoch {epoch+1} train_total={averages[-1]:.4f} P={metrics["precision"]:.4f} R={metrics["recall"]:.4f} mAP50={metrics["map50"]:.4f} mAP50-95={fitness:.4f}',flush=True)
        if not opt.nosave or epoch==opt.epochs-1:
            state={'epoch':epoch,'model':model.state_dict(),'ema':ema.ema.state_dict(),'updates':ema.updates,'optimizer':optimizer.state_dict(),'nc':nc,'names':names,'model_cfg':cfg,'best_fitness':best_fitness}
            torch.save(state,last)
            if improved:torch.save(state,best)
            if opt.save_period>0 and (epoch+1)%opt.save_period==0:torch.save(state,wdir/f'epoch{epoch+1}.pt')
    print(f'Training complete: {save_dir}')
if __name__=='__main__':main()
