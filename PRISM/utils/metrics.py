"""Dataset-wide greedy matching and AP at IoU .50:.95 (COCO-like baseline)."""
import numpy as np
import torch

def box_iou(a,b):
    if a.numel()==0 or b.numel()==0:return a.new_zeros((len(a),len(b)))
    lt=torch.maximum(a[:,None,:2],b[None,:,:2]);rb=torch.minimum(a[:,None,2:],b[None,:,2:])
    area=lambda x:((x[:,2:]-x[:,:2]).clamp_min(0)).prod(-1)
    inter=(rb-lt).clamp_min(0).prod(-1)
    return inter/(area(a)[:,None]+area(b)[None,:]-inter+1e-9)

def calculate_map(records,nc):
    thresholds=np.arange(.5,.951,.05);aps=np.zeros((nc,len(thresholds)));prec=[];rec=[]
    for cls in range(nc):
        gt_counts=sum(int((t['labels']==cls).sum()) for _,t in records)
        if not gt_counts:continue
        scores=[];matches=[]
        for prediction,target in records:
            d=(prediction['labels']==cls).nonzero().flatten();g=(target['labels']==cls).nonzero().flatten()
            if not len(d):continue
            order=prediction['scores'][d].argsort(descending=True);d=d[order]
            ious=box_iou(prediction['boxes'][d],target['boxes'][g]);used=[set() for _ in thresholds]
            for j,index in enumerate(d):
                scores.append(float(prediction['scores'][index])); matched=[]
                for n,thr in enumerate(thresholds):
                    if not len(g):matched.append(0);continue
                    values=ious[j].clone()
                    if used[n]:values[list(used[n])]=-1
                    best=int(values.argmax());ok=float(values[best])>=thr
                    if ok:used[n].add(best)
                    matched.append(int(ok))
                matches.append(matched)
        if not scores:continue
        order=np.argsort(-np.asarray(scores));tp=np.asarray(matches)[order]
        for j in range(len(thresholds)):
            t=np.cumsum(tp[:,j]);fp=np.cumsum(1-tp[:,j]);p=t/(t+fp+1e-9);r=t/gt_counts
            x=np.concatenate(([0.],r,[1.]));y=np.concatenate(([1.],p,[0.]));y=np.maximum.accumulate(y[::-1])[::-1]
            aps[cls,j]=np.trapz(np.interp(np.linspace(0,1,101),x,y),np.linspace(0,1,101))
        t=np.cumsum(tp[:,0]);p=t/(np.arange(len(t))+1);r=t/gt_counts
        prec.append(float(p[-1]));rec.append(float(r[-1]))
    valid=[c for c in range(nc) if any(int((t['labels']==c).sum()) for _,t in records)]
    return {'precision':float(np.mean(prec)) if prec else 0.,'recall':float(np.mean(rec)) if rec else 0.,'map50':float(aps[valid,0].mean()) if valid else 0.,'map50_95':float(aps[valid].mean()) if valid else 0.}
