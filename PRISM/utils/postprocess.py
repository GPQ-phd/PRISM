import torch
from torch.nn import functional as F
from utils.loss import decode_all


def nms(boxes,scores,threshold=.5):
    if len(scores)==0:return torch.empty(0,device=scores.device,dtype=torch.long)
    indices=scores.argsort(descending=True); keep=[]
    while indices.numel():
        idx=indices[0];keep.append(idx)
        if indices.numel()==1:break
        b=boxes[idx];others=boxes[indices[1:]]
        x1=torch.maximum(b[0],others[:,0]);y1=torch.maximum(b[1],others[:,1]);x2=torch.minimum(b[2],others[:,2]);y2=torch.minimum(b[3],others[:,3])
        inter=(x2-x1).clamp_min(0)*(y2-y1).clamp_min(0)
        area=(b[2]-b[0]).clamp_min(0)*(b[3]-b[1]).clamp_min(0)
        other_area=(others[:,2]-others[:,0]).clamp_min(0)*(others[:,3]-others[:,1]).clamp_min(0)
        iou=inter/(area+other_area-inter+1e-7)
        indices=indices[1:][iou<=threshold]
    return torch.stack(keep)

@torch.no_grad()
def postprocess(outputs,confidence=.3,iou=.5,max_candidates=1000,max_detections=300):
    batch=outputs[0]['cls'].shape[0];result=[]
    for bi in range(batch):
        boxes=[];scores=[];classes=[]
        for pred in outputs:
            cls=pred['cls'][bi].sigmoid().flatten(1)
            obj=pred['obj'][bi].sigmoid().flatten()
            scored=cls*obj.unsqueeze(0)
            inds=(scored>confidence).nonzero(as_tuple=False)
            if not inds.numel():continue
            class_ids=inds[:,0];loc_ids=inds[:,1]
            sc=scored[class_ids,loc_ids]
            rect=decode_all({k:v[bi:bi+1] for k,v in pred.items()},None)[0].flatten(1).T
            boxes.append(rect[loc_ids]);scores.append(sc);classes.append(class_ids)
        if not boxes:
            result.append({'boxes':torch.empty((0,4)),'scores':torch.empty(0),'labels':torch.empty(0,dtype=torch.long)});continue
        boxes=torch.cat(boxes);scores=torch.cat(scores);classes=torch.cat(classes)
        if len(scores)>max_candidates:
            top=scores.topk(max_candidates).indices;boxes=boxes[top];scores=scores[top];classes=classes[top]
        selected=[]
        for cls in classes.unique():
            ids=(classes==cls).nonzero().flatten()
            selected.extend(ids[nms(boxes[ids],scores[ids],iou)].tolist())
        selected=torch.tensor(selected,device=scores.device,dtype=torch.long)
        selected=selected[scores[selected].argsort(descending=True)[:max_detections]]
        result.append({'boxes':boxes[selected],'scores':scores[selected],'labels':classes[selected]})
    return result
