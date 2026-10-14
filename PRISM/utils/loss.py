"""Detection and regularization losses.
Engineering choices: point-based assignment, positive center cells, ltrb distances
in stride units, sigmoid independent multi-label class predictions.
"""
import torch
from torch.nn import functional as F
from models.prism import haar


def ciou(a,b,eps=1e-7):
    x1=torch.maximum(a[:,0],b[:,0]);y1=torch.maximum(a[:,1],b[:,1]);x2=torch.minimum(a[:,2],b[:,2]);y2=torch.minimum(a[:,3],b[:,3])
    inter=(x2-x1).clamp_min(0)*(y2-y1).clamp_min(0)
    area_a=(a[:,2]-a[:,0]).clamp_min(0)*(a[:,3]-a[:,1]).clamp_min(0)
    area_b=(b[:,2]-b[:,0]).clamp_min(0)*(b[:,3]-b[:,1]).clamp_min(0)
    union=area_a+area_b-inter+eps;iou=inter/union
    ac=(a[:,:2]+a[:,2:])/2;bc=(b[:,:2]+b[:,2:])/2
    distance=(ac-bc).square().sum(-1)
    bounds1=torch.minimum(a[:,:2],b[:,:2]);bounds2=torch.maximum(a[:,2:],b[:,2:])
    diagonal=(bounds2-bounds1).square().sum(-1)+eps
    ar=torch.atan((b[:,2]-b[:,0]).clamp_min(eps)/(b[:,3]-b[:,1]).clamp_min(eps))
    br=torch.atan((a[:,2]-a[:,0]).clamp_min(eps)/(a[:,3]-a[:,1]).clamp_min(eps))
    v=(4/torch.pi**2)*(ar-br).square()
    with torch.no_grad():alpha=v/(1-iou+v+eps)
    return iou-distance/diagonal-alpha*v


def focal(logits,targets,alpha=.25,gamma=2):
    prob=logits.sigmoid();bce=F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    pt=prob*targets+(1-prob)*(1-targets)
    a=alpha*targets+(1-alpha)*(1-targets)
    return (a*(1-pt).pow(gamma)*bce).sum()


def build_targets(preds,targets,image_hw):
    """targets list of dicts {boxes: N×4 normalized xyxy, labels: N}."""
    result=[];ih,iw=image_hw
    for scale,p in enumerate(preds):
        b,classes,h,w=p['cls'].shape;obj=p['obj'].new_zeros((b,1,h,w));cls=p['cls'].new_zeros((b,classes,h,w))
        box=p['box'].new_zeros((b,4,h,w));mask=torch.zeros((b,1,h,w),device=obj.device,dtype=torch.bool)
        for batch,t in enumerate(targets):
            boxes=t['boxes'].to(obj.device); labels=t['labels'].to(obj.device)
            for bb,label in zip(boxes,labels):
                cx=((bb[0]+bb[2])/2*w).floor().long().clamp(0,w-1)
                cy=((bb[1]+bb[3])/2*h).floor().long().clamp(0,h-1)
                # All four scales supervise each GT center. This is an explicit baseline choice.
                if not (0<=int(label)<classes):continue
                obj[batch,0,cy,cx]=1;cls[batch,label,cy,cx]=1;box[batch,:,cy,cx]=bb
                mask[batch,0,cy,cx]=True
        result.append((obj,cls,box,mask))
    return result


def decode_all(pred,hw):
    b,_,h,w=pred['box'].shape
    gy,gx=torch.meshgrid(torch.arange(h,device=pred['box'].device),torch.arange(w,device=pred['box'].device),indexing='ij')
    center=torch.stack([(gx+.5)/w,(gy+.5)/h],dim=0)[None]
    ltrb=F.softplus(pred['box'])
    sizes=torch.tensor([w,h,w,h],device=ltrb.device,dtype=ltrb.dtype).view(1,4,1,1)
    d=ltrb/sizes
    return torch.cat([center-d[:,:2],center+d[:,2:]],dim=1).clamp(0,1)


def background_mask(targets,shape,device):
    b,h,w=shape; bg=torch.ones((b,1,h,w),device=device)
    for i,t in enumerate(targets):
        for box in t['boxes']:
            x1,y1,x2,y2=box.tolist();xa=max(0,int(x1*w));xb=min(w,int(x2*w)+1)
            ya=max(0,int(y1*h));yb=min(h,int(y2*h)+1)
            bg[i,:,ya:yb,xa:xb]=0
    return bg


def prism_loss(aux,targets,image_hw):
    predictions=aux['outputs']; annotations=build_targets(predictions,targets,image_hw)
    total_cls=predictions[0]['cls'].new_zeros(());total_obj=total_cls.clone();total_box=total_cls.clone()
    for pred,(obj,cls,tbox,mask) in zip(predictions,annotations):
        count=obj.sum().clamp_min(1)
        total_obj=total_obj+F.binary_cross_entropy_with_logits(pred['obj'],obj,reduction='sum')/(count+obj.numel()*.01)
        total_cls=total_cls+focal(pred['cls'],cls)/count
        decoded=decode_all(pred,image_hw).permute(0,2,3,1)[mask[:,0]]
        gt=tbox.permute(0,2,3,1)[mask[:,0]]
        if decoded.numel():total_box=total_box+(1-ciou(decoded,gt)).mean()+F.l1_loss(decoded,gt)
    det=total_cls+total_obj+5*total_box
    ll,lh,hl,hh=haar(aux['feature_p3'])
    smooth=((ll[:,:,:,1:]-ll[:,:,:,:-1]).abs().mean()+(ll[:,:,1:,:]-ll[:,:,:-1,:]).abs().mean())
    bg=background_mask(targets,(ll.shape[0],ll.shape[2],ll.shape[3]),ll.device)
    hf=(lh.abs()*bg).mean()+(hl.abs()*bg).mean()+(hh.abs()*bg).mean()
    wav=smooth+.5*hf
    metadata_mask=aux.get('has_metadata',aux['has_meta']).to(device=det.device,dtype=torch.bool)
    if metadata_mask.any():
        a=aux['policy_logits'].flatten(1)
        log_meta=F.log_softmax(a/aux['tau'].clamp_min(1e-4)[:,None],dim=1)
        log_img=F.log_softmax(a/aux['tau0'].clamp_min(1e-4),dim=1)
        meta_prob=log_meta.exp()
        prior_per_image=(meta_prob*(log_meta-log_img)).sum(-1)
        prior=prior_per_image[metadata_mask].mean()
    else:
        prior=det.new_zeros(())
    total=det+.1*wav+.01*prior
    return {'total':total,'det':det.detach(),'cls':total_cls.detach(),'obj':total_obj.detach(),'box':total_box.detach(),'wav':wav.detach(),'prior':prior.detach()}
