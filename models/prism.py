"""PRISM: priority-guided scanning in the wavelet domain."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from models.vrwkv import VRWKVBlock

def haar(x):
    a=x[...,0::2,0::2]; b=x[...,0::2,1::2]
    c=x[...,1::2,0::2]; d=x[...,1::2,1::2]
    return ((a+b+c+d)/2,(a-b+c-d)/2,(a+b-c-d)/2,(a-b-c+d)/2)

def inverse_haar(ll,lh,hl,hh):
    a=(ll+lh+hl+hh)/2; b=(ll-lh+hl-hh)/2
    c=(ll+lh-hl-hh)/2; d=(ll-lh-hl+hh)/2
    return F.pixel_shuffle(torch.stack([a,b,c,d],dim=2).flatten(1,2),2)

def to_tokens(x):
    return x.flatten(2).transpose(1,2)

def from_tokens(x,h,w):
    return x.transpose(1,2).reshape(x.shape[0],x.shape[2],h,w)

class Metadata(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(3,16),nn.ReLU(),nn.Linear(16,32))
        self.default_embedding=nn.Parameter(torch.zeros(32))

    @staticmethod
    def _batch_tensor(value,batch,x,name,dtype=None):
        if value is None:
            return torch.full((batch,),float('nan'),device=x.device,dtype=dtype or x.dtype)
        tensor=torch.as_tensor(value,device=x.device,dtype=dtype or x.dtype).reshape(-1)
        if tensor.numel()==1:
            return tensor.expand(batch)
        if tensor.numel()!=batch:
            raise ValueError(f'{name} must be scalar or have one value per batch item')
        return tensor

    def forward(self,x,height=None,angle=None,has_metadata=None):
        batch=x.shape[0]
        h=self._batch_tensor(height,batch,x,'height')
        a=self._batch_tensor(angle,batch,x,'angle')
        available=torch.isfinite(h)&torch.isfinite(a)&(h>0)
        if has_metadata is not None:
            requested=self._batch_tensor(has_metadata,batch,x,'has_metadata',dtype=torch.bool)
            available=available&requested
        safe_h=torch.where(available,h,torch.ones_like(h))
        safe_a=torch.where(available,a,torch.zeros_like(a))
        radians=torch.deg2rad(safe_a)
        encoded=self.net(torch.stack([safe_h.log(),radians.sin(),radians.cos()],dim=-1))
        default=self.default_embedding.to(dtype=x.dtype)[None].expand(batch,-1)
        return torch.where(available[:,None],encoded,default),available

class SSAN(nn.Module):
    def __init__(self,c):
        super().__init__(); self.gamma=nn.Linear(44,c,bias=False); self.beta=nn.Linear(44,c,bias=False)
        self.alpha=nn.Parameter(torch.tensor(.01))
    def forward(self,x,z):
        spatial=x.ndim==4
        if spatial: b,c,h,w=x.shape; x=to_tokens(x)
        mean=x.mean(-1,keepdim=True); var=x.var(-1,keepdim=True,unbiased=False)
        x=(x-mean)*torch.rsqrt(var+1e-5)
        x=x*(1+self.alpha*torch.tanh(self.gamma(z))[:,None,:])+self.alpha*torch.tanh(self.beta(z))[:,None,:]
        return from_tokens(x,h,w) if spatial else x

class HFBlock(nn.Module):
    def __init__(self,c):
        super().__init__(); d=3*c
        self.branches=nn.ModuleList([nn.Conv2d(d,d,k,padding=(k//2)*dil,dilation=dil,groups=d) for k,dil in [(3,1),(5,1),(7,1),(3,2),(3,3)]])
        # Interleaving makes each group see the five branch responses of one channel.
        self.mix=nn.Conv2d(5*d,d,1,groups=d)
        self.energy=nn.Sequential(nn.Linear(9,64),nn.ReLU(),nn.Linear(64,2*d))
    def forward(self,u,e):
        branch=torch.stack([f(u) for f in self.branches],dim=2)
        b,c,n,h,w=branch.shape
        m=self.mix(branch.reshape(b,c*n,h,w))
        scale,bias=self.energy(e).chunk(2,dim=-1)
        modulation=m*torch.sigmoid(scale)[...,None,None]+bias[...,None,None]
        return u+modulation

class HFE(nn.Module):
    def __init__(self,c,repeats=3):
        super().__init__(); d=3*c
        self.edge=nn.Conv2d(9,9,3,padding=1); self.proj=nn.Conv2d(9,d,1)
        self.blocks=nn.ModuleList([HFBlock(c) for _ in range(repeats)])
        self.interact=nn.Sequential(nn.Conv2d(c,c,3,padding=1,groups=c),nn.Conv2d(c,c,1))
        self.out=nn.Conv2d(d,c,1); self.prior=nn.Conv2d(d,1,1)
    def forward(self,bands,e):
        hf=torch.cat(bands,dim=1); u=self.proj(hf*(1+torch.sigmoid(self.edge(hf))))
        for block in self.blocks: u=block(u,e)
        a,b,c=u.chunk(3,dim=1)
        v=torch.cat([a+self.interact(b+c),b+self.interact(a+c),c+self.interact(a+b)],dim=1)
        return self.out(v),torch.sigmoid(self.prior(v))

class LFE(nn.Module):
    def __init__(self,c,blocks=2,hidden_rate=3.25):
        super().__init__()
        self.proj=nn.Conv2d(3,c,1)
        self.blocks=nn.ModuleList([VRWKVBlock(c,blocks,i,SSAN,hidden_rate) for i in range(blocks)])

    def forward(self, l, z):
        _, _, h, w = l.shape
        x = to_tokens(self.proj(l))
        for block in self.blocks:
            if self.training and getattr(self, "use_checkpoint", False):
                x = checkpoint(lambda value: block(value,z,(h,w)),x)
            else:
                x = block(x,z,(h,w))
        return from_tokens(x, h, w)

class Policy(nn.Module):
    def __init__(self,c=64):
        super().__init__();self.norm=SSAN(2*c)
        self.logits=nn.Sequential(nn.Conv2d(2*c,c,3,padding=1),nn.ReLU(),nn.Conv2d(c,1,1))
        self.tau0=nn.Parameter(torch.tensor(1.0))
        self.delta_tau=nn.Parameter(torch.tensor(.5))
        self.temp_vector=nn.Parameter(torch.zeros(32))
    def forward(self,l,h,z,meta,has_meta):
        logits=self.logits(self.norm(torch.cat([l,h],dim=1),z))
        batch=logits.shape[0]
        mask=torch.as_tensor(has_meta,device=logits.device,dtype=torch.bool).reshape(-1)
        if mask.numel()==1:
            mask=mask.expand(batch)
        if mask.numel()!=batch:
            raise ValueError('has_meta must be scalar or have one value per batch item')
        adjustment=torch.sigmoid((meta*self.temp_vector).sum(-1))
        tau_meta=self.tau0+self.delta_tau*adjustment
        tau=torch.where(mask,tau_meta,self.tau0.expand_as(tau_meta))
        scaled=(logits/tau.clamp_min(1e-4)[:,None,None,None]).flatten(1)
        if self.training:
            probs=F.gumbel_softmax(scaled,tau=1.0,hard=False,dim=1)
        else:
            probs=F.softmax(scaled,dim=1)
        probs=probs.reshape_as(logits)
        return {'logits':logits,'tau':tau,'tau0':self.tau0,'probabilities':probs}

class PGF(nn.Module):
    def __init__(self,input_c,c=128):
        super().__init__();self.hp=nn.Conv2d(input_c,c,1);self.lp=nn.Conv2d(input_c,c,1)
        self.mix=nn.Sequential(nn.Conv2d(4*c,4*c,3,padding=1,groups=4*c),nn.Conv2d(4*c,c,1))
        self.gamma=nn.Linear(2,1)
        self.causal=nn.Conv1d(c,c,7,groups=c)
        self.norm=SSAN(2*c);self.out=nn.Conv2d(2*c,c,1)
    def causal_refine(self,sequence,gamma):
        causal=self.causal(F.pad(sequence.transpose(1,2),(6,0))).transpose(1,2)
        return sequence+gamma*causal
    def forward(self,l,h,prior,policy,z):
        hp=self.hp(h);lp=self.lp(l);b,c,hh,ww=hp.shape;t=hh*ww
        weight=torch.sigmoid(self.mix(torch.cat([policy.expand(-1,c,-1,-1),prior.expand(-1,c,-1,-1),hp,lp],dim=1)))
        mixed=weight*hp+(1-weight)*lp
        order=policy.flatten(1).argsort(dim=1,descending=True)
        rev=order.argsort(dim=1);idx=order.unsqueeze(-1).expand(-1,-1,c)
        sequence=to_tokens(mixed).gather(1,idx)
        p=policy.flatten(2).transpose(1,2).gather(1,order.unsqueeze(-1))
        s=prior.flatten(2).transpose(1,2).gather(1,order.unsqueeze(-1))
        gamma=torch.sigmoid(self.gamma(torch.cat([p,s],dim=-1)))
        sequence=self.causal_refine(sequence,gamma)
        restored=sequence.gather(1,rev.unsqueeze(-1).expand(-1,-1,c))
        fused=torch.cat([from_tokens(restored,hh,ww),lp],dim=1)
        return self.out(self.norm(fused,z))

class WSDStep(nn.Module):
    def __init__(self,input_c,c=128):
        super().__init__();self.hp=nn.Conv2d(input_c,c,1)
        self.phi1=nn.Conv2d(2*c,c,1);self.norm=SSAN(c)
        self.depth=nn.Conv2d(c,c,3,padding=1,groups=c)
        self.phi2=nn.Conv2d(c,3*c,1)
        self.beta=nn.Conv2d(input_c+2,1,1);self.out=nn.Conv2d(2*c,c,1)
    def forward(self,q,p,h,policy,prior,z_coarse):
        pd=F.avg_pool2d(p,2);hd=F.avg_pool2d(h,2)
        features=torch.cat([pd,F.avg_pool2d(self.hp(h),2)],dim=1)
        bands=self.phi2(self.depth(self.norm(self.phi1(features),z_coarse))).chunk(3,dim=1)
        gate=torch.sigmoid(self.beta(torch.cat([hd,F.avg_pool2d(policy,2),F.avg_pool2d(prior,2)],dim=1)))
        up=inverse_haar(q,*(gate*v for v in bands))
        return self.out(torch.cat([up,p],dim=1))

class HeadBranch(nn.Module):
    def __init__(self,c,out_channels,depth):
        super().__init__()
        self.convs=nn.ModuleList([nn.Conv2d(c,c,3,padding=1) for _ in range(depth)])
        self.norms=nn.ModuleList([SSAN(c) for _ in range(depth)])
        self.out=nn.Conv2d(c,out_channels,1)
    def forward(self,x,z):
        for conv,norm in zip(self.convs,self.norms):
            x=F.relu(norm(conv(x),z),inplace=True)
        return self.out(x)

class Head(nn.Module):
    def __init__(self,classes,c=128):
        super().__init__()
        self.cls=HeadBranch(c,classes,2)
        self.box=HeadBranch(c,4,2)
        self.obj=HeadBranch(c,1,1)
    def forward(self,x,z):return {'cls':self.cls(x,z),'box':self.box(x,z),'obj':self.obj(x,z)}

class PRISM(nn.Module):
    def __init__(self,num_classes=1,channels=(96,192,384,768),hfe_repeats=3,rwkv_blocks=2,rwkv_hidden_rate=3.25,fused_channels=128):
        super().__init__();self.num_classes=num_classes;self.meta=Metadata()
        self.hfe=nn.ModuleList([HFE(c,hfe_repeats) for c in channels]);self.lfe=nn.ModuleList([LFE(c,rwkv_blocks,rwkv_hidden_rate) for c in channels])
        self.policy=Policy(channels[0]);self.pgf=nn.ModuleList([PGF(c,fused_channels) for c in channels])
        self.decoder=nn.ModuleList([WSDStep(channels[k],fused_channels) for k in (2,1,0)])
        self.head=Head(num_classes,fused_channels)
    def forward(self,x,height=None,angle=None,has_metadata=None,return_aux=False):
        if x.ndim!=4 or x.shape[1]!=3 or min(x.shape[-2:])<16 or x.shape[-1]%16 or x.shape[-2]%16:
            raise ValueError('x must be RGB NCHW, dimensions divisible by 16')
        m,has_meta=self.meta(x,height,angle,has_metadata); lows=[]; bands=[]; zs=[]; energies=[]
        y=x
        for _ in range(4):
            ll,lh,hl,hh=haar(y); y=ll;lows.append(ll);bands.append((lh,hl,hh))
            en=torch.cat([v.abs().mean((2,3)) for v in (lh,hl,hh)],dim=1)
            energies.append(en);zs.append(torch.cat([ll.mean((2,3)),en,m],dim=1))
        hf=[];prior=[];ctx=[]
        for i in range(4):
            h,s=self.hfe[i](bands[i],energies[i]);hf.append(h);prior.append(s)
            ctx.append(self.lfe[i](lows[i],zs[i]))
        policy_result=self.policy(ctx[0],hf[0],zs[0],m,has_meta)
        p0=policy_result['probabilities']
        policies=[p0]+[F.avg_pool2d(p0,2**i) for i in range(1,4)]
        p=[self.pgf[i](ctx[i],hf[i],prior[i],policies[i],zs[i]) for i in range(4)]
        decoded={6:p[3]};q=p[3]
        for j,k in enumerate((5,4,3)):
            i=k-3;q=self.decoder[j](q,p[i],hf[i],policies[i],prior[i],zs[i+1]);decoded[k]=q
        outputs=[self.head(decoded[k],zs[k-3]) for k in (3,4,5,6)]
        if return_aux:return {'outputs':outputs,'feature_p3':decoded[3],
            'policy_logits':policy_result['logits'],'tau':policy_result['tau'],
            'tau0':policy_result['tau0'],'policy':policies,'priors':prior,
            'has_metadata':has_meta,'has_meta':has_meta}
        return outputs
