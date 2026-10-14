import torch
from models.prism import PRISM,haar,inverse_haar
from models.vrwkv import BiWKV
from utils.loss import prism_loss
from utils.postprocess import postprocess

def run():
    torch.manual_seed(17); torch.set_num_threads(2)
    x=torch.rand(1,3,32,32)
    bands=haar(x)
    assert torch.allclose(inverse_haar(*bands),x,atol=1e-6),'Haar inverse failed'
    wkv=BiWKV(4,chunk_size=4);k=torch.randn(1,9,4);v=torch.randn(1,9,4)
    out=wkv(k,v);assert out.shape==(1,9,4)
    model=PRISM(2,channels=(8,16,32,64),hfe_repeats=1,rwkv_blocks=1)
    model.train();aux=model(x,height=60.,angle=35.,return_aux=True)
    assert [o['cls'].shape[-2:] for o in aux['outputs']]==[(16,16),(8,8),(4,4),(2,2)]
    targets=[{'boxes':torch.tensor([[.25,.25,.6,.6]]),'labels':torch.tensor([1])}]
    loss=prism_loss(aux,targets,(32,32))['total'];assert torch.isfinite(loss)
    loss.backward();assert model.pgf[0].mix[0].weight.grad is not None
    model.eval()
    with torch.no_grad():dets=postprocess(model(x),confidence=.99)
    assert len(dets)==1
    print('PASS: Haar roundtrip, Bi-WKV, four scales, metadata, loss, backward, inference')
if __name__=='__main__':run()
