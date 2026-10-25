"""Local contract tests for the GPU runner. No real GPU or new dataset is used."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json,sys,types,numpy as np
from PIL import Image
import torch
from unittest.mock import patch
import pinned_encoder_helpers as h
import confirm_once as c

class BatchTensor:
 def __init__(self,t):self.tensor=t
 def to(self,*args,**kwargs):return self.tensor

class Processor:
 def __call__(self,images=None,text=None,**kwargs):
  if images is not None:
   rows=[np.asarray(im.resize((4,4))).transpose(2,0,1)/255. for im in images]
   return {'pixel_values':BatchTensor(torch.tensor(np.stack(rows),dtype=torch.float32))}
  return {'input_ids':BatchTensor(torch.tensor([[len(t),len(t)+1,len(t)+2] for t in text],dtype=torch.float32))}

class Model:
 def __init__(self):self.first=True
 def get_image_features(self,pixel_values):
  if self.first and len(pixel_values)>2:
   self.first=False;raise torch.cuda.OutOfMemoryError('Synthetic memory test')
  x=pixel_values.mean((2,3));return torch.cat([x,torch.ones((len(x),1))],dim=1)
 def get_text_features(self,input_ids):return torch.cat([input_ids,torch.ones((len(input_ids),1))],dim=1)

class FakeDataset:
 def __init__(self,name,root):
  self.classes=[f'class_{i}' for i in range({'CIFAR10':10,'Food101':101,'FGVCAircraft':100}[name])]
  self.root=Path(root);self.name=name;self._image_files=[self.root/name/f'{i}.jpg' for i in range(17)]
 def __len__(self):return 17
 def __getitem__(self,i):return Image.new('RGB',(8,40),(i*13%255,i*3%255,73)),i%len(self.classes)

def main(out):
 checks=[]
 with TemporaryDirectory() as tmp:
  tmp=Path(tmp);p=tmp/'x.npz';arr=np.arange(6).reshape(2,3)
  h.write_verified_npz(p,'identity',a=arr);assert np.array_equal(h.read_verified_npz(p,'identity')['a'],arr);checks.append('checkpoint roundtrip')
  try:h.read_verified_npz(p,'different')
  except RuntimeError:checks.append('identity mismatch rejected')
  else:raise AssertionError('Identity mismatch accepted')
  original=p.read_bytes();p.write_bytes(original+b'bad')
  try:h.read_verified_npz(p,'identity')
  except RuntimeError:checks.append('corrupted bytes rejected')
  else:raise AssertionError('Corruption accepted')
  p.write_bytes(original);h.receipt_path(p).unlink();assert h.read_verified_npz(p,'identity') is None;checks.append('uncommitted bytes quarantined')
  rows=[]
  for i in range(7):
   Image.new('RGB',(8,10),(20*i,33,150)).save(tmp/f'{i}.png');rows.append({'path':f'{i}.png'})
  events=[];m=Model();a,bs=h.encode_images(m,Processor(),rows,tmp,torch,4,events,'FAKE',0)
  assert a.shape==(7,4) and bs==2 and len(events)==1;h.normalized_features(a,'mock image');checks.append('batch retry retains every image')
  t=h.encode_text(m,Processor(),h.MODELS[0],['a','b'],torch);h.normalized_features(t,'mock text');checks.append('text encoding contract')
  logits=h.compute_logits(a,t,2.,-.1);assert logits.shape==(7,2) and np.isfinite(logits).all();checks.append('canonical logit reconstruction')
  def factory(name):return lambda root,**kw:FakeDataset(name,root)
  with patch('torchvision.datasets.CIFAR10',factory('CIFAR10')),patch('torchvision.datasets.Food101',factory('Food101')),patch('torchvision.datasets.FGVCAircraft',factory('FGVCAircraft')):
   study=tmp/'study';study.mkdir();first=c.prepare_datasets(study);second=c.prepare_datasets(study)
   assert first==second and set(first)==set(c.PROTOCOL['datasets']);checks.append('three dataset schemas, deterministic image selection and restart')
   assert first['FGVCAircraft']['rows'][0]['rgb_size']==[8,20];checks.append('aircraft footer cropped by exactly 20 pixels')
  # A transcript schema or corrupted selection must not be accepted.
  row=first['Food101']['rows'][0];bad=study/'data'/row['path'];Image.new('RGB',(1,1)).save(bad)
  with patch('torchvision.datasets.CIFAR10',factory('CIFAR10')),patch('torchvision.datasets.Food101',factory('Food101')),patch('torchvision.datasets.FGVCAircraft',factory('FGVCAircraft')):
   try:c.prepare_datasets(study)
   except RuntimeError:checks.append('changed selected image rejected')
   else:raise AssertionError('Changed selection accepted')
 result={'passed':True,'checks':checks,'count':len(checks),'scope':'mock image/model contracts using CPU Torch; no CUDA, pretrained model inference, network downloads or scientific new-condition execution'}
 Path(out).write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main(sys.argv[1])
