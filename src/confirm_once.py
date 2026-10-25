"""One resumable invocation: new data -> pinned GPU encoders -> full statistical audit.
The script never reports confirmation results until every scheduled stage completes.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,os,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np
from pinned_encoder_helpers import (MODELS,configure_torch,prepare_model_assets,import_models,
 encode_images,encode_text,compute_logits,atomic_json,sha256_file,canonical_hash,write_verified_npz,read_verified_npz,normalized_features,retry_download)

PROTOCOL={
 'version':'Calibration_Count_New_Conditions_v4_20260919',
 'purpose':'Confirmation on three datasets not used to design this revision; no claim of new field data',
 'datasets':['CIFAR10','Food101','FGVCAircraft'],
 'split':'official test pool for each dataset; at most 6000 images selected by a label-independent SHA256 ranking',
 'models':MODELS,'prompt':'a photo of a {}.',
 'aircraft_preprocessing':'remove bottom 20-pixel attribution banner before model preprocessing, as requested by the dataset authors',
 'temperature_labels':256,'temperatures':['native','independently fitted NLL, tau [.25,8]'],
 'prior_pseudocount':1,'ratios':[2,6,20],'alphas':[.1,.05],
 'one_cycle_bags':2048,'three_cycle_bags':512,'complete_set_episodes':128,'queries_per_episode':16,
 'main_unit':'one independent bag or calibration-plus-query episode conditional on the fixed extracted dataset',
 'estimand':'marginal coverage and set size under iid sampling with replacement from the held-back finite population',
 'newness':'no outputs on these three datasets were accessed during v4 development',
 'primary':'one cycle coverage, full conformal coverage, and paired role erosion, with every model/dataset/temperature/ratio retained',
 'secondary':'three cycle same endpoints; complete sets from 13 prespecified methods and exact screening',
 'confidence':'union allocated empirical Bernstein bounds on complete bag/episode means; no individual-role binomial interval',
 'negative_results':'no selection of successful datasets, temperatures, seeds, or class subsets; errors produce INCOMPLETE',
 'sampling_independence':'new random episodes, not independent new environments; overlap of calibration/query identities retained',
 'extra_baselines':'full conformal ridge NLL and randomized RAPS also run on every saved complete-set episode',
 'no_acceptance_probability':'completion of this protocol is not a conference acceptance forecast'
}

def prepare_datasets(root):
 from torchvision.datasets import CIFAR10,Food101,FGVCAircraft
 from PIL import Image
 data=root/'data';data.mkdir(exist_ok=True)
 datasets={};objects=[('CIFAR10',lambda:CIFAR10(str(data),train=False,download=True)),
 ('Food101',lambda:Food101(str(data),split='test',download=True)),
 ('FGVCAircraft',lambda:FGVCAircraft(str(data),split='test',annotation_level='variant',download=True))]
 for name,load in objects:
  print('Preparing',name,flush=True);ds=retry_download(load);classes=[str(x).replace('_',' ') for x in ds.classes]
  expected={'CIFAR10':10,'Food101':101,'FGVCAircraft':100}[name]
  if len(classes)!=expected:raise ValueError('Class mapping mismatch for '+name)
  def identity(i):
   if name=='CIFAR10':return f'test:{i:06d}'
   files=getattr(ds,'_image_files',None)
   if files is None:raise ValueError('Pinned torchvision dataset schema changed: _image_files absent')
   return str(Path(files[i]).relative_to(data))
  ids=[identity(i) for i in range(len(ds))]
  chosen=sorted(range(len(ds)),key=lambda i:hashlib.sha256((PROTOCOL['version']+':'+name+':'+ids[i]).encode()).digest())[:6000]
  rows=[];outdir=data/'selected'/name;outdir.mkdir(parents=True,exist_ok=True)
  for t,i in enumerate(chosen):
   im,label=ds[i];im=im.convert('RGB')
   if name=='FGVCAircraft':
    if im.height<=20:raise ValueError('Aircraft image too short for the declared footer crop')
    original=im;im=original.crop((0,0,original.width,original.height-20));original.close()
   p=outdir/f'{i:08d}.png'
   # Lossless selected-byte snapshot simplifies audit and repeatable encoder input.
   if not p.exists():im.save(p,format='PNG')
   with Image.open(p) as saved:
    if saved.size!=im.size or not np.array_equal(np.asarray(saved.convert('RGB')),np.asarray(im)):
     raise RuntimeError('Selected image bytes changed')
   rows.append({'id':ids[i],'label':int(label),'role':'audit_pool','path':str(p.relative_to(data)),
                'sha256':sha256_file(p),'rgb_size':list(im.size)})
   im.close()
  datasets[name]={'id':name,'class_names':classes,'rows':rows,'original_test_size':len(ds)}
  print('Locked',name,len(rows),'images',len(classes),'labels',flush=True)
 atomic_json(root/'dataset_manifest.json',datasets)
 return datasets

def extract(root):
 root.mkdir(parents=True,exist_ok=True)
 protocol_sha=canonical_hash(PROTOCOL)
 protofile=root/'PROTOCOL.json'
 if protofile.exists() and canonical_hash(json.loads(protofile.read_text()))!=protocol_sha:
  raise RuntimeError('Existing protocol differs; do not overwrite a prior study')
 atomic_json(protofile,PROTOCOL)
 # Bind code before downloading data or computing any model output.
 sources={p.name:sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
 sourcefile=root/'SOURCE_FREEZE.json'
 if sourcefile.exists() and json.loads(sourcefile.read_text())!=sources:
  raise RuntimeError('Code changed after freeze; preserve the prior run and review changes')
 atomic_json(sourcefile,sources)
 torch,env=configure_torch();atomic_json(root/'ENVIRONMENT.json',env)
 a=torch.randn(32,32,device='cuda');assert torch.isfinite(a@a).all().item()
 datasets=prepare_datasets(root);assets=prepare_model_assets(root/'model_cache');atomic_json(root/'MODEL_ASSETS.json',assets)
 AP,CLIP,SIG=import_models();checks=[]
 for spec in MODELS:
  asset=assets[spec['id']];slug=spec['id'].replace('/','__')
  processor=AP.from_pretrained(asset['local_path'],local_files_only=True,use_fast=False,trust_remote_code=False)
  cls=SIG if spec['kind']=='siglip' else CLIP
  model=cls.from_pretrained(asset['local_path'],local_files_only=True,torch_dtype=torch.float32,
    attn_implementation='eager',use_safetensors=spec['safetensors'],trust_remote_code=False).eval().to('cuda:0')
  scale=float(model.logit_scale.detach().float().exp().cpu().item())
  bias=float(getattr(model,'logit_bias',torch.tensor(0.)).detach().float().cpu().item())
  for name,ds in datasets.items():
   identity=canonical_hash({'protocol':protocol_sha,'sources':sources,'dataset':ds,'spec':spec,'environment':env})
   path=root/'cache'/f'{slug}__{name}.npz';old=read_verified_npz(path,identity)
   if old is not None:
    normalized_features(old['image_features'],'resumed images');normalized_features(old['text_features'],'resumed text')
    if not np.array_equal(old['labels'],[r['label'] for r in ds['rows']]):raise RuntimeError('Cache label alignment mismatch')
    if not np.array_equal(compute_logits(old['image_features'],old['text_features'],scale,bias),old['logits']):raise RuntimeError('Cache logits mismatch')
    print('Verified completed cache',path.name,flush=True);checks.append(path.name);continue
   text=encode_text(model,processor,spec,ds['class_names'],torch);features=[];events=[];bs=64
   for begin in range(0,len(ds['rows']),256):
    part=root/'checkpoints'/f'{slug}__{name}__{begin:06d}.npz';done=read_verified_npz(part,identity)
    if done is None:
     feat,bs=encode_images(model,processor,ds['rows'][begin:begin+256],root/'data',torch,bs,events,name,begin)
     write_verified_npz(part,identity,image_features=feat,image_ids=np.array([v['id'] for v in ds['rows'][begin:begin+256]]),batch_size=np.array(bs),batch_events=np.array(json.dumps(events)))
    else:
     feat=done['image_features'];normalized_features(feat,'resumed chunk')
     expected=[v['id'] for v in ds['rows'][begin:begin+256]]
     if not np.array_equal(done['image_ids'],expected):raise RuntimeError('Checkpoint row identity mismatch')
     bs=min(bs,int(done['batch_size']))
     for ev in json.loads(str(done['batch_events'])):
      if ev not in events:events.append(ev)
    if feat.shape[0]!=len(ds['rows'][begin:begin+256]):raise RuntimeError('Checkpoint image count mismatch')
    features.append(feat);print(path.name,min(begin+256,len(ds['rows'])),'/',len(ds['rows']),flush=True)
   feat=np.concatenate(features);logits=compute_logits(feat,text,scale,bias)
   meta={'identity_sha256':identity,'spec':spec,'dataset':name,'evidence_origin':'new_conditions_cuda_extraction',
         'environment':env,'batch_events':events,'prompt':PROTOCOL['prompt']}
   write_verified_npz(path,identity,image_features=feat,text_features=text,logits=logits,
      image_ids=np.array([r['id'] for r in ds['rows']]),labels=np.array([r['label'] for r in ds['rows']]),
      roles=np.array(['audit_pool']*len(ds['rows'])),scale=np.array(scale),bias=np.array(bias),metadata=np.array(json.dumps(meta)))
   checks.append(path.name)
  del model,processor;gc.collect();torch.cuda.empty_cache()
 if len(checks)!=9:raise RuntimeError('Missing scheduled model/dataset cache')
 atomic_json(root/'EXTRACTION_COMPLETE.json',{'caches':checks,'count':9})

def invoke(script,args,log):
 with log.open('a') as f:
  p=subprocess.Popen([sys.executable,str(Path(__file__).parent/script),*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  for line in p.stdout:
   print(line,end='',flush=True);f.write(line);f.flush()
  if p.wait()!=0:raise RuntimeError(f'{script} failed. Full log: {log}')

def main(root):
 root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);start=time.time()
 status={'status':'RUNNING','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};atomic_json(root/'STATUS.json',status)
 try:
  extract(root)
  for script in ['transport_audit.py','confirmation_audit.py','confirmation_extra.py','confirmation_summary.py','score_diagnostics.py']:
   target=root/'code'/script;target.parent.mkdir(exist_ok=True);shutil.copy2(Path(__file__).parent/script,target)
  natural=root/'results/natural'
  if not (natural/'SUMMARY.json').exists():invoke('confirmation_audit.py',['--cache',str(root/'cache'),'--out',str(natural)],root/'analysis.log')
  extra=root/'results/extra_baselines'
  if not (extra/'SUMMARY.json').exists():invoke('confirmation_extra.py',['--root',str(root),'--out',str(extra)],root/'extra.log')
  invoke('confirmation_summary.py',['--root',str(root)],root/'summary.log')
  invoke('score_diagnostics.py',['--root',str(root)],root/'score_diagnostics.log')
  for p in [natural/'SUMMARY.json',extra/'SUMMARY.json']:
   if not json.loads(p.read_text()).get('complete'):raise RuntimeError('Analysis did not complete')
  status.update(status='COMPLETE',seconds=time.time()-start,notes='Scientific outcomes are in results; completion is not a success-rate or acceptance claim')
 except BaseException as e:
  status.update(status='INCOMPLETE',error=str(e),traceback=traceback.format_exc());atomic_json(root/'STATUS.json',status);raise
 atomic_json(root/'STATUS.json',status)
 bundle=root.parent/'Calibration_Count_NEW_CONDITIONS_RESULTS.zip'
 with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
  for p in root.rglob('*'):
   if p.is_file() and not set(p.relative_to(root).parts)&{'data','model_cache','checkpoints'}:
    z.write(p,p.relative_to(root))
 print('\nCOMPLETE. Return this file:',bundle,'\n',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='calibration_count_confirmation');a=p.parse_args();main(a.root)
