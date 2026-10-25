"""Paired ViT architecture screening on fixed official train/validation subsets."""
import ctypes,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/vit_structure_comparison'
def write(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)
def main():
 OUT.mkdir(exist_ok=True)
 if (OUT/'status.json').exists(): raise RuntimeError('Run exists; inspect before restarting')
 state={'state':'running','pid':os.getpid(),'started':time.time(),'stage':'preparing'}
 def update(**kw):
  state.update(kw,updated=time.time());write(OUT/'status.json',state)
 ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
 try:
  update()
  from dataset.ffpp import read_splits,FFPPDataset
  source=ROOT/'runs/branch1_experiment/tuning_splits'
  subset=read_splits(source);official=read_splits(ROOT/'data/splits')
  for split in ['train','val']:
   assert all(pair in official[split] for pair in subset[split])
  target=OUT/'splits';target.mkdir()
  for split in ['train','val','test']:shutil.copy2(source/f'{split}.json',target/f'{split}.json')
  for split,count in [('train',200),('val',100)]:
   ds=FFPPDataset(ROOT/'data/FaceForensics++',subset[split]);assert len(ds)==count
  trials=[]
  for variant in ['vit_b_16','vit_b_32']:
   folder=OUT/variant;folder.mkdir()
   cmd=[sys.executable,'-u',str(ROOT/'vit_structure_train.py'),'--variant',variant,
    '--data-root',str(ROOT/'data/FaceForensics++'),'--splits-dir',str(target),'--output',str(folder),
    '--epochs','6','--frames','16','--batch-size','1','--accumulation-steps','8','--workers','2',
    '--warmup-epochs','1','--train-last-blocks','4','--backbone-lr','0.00001','--lr','0.0001',
    '--weight-decay','0.0001','--seed','42','--gpu-duty','0.6','--gpu-memory-fraction','0.6']
   write(folder/'command.json',cmd);started=time.time()
   with (folder/'console.log').open('w',encoding='utf-8') as log:
    proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    update(stage='training',variant=variant,child_pid=proc.pid,log=str(folder/'console.log'))
    if proc.wait():raise RuntimeError(f'{variant} failed; inspect console.log')
   history=json.loads((folder/'history.json').read_text());best=max(history,key=lambda e:e['val']['macro_f1'])
   trials.append({'variant':variant,'best_epoch':best['epoch'],'validation':best['val'],
    'elapsed_seconds':time.time()-started,'checkpoint':str(folder/'best.pt')})
   write(OUT/'comparison.json',trials)
  ranked=sorted(trials,key=lambda x:x['validation']['macro_f1'],reverse=True)
  report=['# ViT architecture screening','', 'Fixed subset: 200 train / 100 validation videos, 16 frames, seed 42. No test evaluation.',
   'Both use ImageNet-1K V1 initialisation, 1 warmup epoch, 5 epochs tuning the last 4 blocks, and identical optimisation settings.',
   'This is preliminary screening; a single small-subset run cannot establish a definitive architecture ranking.','',
   '| Variant | Best epoch | Validation accuracy | Validation macro F1 |','|---|---:|---:|---:|']
  for x in ranked:report.append(f"| {x['variant']} | {x['best_epoch']} | {x['validation']['accuracy']:.2%} | {x['validation']['macro_f1']:.2%} |")
  (OUT/'comparison.md').write_text('\n'.join(report),encoding='utf-8')
  update(state='complete',stage='complete',selected=ranked[0]['variant'],results=str(OUT/'comparison.json'))
 except Exception as exc:
  update(state='failed',error=str(exc));traceback.print_exc();raise
 finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':main()
