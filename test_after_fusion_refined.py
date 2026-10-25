"""One-shot test continuation explicitly requested after current fusion training."""
import ctypes,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/fusion_refined_vit'
def main():
 status_path=RUN/'test_continuation_status.json'
 # Exclusive claim prevents duplicate evaluators.
 with status_path.open('x',encoding='utf-8') as f:json.dump({'state':'starting','pid':os.getpid()},f)
 state={'state':'waiting_for_training','pid':os.getpid(),'started':time.time()}
 def save(**values):
  state.update(values,updated=time.time());tmp=status_path.with_suffix('.tmp')
  tmp.write_text(json.dumps(state,indent=2),encoding='utf-8');os.replace(tmp,status_path)
 ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
 try:
  save();deadline=time.monotonic()+36*3600
  while True:
   training=json.loads((RUN/'status.json').read_text())
   if training['state']=='complete':break
   if training['state']=='failed':raise RuntimeError('Training failed; test not started')
   if time.monotonic()>deadline:raise TimeoutError('Training wait exceeded 36 hours; test not started')
   time.sleep(15)
  out=RUN/'test'
  if (out/'test_metrics.json').exists():raise FileExistsError('Test results already exist; not overwriting')
  cmd=[sys.executable,'-u',str(ROOT/'fusion_test.py'),'--data-root',str(ROOT/'data/FaceForensics++'),
   '--splits-dir',str(ROOT/'data/splits'),'--checkpoint',str(RUN/'best.pt'),'--output',str(out),
   '--batch-size','2','--workers','4','--device','cuda','--amp','--gpu-duty','0.6','--gpu-memory-fraction','0.6']
  (RUN/'test_command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
  with (RUN/'test_console.log').open('w',encoding='utf-8') as log:
   proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   save(state='testing',child_pid=proc.pid,best_epoch=training['best_epoch'])
   if proc.wait():raise RuntimeError('Test failed; inspect test_console.log')
  m=json.loads((out/'test_metrics.json').read_text())
  assert m['samples']==700 and not m['incomplete_test_set']
  save(state='complete',accuracy=m['accuracy'],macro_f1=m['macro_f1'],samples=m['samples'],results=str(out/'test_metrics.json'))
 except Exception as exc:
  save(state='failed',error=str(exc));traceback.print_exc();raise
 finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':main()
