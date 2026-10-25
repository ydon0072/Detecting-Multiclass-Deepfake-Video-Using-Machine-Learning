"""Download exactly the five FF++ classes with the supplied official script."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
output = root / 'data' / 'FaceForensics++'
status_path = root / 'ffpp_download_status.json'
classes = ['original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
status = {'state': 'running', 'completed_classes': [], 'output': str(output)}

def save():
    status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')

try:
    for name in classes:
        status['current_class'] = name
        save()
        for attempt in range(1, 4):
            print(f'Downloading {name}, attempt {attempt}', flush=True)
            command = [sys.executable, '-u', str(root / 'download_ffpp_v4.py'), str(output),
                       '-d', name, '-c', 'c23', '-t', 'videos', '--server', 'EU2']
            result = subprocess.run(command, input='\n', text=True)
            if result.returncode == 0:
                break
            if attempt == 3:
                raise RuntimeError(f'Official downloader failed for {name}: exit {result.returncode}')
            time.sleep(10)
        relative = 'original_sequences/youtube' if name == 'original' else f'manipulated_sequences/{name}'
        count = len(list((output / relative / 'c23/videos').glob('*.mp4')))
        if count != 1000:
            raise RuntimeError(f'{name}: expected 1000 files, found {count}')
        status['completed_classes'].append(name)
        save()
    status['state'] = 'downloaded'
    status['note'] = '5000 video files downloaded; full decoding validation pending.'
except Exception as exc:
    status['state'] = 'failed'
    status['error'] = str(exc)
    raise
finally:
    save()
