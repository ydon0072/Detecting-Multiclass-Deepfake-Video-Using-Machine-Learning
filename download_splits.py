"""Download public split metadata only, not the FF++ videos."""
import argparse
from pathlib import Path
from urllib.request import urlopen
from dataset.ffpp import read_splits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/splits')
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ['train', 'val', 'test']:
        path = folder / f'{name}.json'
        if not path.exists():
            url = f'https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{name}.json'
            with urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
    splits = read_splits(folder)
    print({name: len(pairs) * 2 for name, pairs in splits.items()})
