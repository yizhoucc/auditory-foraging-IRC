from pathlib import Path
import yaml

with open(Path(__file__).parent/'VERSION.txt', 'r') as f:
    __version__ = f.readline().split('"')[1]
with open(Path(__file__).parent/'rcParams.yaml', 'r') as f:
    rcParams: dict = yaml.safe_load(f)
