from pathlib import Path
import sys

root = Path(__file__).resolve().parent
src = root / 'src'
if src.is_dir():
    sys.path.insert(0, str(src))
