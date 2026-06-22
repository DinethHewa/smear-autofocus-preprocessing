from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

src = root / 'src'
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
