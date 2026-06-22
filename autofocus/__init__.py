from pathlib import Path
import pkgutil

__version__ = '0.1.0'

# Allow imports from src/ layout without installation.
__path__ = pkgutil.extend_path(__path__, __name__)
_src_pkg = Path(__file__).resolve().parent.parent / 'src' / 'autofocus'
if _src_pkg.is_dir():
    __path__.append(str(_src_pkg))
