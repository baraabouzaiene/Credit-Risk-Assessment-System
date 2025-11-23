# tests/conftest.py
from pathlib import Path
import sys

# Path to project root: .../Credit-Risk-Assessment-System
ROOT_DIR = Path(__file__).resolve().parent.parent

# Ensure that "src" exists where we expect it
SRC_DIR = ROOT_DIR / "src"
if not SRC_DIR.exists():
    raise RuntimeError(f"'src' folder not found at: {SRC_DIR}")

# Add project root to Python path so `import src...` works
sys.path.insert(0, str(ROOT_DIR))
