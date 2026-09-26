"""Start the interactive viewer:  python3 serve.py [--port 8765]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from matura.viewer.server import main

main()
