"""Grade answers against the official marking scheme:  python3 evaluate.py [--dry-run] [--only 1,2.1]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from matura.eval.judge import main

main()
