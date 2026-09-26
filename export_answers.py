"""Write outputs/full_answers.json from the cached answers (partial is fine):  python3 export_answers.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from matura.export import main

main()
