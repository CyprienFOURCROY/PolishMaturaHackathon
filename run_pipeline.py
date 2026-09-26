"""Run the pipeline from the terminal:  python3 run_pipeline.py [--only 1,2.1,3]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from matura.run import main

main()
