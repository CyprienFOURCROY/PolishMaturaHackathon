"""Single place to choose which model fills each role. Weights live in models/<role>/<name>/."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / "cache"

EXAM_PATH = ROOT / "full_exam_question.json"
IMAGES_DIR = ROOT / "ImagesMatura"  # JSON says images/Z01.png; we resolve by basename
GLOSSARY_PATH = ROOT / "data" / "glossary" / "glossary.json"
KB_DIR = ROOT / "data" / "kb"
PROMPTS_DIR = ROOT / "prompts"
OUT_PATH = ROOT / "outputs" / "full_answers.json"

# role -> which implementation ("impl" is a key of models.registry.IMPLS) and its weights
ROLES = {
    "ocr": {"impl": "easyocr", "name": "easyocr", "langs": ["pl", "en"]},
    "vlm": {"impl": "smolvlm", "name": "smolvlm2-500m", "hf": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "dtype": "bfloat16"},
    "translator": {"impl": "marian", "name": "opus-mt-pl-en", "hf": "Helsinki-NLP/opus-mt-pl-en"},
    "translator_back": {"impl": "marian", "name": "opus-mt-en-zlw", "hf": "Helsinki-NLP/opus-mt-en-zlw", "prefix": ">>pol<< "},
    "llm": {"impl": "gemma", "name": "gemma-3-270m-it", "hf": "google/gemma-3-270m-it", "dtype": "float32"},
}

GLOSSARY_MAX_TERMS = 8   # cap on definitions appended to the English text
KB_TOP_K = 3             # fact cards given to the LLM
