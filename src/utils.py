import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env relative to project root
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_BASE = os.getenv("AZURE_API_BASE")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini")

DEBUG = True
DEBUG_LIMIT = 10