import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env relative to project root
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Stores relevant information from .env into usable variables
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_BASE = os.getenv("AZURE_API_BASE")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini")

# Set the Azure endpoint for pydantic-ai
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or AZURE_API_BASE

PROMPT_CANARY = os.getenv("PROMPT_CANARY", "CANARY_PROMPT_a1b2c3")
DATA_CANARY = os.getenv("DATA_CANARY", "CANARY_DATA_x9y8z7")

# Set environment variable for pydantic-ai Azure provider
if AZURE_OPENAI_ENDPOINT:
    os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
if AZURE_API_KEY:
    os.environ["AZURE_OPENAI_API_KEY"] = AZURE_API_KEY
