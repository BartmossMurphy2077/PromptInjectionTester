import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env relative to project root
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Stores relevant information from .env into usable variables
#AZURE_DEPLOYMENT_NAME defaults to "gpt-4o-mini" if not set
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_BASE = os.getenv("AZURE_API_BASE")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini")

#Model randomness
TESTER_TEMPERATURE = 1
AUDITOR_TEMPERATURE = 1
AUDITOR_CHECKS_PROMPT_AND_RESPONSE = True

#Concurrency limit
CONCURRENCY_LIMIT = 20

#Debug settings
#If True you will see token usage per request
#If RUN_LIMIT > 0 the dataset will be limited to that many entries for faster testing
DEBUG = True
RUN_LIMIT = 300