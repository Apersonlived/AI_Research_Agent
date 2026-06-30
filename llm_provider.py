"""
llm_provider.py
-------------------------------------
Priority:
  1. Google Gemini 1.5 Flash  (1,500 req/day free, no credit card, cloud)
  2. Ollama / Llama 3.2       (fully local model)

Ollama setup:
  1. Download and install: https://ollama.com/download/windows
  2. Open a new PowerShell and run: ollama pull llama3.2
  3. Ollama runs in the background automatically after install.
  No API key needed. Works fully offline once the model is downloaded.

Usage:
    from llm_provider import get_llm, get_available_providers

    llm = get_llm()                     # auto-selects best available model
    llm = get_llm(provider="gemini")    # force Gemini (cloud, free tier)
    llm = get_llm(provider="ollama")    # force Ollama (local)
"""

import os
from dotenv import load_dotenv
load_dotenv()

from enum import Enum


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm(provider: str = "auto", max_tokens: int = 2000):
    """
    Return a LangChain compatible chat model.
    provider='auto' tries Gemini first, falls back to Ollama.
    """
    if provider == "auto":
        if os.getenv("GOOGLE_API_KEY") and _gemini_likely_works():
            provider = "gemini"
        elif _ollama_is_running():
            provider = "ollama"
        elif os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        else:
            raise EnvironmentError(
                "No LLM available.\n"
                "  Option 1 (cloud, free): set GOOGLE_API_KEY from aistudio.google.com\n"
                "  Option 2 (local, free): install Ollama from ollama.com/download/windows\n"
                "                          then run: ollama pull llama3.2"
            )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Run: pip install langchain-google-genai")
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not set. Get a free key at aistudio.google.com"
            )
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=key,
            max_output_tokens=max_tokens,
            temperature=0.3,
        )

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("Run: pip install langchain-ollama")
        if not _ollama_is_running():
            raise EnvironmentError(
                "Ollama is not running.\n"
                "  1. Download and install: https://ollama.com/download/windows\n"
                f"  2. Open a new PowerShell and run: ollama pull {OLLAMA_MODEL}\n"
                "  3. Ollama starts automatically after install."
            )
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            num_predict=max_tokens,
            temperature=0.3,
        )

    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            "Valid options: auto | gemini | ollama"
        )


def _ollama_is_running() -> bool:
    """Check if Ollama server is reachable at the configured base URL."""
    import urllib.request
    try:
        urllib.request.urlopen(OLLAMA_BASE_URL, timeout=2)
        return True
    except Exception:
        return False


def _gemini_likely_works() -> bool:
    """
    This does NOT make an API call — just checks if the key is set.
    """
    return bool(os.getenv("GOOGLE_API_KEY"))


def get_available_providers() -> list[str]:
    """Return human-readable list of providers currently available."""
    available = []
    if os.getenv("GOOGLE_API_KEY"):
        available.append("gemini")
    if _ollama_is_running():
        available.append(f"ollama/{OLLAMA_MODEL}")
    return available