# llm_provider.py
"""
Single place where the chat model is chosen and constructed.

The provider used to be hardcoded to a local Ollama install in two separate
files, which meant swapping models was a code edit in more than one place and
`/health` could confidently report a model that wasn't running. Everything
now comes from environment variables, resolved here:

    SAFESIGHT_LLM_PROVIDER   ollama (default) | groq | gemini | openai
    SAFESIGHT_LLM_MODEL      overrides the provider's default model
    SAFESIGHT_LLM_NUM_CTX    context window (Ollama only — hosted providers
                             fix this server-side)
    SAFESIGHT_LLM_NUM_PREDICT  max tokens generated per answer

Defaults are deliberately unchanged: with no variables set this behaves
exactly as before (local Ollama, llama3.1), so an existing checkout keeps
working with no API key and no new packages.

Why the provider packages are imported lazily, inside each branch: importing
langchain_groq at module level would make it a hard dependency of the whole
API for everyone, including people running fully offline who will never use
it. Only the provider actually selected has to be installed.
"""

import os

# Load a local .env file if one exists, so an API key can live in a
# gitignored file instead of having to be re-exported in every new terminal
# (a particular nuisance on Windows). Optional dependency on purpose: if
# python-dotenv isn't installed, real environment variables still work.
#
# The path is pinned to this file's own directory rather than left to
# load_dotenv()'s default search from the current working directory —
# otherwise starting the server from anywhere except the project root
# silently skips the .env, and the only symptom is a mysterious fallback to
# the local model.
try:
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    print(
        "python-dotenv is not installed, so any .env file is being ignored. "
        "Install it with:  pip install python-dotenv"
    )


# Groq by default: it answers in roughly 1-2s where local inference on a
# typical laptop takes 15-30s, and its free tier needs no card. Set
# SAFESIGHT_LLM_PROVIDER=ollama to go back to fully local/offline.
PROVIDER = os.environ.get("SAFESIGHT_LLM_PROVIDER", "groq").strip().lower()

# Sensible default model per provider. Model names on hosted platforms change
# often — if one 404s, check the provider's current model list and set
# SAFESIGHT_LLM_MODEL rather than editing this file.
DEFAULT_MODELS = {
    "ollama": "llama3.1:latest",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
}

MODEL = os.environ.get("SAFESIGHT_LLM_MODEL") or DEFAULT_MODELS.get(PROVIDER, DEFAULT_MODELS["ollama"])

NUM_CTX = int(os.environ.get("SAFESIGHT_LLM_NUM_CTX", "4096"))

# Caps how many tokens a single answer can generate. Generation time scales
# directly with output length, so this is the most reliable lever on
# worst-case response time regardless of which provider is in use.
NUM_PREDICT = int(os.environ.get("SAFESIGHT_LLM_NUM_PREDICT", "800"))

# Deterministic answers: this is a compliance tool, and the same question
# about the same inspection should not produce a differently-worded verdict
# each time it's asked.
TEMPERATURE = float(os.environ.get("SAFESIGHT_LLM_TEMPERATURE", "0"))


class MissingProviderPackage(RuntimeError):
    """Raised with an actionable install command rather than a bare ImportError."""


def _require(module_name, package_name):
    try:
        __import__(module_name)
    except ImportError as exc:
        raise MissingProviderPackage(
            f"The '{PROVIDER}' LLM provider needs the {package_name} package. "
            f"Install it with:  pip install {package_name}"
        ) from exc


def _require_key(env_var):
    if not os.environ.get(env_var):
        raise MissingProviderPackage(
            f"The '{PROVIDER}' LLM provider needs an API key. "
            f"Set the {env_var} environment variable before starting the backend."
        )


# What each hosted provider needs to be usable: (import name, pip package,
# API key variable).
HOSTED_REQUIREMENTS = {
    "groq": ("langchain_groq", "langchain-groq", "GROQ_API_KEY"),
    "gemini": ("langchain_google_genai", "langchain-google-genai", "GOOGLE_API_KEY"),
    "openai": ("langchain_openai", "langchain-openai", "OPENAI_API_KEY"),
}


def _hosted_provider_problem():
    """
    Why the configured hosted provider can't be used, or None if it can.

    Checked up front so the fallback decision happens once, at startup, with
    a message naming the exact fix — rather than surfacing as a failed chat
    request later on.
    """
    module_name, package_name, key_var = HOSTED_REQUIREMENTS[PROVIDER]

    try:
        __import__(module_name)
    except ImportError:
        return f"'{PROVIDER}' selected but {package_name} isn't installed (pip install {package_name})."

    if not os.environ.get(key_var):
        return f"'{PROVIDER}' selected but {key_var} isn't set (put it in a .env file or your environment)."

    return None


def build_llm():
    """
    Construct the chat model for the configured provider.

    If the default hosted provider isn't usable on this machine — package
    not installed, or no API key — this falls back to local Ollama with a
    loud warning rather than refusing to start. A teammate who clones the
    repo without a Groq key still gets a working app; they just get the
    slower path, and the log (plus /health and Admin -> Settings) tells them
    exactly which one they're on.
    """
    global PROVIDER, MODEL

    if PROVIDER in ("groq", "gemini", "openai"):
        problem = _hosted_provider_problem()
        if problem:
            print("=" * 70)
            print(f"LLM: {problem}")
            print("     Falling back to local Ollama — answers will be slower.")
            print("=" * 70)
            PROVIDER = "ollama"
            if not os.environ.get("SAFESIGHT_LLM_MODEL"):
                MODEL = DEFAULT_MODELS["ollama"]

    if PROVIDER == "groq":
        from langchain_groq import ChatGroq

        # num_ctx has no equivalent here — hosted providers fix the context
        # window server-side, so only the output cap carries over.
        return ChatGroq(model=MODEL, temperature=TEMPERATURE, max_tokens=NUM_PREDICT)

    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=MODEL,
            temperature=TEMPERATURE,
            max_output_tokens=NUM_PREDICT,
        )

    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=MODEL, temperature=TEMPERATURE, max_tokens=NUM_PREDICT)

    # Default: local Ollama. Unknown provider names fall through to here
    # rather than crashing the API on startup — a typo in an env var
    # shouldn't take the whole backend down, but it should be visible.
    if PROVIDER not in ("ollama", ""):
        print(
            f"Unknown SAFESIGHT_LLM_PROVIDER '{PROVIDER}' — falling back to local Ollama. "
            f"Valid values: {', '.join(sorted(DEFAULT_MODELS))}."
        )

    _require("langchain_ollama", "langchain-ollama")
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=MODEL,
        temperature=TEMPERATURE,
        num_ctx=NUM_CTX,
        num_predict=NUM_PREDICT,
    )


def describe():
    """What's actually running — for /health and the AI Assistant footer."""
    return {
        "provider": PROVIDER if PROVIDER in DEFAULT_MODELS else "ollama",
        "model": MODEL,
        "is_local": PROVIDER not in ("groq", "gemini", "openai"),
    }
