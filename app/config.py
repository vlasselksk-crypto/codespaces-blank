import os


def get_api_key() -> str:
    # Production should use a secure secret store; env var is a simple default for demos.
    return os.environ.get("SLOTHAC_API_KEY", "test-key")
