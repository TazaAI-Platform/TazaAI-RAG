"""Allow `python -m taza_rag.cli` without package install."""

from taza_rag.cli import app

if __name__ == "__main__":
    app()
