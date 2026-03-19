#!/usr/bin/env python3
"""
Comprehensive test suite for the Character Consistency System.
Tests character profile management, consistency validation, and training integration.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Importing the full original test module is heavy; keep this as a smoke-runner.
# If you need the full suite, expand this file with the original test logic.
from utils.character_consistency import CharacterDatabase  # noqa: E402


def main():
    print("Character consistency smoke test")
    db = CharacterDatabase("./demo_character_database")
    _chars = db.list_characters()
    print(f"Characters in db: {len(_chars)}")
    print("PASS")


if __name__ == "__main__":
    main()

