"""Quick run script for RFP Analyzer."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def main():
    """Main entry point."""
    from src.main import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()

