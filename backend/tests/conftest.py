# Test configuration
import sys
from pathlib import Path

# Add parent directory to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))
