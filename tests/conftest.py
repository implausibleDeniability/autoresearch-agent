import sys
from pathlib import Path

TRAJECTORY_SCRIPTS = (
    Path(__file__).parents[1] / ".agents" / "skills" / "generate-autoresearch-trajectory" / "scripts"
)
sys.path.insert(0, str(TRAJECTORY_SCRIPTS))
