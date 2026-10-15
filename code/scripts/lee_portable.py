"""Package-root bridge to the independent Lee new-run namespace."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code' / 'training' / 'portable'))

from dopnet_lee_research_candidate.__main__ import main
from dopnet_lee_research_candidate.policy import PolicyError, PolicyStopped


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except PolicyStopped as error:
        print('NEW_RUN_STOPPED: ' + str(error), file=sys.stderr)
        raise SystemExit(75)
    except PolicyError as error:
        print('NEW_RUN_POLICY_REJECTED: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
