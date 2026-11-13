"""Roll collections back to the embedding model they used before a switch.

The same freeze window as the switch: freeze, wait for admitted writers, embed
what the old model still lacks (documents ingested after the switch), prune,
accept, flip the binding back, unfreeze. The old model must still be configured
under the id it is recorded with.

    python -m xagent.migrations.embedding.rollback --model <old-model-id> \
        [--collection NAME ... | --team ID] [--concurrency N]
"""

from __future__ import annotations

import sys
from typing import List, Optional

from xagent.migrations.embedding.switch import MODE_ROLLBACK, run_cli


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(
        MODE_ROLLBACK,
        argv,
        description=(
            "Roll collections back to a previous embedding model.\n\n"
            "Runs the same freeze window as the switch, embedding with the old\n"
            "model everything ingested since. The old model must still be\n"
            "configured under the id it is recorded with."
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
