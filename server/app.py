"""FastAPI application for the Engineer Manager OpenEnv server."""

from __future__ import annotations

import uvicorn
from openenv.core import create_app

try:
    from ..models import EngineerManagerAction, EngineerManagerObservation
    from .engineer_manager_environment import EngineerManagerEnvironment
except ImportError:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from models import EngineerManagerAction, EngineerManagerObservation
    from server.engineer_manager_environment import EngineerManagerEnvironment


app = create_app(
    EngineerManagerEnvironment,
    EngineerManagerAction,
    EngineerManagerObservation,
    env_name="engineer-manager",
    max_concurrent_envs=2,
)


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the OpenEnv HTTP server."""
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """CLI entrypoint expected by the OpenEnv validator."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
