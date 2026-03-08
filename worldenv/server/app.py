"""FastAPI application for the WorldEnv environment server."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(f"openenv is required: {e}") from e

from models import WorldAction, WorldObservation
from .world_environment import WorldEnvironment

app = create_app(
    WorldEnvironment,
    WorldAction,
    WorldObservation,
    env_name="worldenv",
    max_concurrent_envs=8,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
