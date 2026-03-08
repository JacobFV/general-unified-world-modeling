"""FastAPI app for the Robot World Environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(f"openenv required: {e}") from e

from models import RobotWorldAction, RobotWorldObservation
from .robot_world_env_environment import RobotWorldEnvironment

app = create_app(RobotWorldEnvironment, RobotWorldAction, RobotWorldObservation,
                 env_name="robot_world_env", max_concurrent_envs=4)

def main(host="0.0.0.0", port=8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
