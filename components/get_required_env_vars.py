import os
import sys
from dotenv import load_dotenv

load_dotenv()

def get_required_env_var(name, default=None):
    """Get an environment variable or exit if not available and no default provided.

    Args:
        name: The name of the environment variable
        default: Optional default value to use if the variable is not set

    Returns:
        The value of the environment variable, or the default if provided

    Raises:
        SystemExit: If the variable is not set and no default is provided
    """
    print(f"Searching for {name} in the environment variables")
    value = os.getenv(name)
    print(f"Found value: for {name}")
    if value is None and default is None:
        print(f"Error: Required environment variable {name} is not set", file=sys.stderr)
        sys.exit(1)
    return value
