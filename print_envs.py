import sys
import os
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
    print(f"Found value: {value} for {name}")
    if value is None and default is None:
        print(f"Error: Required environment variable {name} is not set", file=sys.stderr)
        sys.exit(1)
    return value


def main():
    print(f"Topic to use: {get_required_env_var("TOPIC")} from function")
    print(f"Topic to use: {os.getenv("TOPIC")} from env")
    print(f"Site URL to crawl {get_required_env_var("SITE_URL")} from function")
    print(f"Site URL to crawl {os.getenv("SITE_URL")} from env")

if __name__ == "__main__":
    main()
