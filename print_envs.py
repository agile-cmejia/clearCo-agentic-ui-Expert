from components.get_required_env_vars import get_required_env_var

def main():
    print(f"Topic to use: {get_required_env_var("TOPIC")} from function")
    print(f"Site URL to crawl {get_required_env_var("SITE_URL")} from function")

if __name__ == "__main__":
    main()
