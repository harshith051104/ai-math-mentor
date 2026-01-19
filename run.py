import sys
import os
from pathlib import Path

# Add the current directory to sys.path to ensure 'aimath' module is found
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Also try to resolve relative to this script
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

import streamlit.web.cli as stcli

def main():
    print(f"Starting Math Mentor App...")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Python Path: {sys.path[0]}")
    
    app_path = script_dir / "aimath" / "ui" / "app.py"
    
    if not app_path.exists():
        print(f"Error: Could not find app at {app_path}")
        return

    # Set up arguments for streamlit
    sys.argv = ["streamlit", "run", str(app_path)]
    
    # Run streamlit
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
