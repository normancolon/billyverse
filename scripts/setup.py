import subprocess
import sys
import os
import venv
import logging
from pythonjsonlogger import jsonlogger

# Configure logging
logger = logging.getLogger("billieverse.setup")
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

def create_venv():
    """Create virtual environment"""
    venv_path = "venv"
    logger.info(f"Creating virtual environment at {venv_path}")
    venv.create(venv_path, with_pip=True)
    return venv_path

def get_python_path(venv_path):
    """Get path to Python executable in virtual environment"""
    if sys.platform == "win32":
        return os.path.join(venv_path, "Scripts", "python.exe")
    return os.path.join(venv_path, "bin", "python")

def install_dependencies(python_path):
    """Install dependencies from requirements.txt"""
    logger.info("Installing dependencies...")
    try:
        subprocess.run(
            [python_path, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True
        )
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        raise

def main():
    """Main setup function"""
    try:
        venv_path = create_venv()
        python_path = get_python_path(venv_path)
        install_dependencies(python_path)
        
        logger.info("""
Setup completed successfully!

To activate the virtual environment:
- Windows: .\\venv\\Scripts\\activate
- Linux/Mac: source venv/bin/activate

To start the development server:
uvicorn api.main:app --reload
        """)
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 