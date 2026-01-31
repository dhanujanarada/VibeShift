#!/usr/bin/env python
"""
Setup script for VibeShift - Automated environment setup
"""
import subprocess
import sys
import platform
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result

def check_uv_installed():
    """Check if uv is installed"""
    result = run_command("uv --version", check=False)
    return result.returncode == 0

def install_uv():
    """Install uv package manager"""
    print("\n📦 Installing uv package manager...")
    system = platform.system()
    
    if system == "Windows":
        cmd = 'powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
    else:
        cmd = 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    
    run_command(cmd)
    print("✅ uv installed successfully")

def setup_environment():
    """Set up the Python environment"""
    print("\n🐍 Setting up Python environment...")
    
    # Install Python 3.10 if needed
    print("Ensuring Python 3.10 is available...")
    run_command("uv python install 3.10")
    
    # Create/sync virtual environment
    print("Creating virtual environment and installing dependencies...")
    run_command("uv sync")
    
    print("✅ Environment setup complete")

def verify_installation():
    """Verify the installation"""
    print("\n🔍 Verifying installation...")
    
    # Check if venv exists
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found")
        return False
    
    # Check Python version in venv
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"
    
    if not python_exe.exists():
        print("❌ Python executable not found in virtual environment")
        return False
    
    # Check if torch is installed
    result = run_command(f'"{python_exe}" -c "import torch; print(torch.__version__)"', check=False)
    if result.returncode != 0:
        print("❌ PyTorch not installed correctly")
        return False
    
    print("✅ Installation verified successfully")
    return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("🎵 VibeShift - Automated Setup")
    print("=" * 60)
    
    # Check if uv is installed
    if not check_uv_installed():
        print("⚠️  uv not found. Installing...")
        install_uv()
    else:
        print("✅ uv is already installed")
    
    # Setup environment
    setup_environment()
    
    # Verify installation
    if verify_installation():
        print("\n" + "=" * 60)
        print("🎉 Setup completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Activate the virtual environment:")
        if platform.system() == "Windows":
            print("   .venv\\Scripts\\activate")
        else:
            print("   source .venv/bin/activate")
        print("\n2. Start the application:")
        print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
