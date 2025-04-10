import os
import subprocess
import sys

def setup_backend():
    print("Setting up Python backend environment...")
    try:
        # Upgrade pip, setuptools, and wheel
        subprocess.check_call(['venv\\Scripts\\python', '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not upgrade pip, setuptools, or wheel. Proceeding with the existing versions. Error: {e}")
    try:
        # Install dependencies from requirements.txt
        subprocess.check_call(['venv\\Scripts\\pip', 'install', '-r', 'requirements.txt'])
    except subprocess.CalledProcessError as e:
        print(f"Error installing backend dependencies: {e}")
        return False
    return True

def setup_frontend():
    """Set up the frontend and install dependencies"""
    print("Setting up React frontend...")
    
    try:
        os.chdir('frontend')
        subprocess.check_call(['npm', 'install'])
        os.chdir('..')
        
        print("Frontend setup complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up frontend: {e}")
        os.chdir('..')
        return False

def main():
    """Main setup function"""
    print("=== ML Visualizer Setup ===")
    
    backend_success = setup_backend()
    #frontend_success = setup_frontend()
    
    if backend_success:
        print("\n=== Setup completed successfully! ===")
        print("\nTo run the application:")
        print("1. Start the backend server: python app.py")
        print("2. In another terminal, start the frontend: cd frontend && npm start")
        print("\nOr use the development script to run both: python run_dev.py")
        return 0
    else:
        print("\n=== Setup incomplete. Please check the errors above. ===")
        return 1

if __name__ == "__main__":
    sys.exit(main())
