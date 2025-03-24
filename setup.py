import os
import subprocess
import sys

def setup_backend():
    """Set up the backend environment and install dependencies"""
    print("Setting up Python backend environment...")
    
    try:
        # Check if venv exists, create if not
        if not os.path.exists('venv'):
            if sys.platform == 'win32':
                subprocess.check_call([sys.executable, '-m', 'venv', 'venv'])
            else:
                subprocess.check_call([sys.executable, '-m', 'venv', 'venv'])
        
        # Activate venv and install requirements
        if sys.platform == 'win32':
            pip_path = os.path.join('venv', 'Scripts', 'pip')
        else:
            pip_path = os.path.join('venv', 'bin', 'pip')
        
        subprocess.check_call([pip_path, 'install', '--upgrade', 'pip'])
        subprocess.check_call([pip_path, 'install', '-r', 'requirements.txt'])
        
        print("Backend setup complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up backend: {e}")
        return False

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
    frontend_success = setup_frontend()
    
    if backend_success and frontend_success:
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
