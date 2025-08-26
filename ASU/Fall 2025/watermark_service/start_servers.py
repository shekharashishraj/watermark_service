#!/usr/bin/env python3
"""
Startup script for PDF Watermark Service
Runs both backend and frontend servers
"""

import subprocess
import sys
import time
import signal
import os

def install_backend_dependencies():
    """Install backend dependencies"""
    print("Installing backend dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'backend/requirements.txt'], check=True)
        print("✅ Backend dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install backend dependencies")
        return False

def install_frontend_dependencies():
    """Install frontend dependencies"""
    print("Installing frontend dependencies...")
    try:
        subprocess.run(['npm', 'install'], cwd='frontend', check=True)
        print("✅ Frontend dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install frontend dependencies")
        return False

def start_backend_server():
    """Start the backend API server"""
    print("Starting backend API server...")
    try:
        backend_process = subprocess.Popen([
            sys.executable, 'backend/app.py'
        ], cwd='backend', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return backend_process
    except Exception as e:
        print(f"❌ Failed to start backend server: {e}")
        return None

def start_frontend_server():
    """Start the React frontend server"""
    print("Starting React frontend server...")
    try:
        frontend_process = subprocess.Popen([
            'npm', 'start'
        ], cwd='frontend', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return frontend_process
    except Exception as e:
        print(f"❌ Failed to start frontend server: {e}")
        return None

def main():
    print("🚀 Starting PDF Watermark Service...")
    print("=" * 60)
    
    # Check if Node.js is installed
    try:
        subprocess.run(['node', '--version'], check=True, capture_output=True)
        print("✅ Node.js is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Node.js is not installed. Please install Node.js first.")
        print("   Download from: https://nodejs.org/")
        sys.exit(1)
    
    # Check if npm is installed
    try:
        subprocess.run(['npm', '--version'], check=True, capture_output=True)
        print("✅ npm is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ npm is not installed. Please install npm first.")
        sys.exit(1)
    
    # Install backend dependencies
    if not install_backend_dependencies():
        sys.exit(1)
    
    # Install frontend dependencies
    if not install_frontend_dependencies():
        sys.exit(1)
    
    # Start backend server
    backend_process = start_backend_server()
    if not backend_process:
        sys.exit(1)
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    # Start frontend server
    frontend_process = start_frontend_server()
    if not frontend_process:
        backend_process.terminate()
        sys.exit(1)
    
    print("\n🎉 Both servers started successfully!")
    print("=" * 60)
    print("📱 React Frontend: http://localhost:3000")
    print("🔧 Backend API Server: http://localhost:5001")
    print("🔌 WebSocket Server: ws://localhost:5001")
    print("\n💡 Features:")
    print("   • Real-time drag-and-drop watermark positioning")
    print("   • WebSocket communication for instant updates")
    print("   • Multiple watermark support")
    print("   • Live coordinate tracking")
    print("   • Modern React UI with animations")
    print("\n🛑 Press Ctrl+C to stop both servers")
    print("=" * 60)
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend server stopped unexpectedly")
                break
                
            if frontend_process.poll() is not None:
                print("❌ Frontend server stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        
        # Terminate processes
        if backend_process:
            backend_process.terminate()
            print("✅ Backend server stopped")
            
        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend server stopped")
            
        print("👋 Goodbye!")

if __name__ == '__main__':
    main()
