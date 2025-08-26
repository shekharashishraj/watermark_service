#!/usr/bin/env python3
"""
Test script to verify PDF Watermark Service installation
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import PyPDF2
        print("✓ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"✗ PyPDF2 import failed: {e}")
        return False
    
    try:
        from reportlab.pdfgen import canvas
        print("✓ ReportLab imported successfully")
    except ImportError as e:
        print(f"✗ ReportLab import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    try:
        import werkzeug
        print("✓ Werkzeug imported successfully")
    except ImportError as e:
        print(f"✗ Werkzeug import failed: {e}")
        return False
    
    return True

def test_watermark_service():
    """Test if the watermark service can be imported and instantiated"""
    print("\nTesting watermark service...")
    
    try:
        from watermark_service import PDFWatermarker
        watermarker = PDFWatermarker()
        print("✓ PDFWatermarker imported and instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Watermark service test failed: {e}")
        return False

def test_flask_app():
    """Test if the Flask app can be imported"""
    print("\nTesting Flask application...")
    
    try:
        import app
        print("✓ Flask application imported successfully")
        return True
    except Exception as e:
        print(f"✗ Flask application test failed: {e}")
        return False

def test_directories():
    """Test if required directories exist or can be created"""
    print("\nTesting directory structure...")
    
    directories = ['uploads', 'outputs', 'templates', 'static/css', 'static/js']
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ Directory '{directory}' exists")
        else:
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✓ Directory '{directory}' created")
            except Exception as e:
                print(f"✗ Failed to create directory '{directory}': {e}")
                return False
    
    return True

def test_files():
    """Test if required files exist"""
    print("\nTesting required files...")
    
    required_files = [
        'app.py',
        'watermark_service.py',
        'requirements.txt',
        'README.md',
        'templates/index.html',
        'static/css/style.css',
        'static/js/script.js'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ File '{file_path}' exists")
        else:
            print(f"✗ File '{file_path}' missing")
            return False
    
    return True

def main():
    """Run all tests"""
    print("PDF Watermark Service - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Watermark Service", test_watermark_service),
        ("Flask Application", test_flask_app),
        ("Directory Structure", test_directories),
        ("Required Files", test_files)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  {test_name} failed!")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is successful.")
        print("\nTo start the application:")
        print("  python app.py")
        print("\nThen open your browser to: http://localhost:5000")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
