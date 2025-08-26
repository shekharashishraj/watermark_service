#!/usr/bin/env python3
"""
Test script for multiple watermark functionality
"""

import os
from watermark_service import PDFWatermarker
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_test_pdf():
    """Create a test PDF"""
    filename = "test_multiple.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    
    c.setFont("Helvetica", 16)
    c.drawString(100, 750, "Test Document for Multiple Watermarks")
    c.drawString(100, 720, "This document will have multiple watermarks applied.")
    
    c.setFont("Helvetica", 12)
    c.drawString(100, 650, "Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
    c.drawString(100, 630, "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")
    c.drawString(100, 610, "Ut enim ad minim veniam, quis nostrud exercitation ullamco.")
    
    c.save()
    print(f"✓ Created test PDF: {filename}")
    return filename

def test_multiple_watermarks():
    """Test multiple watermark functionality"""
    print("Testing Multiple Watermark Functionality")
    print("=" * 50)
    
    # Create test PDF
    input_file = create_test_pdf()
    output_file = "test_multiple_watermarked.pdf"
    
    # Define multiple watermarks
    watermarks = [
        {
            'text': 'CONFIDENTIAL',
            'position': 'top-center',
            'font_size': 24,
            'color': '#FF0000',
            'opacity': 0.5,
            'rotation': 0
        },
        {
            'text': 'DRAFT',
            'position': 'bottom-right',
            'font_size': 18,
            'color': '#0000FF',
            'opacity': 0.7,
            'rotation': 0
        },
        {
            'text': 'WATERMARK',
            'position': 'center',
            'font_size': 36,
            'color': '#00FF00',
            'opacity': 0.3,
            'rotation': 45
        },
        {
            'text': 'CUSTOM',
            'position': '200,300',
            'font_size': 16,
            'color': '#FF6600',
            'opacity': 0.6,
            'rotation': 15
        }
    ]
    
    try:
        # Apply multiple watermarks
        watermarker = PDFWatermarker()
        watermarker.add_multiple_watermarks(input_file, output_file, watermarks)
        
        print("✓ Multiple watermarks applied successfully!")
        print(f"✓ Output file: {output_file}")
        print(f"✓ Applied {len(watermarks)} watermarks:")
        
        for i, wm in enumerate(watermarks, 1):
            print(f"  {i}. '{wm['text']}' at {wm['position']} (size: {wm['font_size']}, color: {wm['color']})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error applying multiple watermarks: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    files_to_remove = ['test_multiple.pdf', 'test_multiple_watermarked.pdf']
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ Removed: {file}")

if __name__ == "__main__":
    success = test_multiple_watermarks()
    
    if success:
        print("\n🎉 Multiple watermark test completed successfully!")
        response = input("\nDo you want to clean up test files? (y/n): ")
        if response.lower() in ['y', 'yes']:
            cleanup_test_files()
        else:
            print("Test files preserved for inspection.")
    else:
        print("\n❌ Multiple watermark test failed!")
    
    exit(0 if success else 1)
