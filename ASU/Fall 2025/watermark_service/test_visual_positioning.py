#!/usr/bin/env python3
"""
Test script for visual positioning functionality
"""

import os
from watermark_service import PDFWatermarker
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_test_pdf():
    """Create a test PDF for visual positioning"""
    filename = "test_visual.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    
    c.setFont("Helvetica", 16)
    c.drawString(100, 750, "Test Document for Visual Positioning")
    c.drawString(100, 720, "This document will test drag-and-drop watermark positioning.")
    
    c.setFont("Helvetica", 12)
    c.drawString(100, 650, "Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
    c.drawString(100, 630, "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")
    c.drawString(100, 610, "Ut enim ad minim veniam, quis nostrud exercitation ullamco.")
    
    # Add some visual elements to help with positioning
    c.setFont("Helvetica", 10)
    c.drawString(50, 500, "Top Left Area")
    c.drawString(300, 500, "Top Center Area")
    c.drawString(550, 500, "Top Right Area")
    
    c.drawString(50, 300, "Center Left")
    c.drawString(300, 300, "Center")
    c.drawString(550, 300, "Center Right")
    
    c.drawString(50, 100, "Bottom Left")
    c.drawString(300, 100, "Bottom Center")
    c.drawString(550, 100, "Bottom Right")
    
    c.save()
    print(f"✓ Created test PDF: {filename}")
    return filename

def test_visual_positioning():
    """Test visual positioning with custom coordinates"""
    print("Testing Visual Positioning Functionality")
    print("=" * 50)
    
    # Create test PDF
    input_file = create_test_pdf()
    output_file = "test_visual_watermarked.pdf"
    
    # Define watermarks with custom positions (simulating drag-and-drop)
    watermarks = [
        {
            'text': 'DRAG ME',
            'position': '150,200',  # Custom position (simulated drag)
            'font_size': 20,
            'color': '#FF0000',
            'opacity': 0.6,
            'rotation': 0
        },
        {
            'text': 'POSITIONED',
            'position': '400,350',  # Another custom position
            'font_size': 18,
            'color': '#0000FF',
            'opacity': 0.7,
            'rotation': 15
        },
        {
            'text': 'VISUALLY',
            'position': '250,150',  # Third custom position
            'font_size': 16,
            'color': '#00FF00',
            'opacity': 0.5,
            'rotation': -10
        }
    ]
    
    try:
        # Apply watermarks with custom positions
        watermarker = PDFWatermarker()
        watermarker.add_multiple_watermarks(input_file, output_file, watermarks)
        
        print("✓ Visual positioning test completed successfully!")
        print(f"✓ Output file: {output_file}")
        print(f"✓ Applied {len(watermarks)} watermarks with custom positions:")
        
        for i, wm in enumerate(watermarks, 1):
            print(f"  {i}. '{wm['text']}' at position {wm['position']}")
            print(f"     - Size: {wm['font_size']}, Color: {wm['color']}, Opacity: {wm['opacity']}")
        
        print("\n🎯 Visual positioning features:")
        print("  - Drag and drop watermarks to any position")
        print("  - Real-time position coordinates display")
        print("  - Visual feedback during dragging")
        print("  - Automatic position saving")
        print("  - Reset positions to center")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing visual positioning: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    files_to_remove = ['test_visual.pdf', 'test_visual_watermarked.pdf']
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ Removed: {file}")

if __name__ == "__main__":
    success = test_visual_positioning()
    
    if success:
        print("\n🎉 Visual positioning test completed successfully!")
        print("\n📋 How to use visual positioning:")
        print("  1. Upload a PDF file")
        print("  2. Add watermarks in the configuration section")
        print("  3. Click 'Preview' to open visual positioning")
        print("  4. Drag watermarks to desired positions")
        print("  5. Click 'Save Positions' to update coordinates")
        print("  6. Click 'Apply Watermarks' to create final PDF")
        
        response = input("\nDo you want to clean up test files? (y/n): ")
        if response.lower() in ['y', 'yes']:
            cleanup_test_files()
        else:
            print("Test files preserved for inspection.")
    else:
        print("\n❌ Visual positioning test failed!")
    
    exit(0 if success else 1)
