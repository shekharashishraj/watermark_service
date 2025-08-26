#!/usr/bin/env python3
"""
Example script demonstrating PDF Watermark Service usage
"""

import os
from watermark_service import PDFWatermarker

def create_sample_pdf():
    """Create a sample PDF for testing"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    # Create a simple PDF
    filename = "sample.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    
    # Add some content
    c.setFont("Helvetica", 16)
    c.drawString(100, 750, "Sample PDF Document")
    c.drawString(100, 720, "This is a test document for watermarking.")
    c.drawString(100, 690, "It contains multiple lines of text.")
    c.drawString(100, 660, "The watermark will be applied to this document.")
    
    # Add more content
    c.setFont("Helvetica", 12)
    c.drawString(100, 600, "Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
    c.drawString(100, 580, "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")
    c.drawString(100, 560, "Ut enim ad minim veniam, quis nostrud exercitation ullamco.")
    
    c.save()
    print(f"✓ Created sample PDF: {filename}")
    return filename

def example_basic_watermark():
    """Example: Basic watermark in center"""
    print("\n=== Example 1: Basic Center Watermark ===")
    
    input_file = "sample.pdf"
    output_file = "watermarked_basic.pdf"
    
    watermarker = PDFWatermarker()
    watermarker.add_watermark(
        input_file, 
        output_file, 
        "CONFIDENTIAL", 
        "center", 
        24, 
        "#FF0000", 
        0.5, 
        0
    )
    
    print(f"✓ Applied basic watermark: {output_file}")

def example_corner_watermark():
    """Example: Watermark in top-right corner"""
    print("\n=== Example 2: Corner Watermark ===")
    
    input_file = "sample.pdf"
    output_file = "watermarked_corner.pdf"
    
    watermarker = PDFWatermarker()
    watermarker.add_watermark(
        input_file, 
        output_file, 
        "DRAFT", 
        "top-right", 
        18, 
        "#0000FF", 
        0.7, 
        0
    )
    
    print(f"✓ Applied corner watermark: {output_file}")

def example_rotated_watermark():
    """Example: Rotated watermark"""
    print("\n=== Example 3: Rotated Watermark ===")
    
    input_file = "sample.pdf"
    output_file = "watermarked_rotated.pdf"
    
    watermarker = PDFWatermarker()
    watermarker.add_watermark(
        input_file, 
        output_file, 
        "WATERMARK", 
        "center", 
        36, 
        "#00FF00", 
        0.3, 
        45
    )
    
    print(f"✓ Applied rotated watermark: {output_file}")

def example_custom_position():
    """Example: Custom position watermark"""
    print("\n=== Example 4: Custom Position Watermark ===")
    
    input_file = "sample.pdf"
    output_file = "watermarked_custom.pdf"
    
    watermarker = PDFWatermarker()
    watermarker.add_watermark(
        input_file, 
        output_file, 
        "CUSTOM", 
        "200,300",  # Custom coordinates
        20, 
        "#FF6600", 
        0.6, 
        15
    )
    
    print(f"✓ Applied custom position watermark: {output_file}")

def example_diagonal_watermark():
    """Example: Diagonal watermark pattern"""
    print("\n=== Example 5: Diagonal Watermark Pattern ===")
    
    input_file = "sample.pdf"
    output_file = "watermarked_diagonal.pdf"
    
    watermarker = PDFWatermarker()
    watermarker.add_diagonal_watermark(
        input_file, 
        output_file, 
        "DIAGONAL", 
        16, 
        "#800080", 
        0.2, 
        80
    )
    
    print(f"✓ Applied diagonal watermark pattern: {output_file}")

def example_multiple_watermarks():
    """Example: Multiple watermarks with different settings"""
    print("\n=== Example 6: Multiple Watermarks ===")
    
    input_file = "sample.pdf"
    output_file = "watermarked_multiple.pdf"
    
    watermarker = PDFWatermarker()
    
    # First watermark
    watermarker.add_watermark(
        input_file, 
        "temp1.pdf", 
        "TOP", 
        "top-center", 
        16, 
        "#FF0000", 
        0.4, 
        0
    )
    
    # Second watermark on the result
    watermarker.add_watermark(
        "temp1.pdf", 
        output_file, 
        "BOTTOM", 
        "bottom-center", 
        16, 
        "#0000FF", 
        0.4, 
        0
    )
    
    # Clean up temporary file
    if os.path.exists("temp1.pdf"):
        os.remove("temp1.pdf")
    
    print(f"✓ Applied multiple watermarks: {output_file}")

def cleanup_files():
    """Clean up generated files"""
    print("\n=== Cleaning up files ===")
    
    files_to_remove = [
        "sample.pdf",
        "watermarked_basic.pdf",
        "watermarked_corner.pdf", 
        "watermarked_rotated.pdf",
        "watermarked_custom.pdf",
        "watermarked_diagonal.pdf",
        "watermarked_multiple.pdf"
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ Removed: {file}")

def main():
    """Run all examples"""
    print("PDF Watermark Service - Examples")
    print("=" * 40)
    
    try:
        # Create sample PDF
        sample_file = create_sample_pdf()
        
        # Run examples
        example_basic_watermark()
        example_corner_watermark()
        example_rotated_watermark()
        example_custom_position()
        example_diagonal_watermark()
        example_multiple_watermarks()
        
        print("\n" + "=" * 40)
        print("🎉 All examples completed successfully!")
        print("\nGenerated files:")
        for file in os.listdir('.'):
            if file.startswith('watermarked_') and file.endswith('.pdf'):
                print(f"  - {file}")
        
        # Ask if user wants to clean up
        response = input("\nDo you want to clean up the generated files? (y/n): ")
        if response.lower() in ['y', 'yes']:
            cleanup_files()
        else:
            print("Files preserved for inspection.")
            
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
