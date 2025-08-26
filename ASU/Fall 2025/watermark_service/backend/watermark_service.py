import os
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import tempfile
import math

class PDFWatermarker:
    def __init__(self):
        self.supported_positions = {
            'top-left': (50, 750),
            'top-center': (300, 750),
            'top-right': (550, 750),
            'center-left': (50, 400),
            'center': (300, 400),
            'center-right': (550, 400),
            'bottom-left': (50, 50),
            'bottom-center': (300, 50),
            'bottom-right': (550, 50)
        }
    
    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    def create_watermark_pdf(self, text, position, font_size, color, opacity, rotation, page_width, page_height):
        """Create a watermark PDF with the specified text and properties"""
        # Create temporary file for watermark
        watermark_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        watermark_file.close()
        
        # Create canvas for watermark
        c = canvas.Canvas(watermark_file.name, pagesize=(page_width, page_height))
        
        # Set transparency
        c.setFillAlpha(opacity)
        
        # Convert hex color to RGB
        rgb_color = self.hex_to_rgb(color)
        c.setFillColorRGB(*rgb_color)
        
        # Set font and size
        c.setFont("Helvetica-Bold", font_size)
        
        # Calculate text position
        if position in self.supported_positions:
            x, y = self.supported_positions[position]
        elif position == 'custom':
            # Custom position - this function doesn't handle custom_x/y
            # so default to center
            x, y = 300, 400
        else:
            # Try to parse as comma-separated coordinates
            try:
                coords = position.split(',')
                x, y = float(coords[0]), float(coords[1])
            except:
                x, y = 300, 400  # Default to center
        
        # Apply rotation
        if rotation != 0:
            c.saveState()
            c.translate(x, y)
            c.rotate(rotation)
            c.drawString(0, 0, text)
            c.restoreState()
        else:
            c.drawString(x, y, text)
        
        c.save()
        return watermark_file.name
    
    def create_multiple_watermarks_pdf(self, watermarks, page_width, page_height):
        """Create a watermark PDF with multiple watermarks"""
        # Create temporary file for watermark
        watermark_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        watermark_file.close()
        
        # Create canvas for watermark
        c = canvas.Canvas(watermark_file.name, pagesize=(page_width, page_height))
        
        # Apply each watermark
        for i, watermark in enumerate(watermarks):
            print(f"Processing watermark {i+1}: {watermark}")  # Debug print
            text = watermark.get('text', '')
            position = watermark.get('position', 'center')
            font_size = watermark.get('font_size', 24)
            color = watermark.get('color', '#000000')
            opacity = watermark.get('opacity', 0.5)
            rotation = watermark.get('rotation', 0)
            
            # Set transparency for this watermark
            c.setFillAlpha(opacity)
            
            # Convert hex color to RGB
            rgb_color = self.hex_to_rgb(color)
            c.setFillColorRGB(*rgb_color)
            
            # Set font and size
            c.setFont("Helvetica-Bold", font_size)
            
            # Calculate text position
            if position in self.supported_positions:
                x, y = self.supported_positions[position]
            elif position == 'custom':
                # Custom position from custom_x and custom_y
                x = float(watermark.get('custom_x', 300))
                # PDF coordinates start from bottom-left, but we need to adjust for text baseline
                y = page_height - float(watermark.get('custom_y', 400)) - font_size
                print(f"Custom position: x={x}, y={y}, custom_x={watermark.get('custom_x')}, custom_y={watermark.get('custom_y')}, page_height={page_height}, font_size={font_size}")  # Debug print
            else:
                # Try to parse as comma-separated coordinates
                try:
                    coords = position.split(',')
                    x, y = float(coords[0]), float(coords[1])
                    print(f"Parsed position: x={x}, y={y}")  # Debug print
                except:
                    x, y = 300, 400  # Default to center
                    print(f"Default position: x={x}, y={y}")  # Debug print
            
            # Apply rotation
            print(f"Drawing watermark '{text}' at position ({x}, {y}) with rotation {rotation}")  # Debug print
            if rotation != 0:
                c.saveState()
                c.translate(x, y)
                c.rotate(rotation)
                c.drawString(0, 0, text)
                c.restoreState()
            else:
                c.drawString(x, y, text)
        
        c.save()
        return watermark_file.name
    
    def add_watermark(self, input_path, output_path, text, position='center', 
                     font_size=24, color='#000000', opacity=0.5, rotation=0):
        """
        Add watermark to PDF file
        
        Args:
            input_path (str): Path to input PDF file
            output_path (str): Path to output PDF file
            text (str): Watermark text
            position (str): Position of watermark (e.g., 'center', 'top-left', '50,100')
            font_size (int): Font size
            color (str): Hex color code (e.g., '#FF0000')
            opacity (float): Opacity (0.0 to 1.0)
            rotation (int): Rotation angle in degrees
        """
        return self.add_multiple_watermarks(input_path, output_path, [{
            'text': text,
            'position': position,
            'font_size': font_size,
            'color': color,
            'opacity': opacity,
            'rotation': rotation
        }])

    def add_multiple_watermarks(self, input_path, output_path, watermarks):
        """
        Add multiple watermarks to PDF file
        
        Args:
            input_path (str): Path to input PDF file
            output_path (str): Path to output PDF file
            watermarks (list): List of watermark configurations
                Each watermark should be a dict with keys:
                - text (str): Watermark text
                - position (str): Position of watermark
                - font_size (int): Font size
                - color (str): Hex color code
                - opacity (float): Opacity (0.0 to 1.0)
                - rotation (int): Rotation angle in degrees
        """
        try:
            # Read input PDF
            reader = PdfReader(input_path)
            writer = PdfWriter()
            
            # Get first page to determine page size
            first_page = reader.pages[0]
            page_width = float(first_page.mediabox.width)
            page_height = float(first_page.mediabox.height)
            
            # Create combined watermark PDF with all watermarks
            combined_watermark_path = self.create_multiple_watermarks_pdf(
                watermarks, page_width, page_height
            )
            
            # Read combined watermark PDF
            watermark_reader = PdfReader(combined_watermark_path)
            watermark_page = watermark_reader.pages[0]
            
            # Apply watermark to all pages
            for page in reader.pages:
                # Merge watermark with page
                page.merge_page(watermark_page)
                writer.add_page(page)
            
            # Write output PDF
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)
            
            # Clean up temporary watermark file
            os.unlink(combined_watermark_path)
            
            return True
            
        except Exception as e:
            # Clean up temporary file if it exists
            if 'combined_watermark_path' in locals():
                try:
                    os.unlink(combined_watermark_path)
                except:
                    pass
            raise e
    
    def get_pdf_info(self, pdf_path):
        """Get basic information about a PDF file"""
        try:
            reader = PdfReader(pdf_path)
            info = {
                'num_pages': len(reader.pages),
                'page_size': {
                    'width': float(reader.pages[0].mediabox.width),
                    'height': float(reader.pages[0].mediabox.height)
                }
            }
            return info
        except Exception as e:
            raise e
    
    def add_diagonal_watermark(self, input_path, output_path, text, 
                             font_size=24, color='#000000', opacity=0.3, 
                             spacing=100, start_position='top-left'):
        """
        Add diagonal watermark across the entire page
        
        Args:
            input_path (str): Path to input PDF file
            output_path (str): Path to output PDF file
            text (str): Watermark text
            font_size (int): Font size
            color (str): Hex color code
            opacity (float): Opacity
            spacing (int): Spacing between watermarks
            start_position (str): Starting position for diagonal pattern
        """
        try:
            # Read input PDF
            reader = PdfReader(input_path)
            writer = PdfWriter()
            
            # Get page dimensions
            first_page = reader.pages[0]
            page_width = float(first_page.mediabox.width)
            page_height = float(first_page.mediabox.height)
            
            # Create diagonal watermark pattern
            watermark_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            watermark_file.close()
            
            c = canvas.Canvas(watermark_file.name, pagesize=(page_width, page_height))
            c.setFillAlpha(opacity)
            rgb_color = self.hex_to_rgb(color)
            c.setFillColorRGB(*rgb_color)
            c.setFont("Helvetica-Bold", font_size)
            
            # Calculate diagonal pattern
            text_width = c.stringWidth(text, "Helvetica-Bold", font_size)
            diagonal_length = math.sqrt(page_width**2 + page_height**2)
            num_watermarks = int(diagonal_length / spacing) + 1
            
            for i in range(num_watermarks):
                x = i * spacing
                y = i * spacing
                
                if x < page_width and y < page_height:
                    c.saveState()
                    c.translate(x, y)
                    c.rotate(45)
                    c.drawString(0, 0, text)
                    c.restoreState()
            
            c.save()
            
            # Read watermark PDF
            watermark_reader = PdfReader(watermark_file.name)
            watermark_page = watermark_reader.pages[0]
            
            # Apply to all pages
            for page in reader.pages:
                page.merge_page(watermark_page)
                writer.add_page(page)
            
            # Write output PDF
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)
            
            # Clean up
            os.unlink(watermark_file.name)
            
            return True
            
        except Exception as e:
            if 'watermark_file' in locals():
                try:
                    os.unlink(watermark_file.name)
                except:
                    pass
            raise e
