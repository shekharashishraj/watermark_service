import os
import uuid
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from watermark_service import PDFWatermarker
import tempfile
import shutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
        file.save(file_path)
        
        return jsonify({
            'success': True,
            'file_id': unique_id,
            'filename': filename,
            'message': 'File uploaded successfully'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/watermark', methods=['POST'])
def apply_watermark():
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        watermarks = data.get('watermarks', [])
        
        # Handle single watermark for backward compatibility
        if not watermarks and data.get('watermark_text'):
            watermarks = [{
                'text': data.get('watermark_text', ''),
                'position': data.get('position', 'center'),
                'font_size': int(data.get('font_size', 24)),
                'color': data.get('color', '#000000'),
                'opacity': float(data.get('opacity', 0.5)),
                'rotation': int(data.get('rotation', 0))
            }]
        
        if not watermarks:
            return jsonify({'error': 'No watermarks specified'}), 400
        
        # Find the uploaded file
        upload_dir = app.config['UPLOAD_FOLDER']
        input_file = None
        for filename in os.listdir(upload_dir):
            if filename.startswith(file_id):
                input_file = os.path.join(upload_dir, filename)
                break
        
        if not input_file:
            return jsonify({'error': 'File not found'}), 404
        
        # Create watermarker instance
        watermarker = PDFWatermarker()
        
        # Generate output filename
        output_filename = f"watermarked_{file_id}.pdf"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Apply watermarks
        watermarker.add_multiple_watermarks(input_file, output_path, watermarks)
        
        return jsonify({
            'success': True,
            'output_file': output_filename,
            'message': f'{len(watermarks)} watermark(s) applied successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(
            os.path.join(app.config['OUTPUT_FOLDER'], filename),
            as_attachment=True,
            download_name=filename
        )
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/preview/<file_id>')
def preview_file(file_id):
    try:
        # Get watermark data from request parameters
        watermarks_data = request.args.get('watermarks', '[]')
        
        try:
            import json
            watermarks = json.loads(watermarks_data)
            # Ensure watermarks is a list
            if not isinstance(watermarks, list):
                watermarks = []
        except Exception as e:
            print(f"Error parsing watermarks: {e}")
            # Fallback to single watermark for backward compatibility
            watermark_text = request.args.get('text', 'PREVIEW')
            position = request.args.get('position', 'center')
            font_size = int(request.args.get('font_size', 24))
            color = request.args.get('color', '#FF0000')
            opacity = float(request.args.get('opacity', 0.3))
            rotation = int(request.args.get('rotation', 45))
            
            watermarks = [{
                'text': watermark_text,
                'position': position,
                'font_size': font_size,
                'color': color,
                'opacity': opacity,
                'rotation': rotation
            }]
        
        # If no watermarks provided, create a default one
        if not watermarks:
            watermarks = [{
                'text': 'PREVIEW',
                'position': 'center',
                'font_size': 24,
                'color': '#FF0000',
                'opacity': 0.3,
                'rotation': 45
            }]
        
        # Find the uploaded file
        upload_dir = app.config['UPLOAD_FOLDER']
        input_file = None
        for filename in os.listdir(upload_dir):
            if filename.startswith(file_id):
                input_file = os.path.join(upload_dir, filename)
                break
        
        if not input_file:
            return jsonify({'error': 'File not found'}), 404
        
        # Create a temporary watermarked file for preview
        watermarker = PDFWatermarker()
        temp_output = os.path.join(app.config['OUTPUT_FOLDER'], f"preview_{file_id}.pdf")
        
        # Apply watermarks with actual settings for preview
        watermarker.add_multiple_watermarks(input_file, temp_output, watermarks)
        
        return send_file(temp_output, mimetype='application/pdf')
        
    except Exception as e:
        print(f"Preview error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/original/<file_id>')
def get_original_file(file_id):
    """Get the original PDF file without watermarks"""
    try:
        # Find the uploaded file
        upload_dir = app.config['UPLOAD_FOLDER']
        input_file = None
        for filename in os.listdir(upload_dir):
            if filename.startswith(file_id):
                input_file = os.path.join(upload_dir, filename)
                break
        
        if not input_file:
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(input_file, mimetype='application/pdf')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        
        # Clean up uploaded file
        upload_dir = app.config['UPLOAD_FOLDER']
        for filename in os.listdir(upload_dir):
            if filename.startswith(file_id):
                os.remove(os.path.join(upload_dir, filename))
                break
        
        # Clean up output files
        output_dir = app.config['OUTPUT_FOLDER']
        for filename in os.listdir(output_dir):
            if file_id in filename:
                os.remove(os.path.join(output_dir, filename))
        
        return jsonify({'success': True, 'message': 'Files cleaned up successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
