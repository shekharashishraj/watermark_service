#!/usr/bin/env python3
"""
Main Flask Backend Application for PDF Watermark Service
Handles real-time watermark positioning via WebSockets
"""

import os
import uuid
import json
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from werkzeug.utils import secure_filename
from watermark_service import PDFWatermarker
import tempfile
import shutil

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enable CORS for React frontend
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000", async_mode='eventlet')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Store active sessions
active_sessions = {}

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'PDF Watermark Service is running'})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload PDF file"""
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
        
        # Store session info
        active_sessions[unique_id] = {
            'filename': filename,
            'file_path': file_path,
            'watermarks': [],
            'room': f"session_{unique_id}"
        }
        
        return jsonify({
            'success': True,
            'file_id': unique_id,
            'filename': filename,
            'message': 'File uploaded successfully'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/watermark', methods=['POST'])
def apply_watermark():
    """Apply watermarks to PDF"""
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        watermarks = data.get('watermarks', [])
        
        if not watermarks:
            return jsonify({'error': 'No watermarks specified'}), 400
        
        # Find the uploaded file
        if file_id not in active_sessions:
            return jsonify({'error': 'File not found'}), 404
        
        session = active_sessions[file_id]
        input_file = session['file_path']
        
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

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download watermarked PDF"""
    try:
        return send_file(
            os.path.join(app.config['OUTPUT_FOLDER'], filename),
            as_attachment=True,
            download_name=filename
        )
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/original/<file_id>')
def get_original_file(file_id):
    """Get original uploaded PDF for preview"""
    try:
        if file_id not in active_sessions:
            return jsonify({'error': 'File not found'}), 404
        
        session = active_sessions[file_id]
        file_path = session['file_path']
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            file_path,
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/preview/<file_id>')
def get_pdf_preview(file_id):
    """Get PDF preview as image for visual positioning"""
    try:
        if file_id not in active_sessions:
            return jsonify({'error': 'File not found'}), 404
        
        session = active_sessions[file_id]
        file_path = session['file_path']
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # For now, return the PDF directly
        # In a production environment, you'd convert PDF to image
        return send_file(
            file_path,
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup_files():
    """Clean up temporary files"""
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        
        if file_id in active_sessions:
            session = active_sessions[file_id]
            
            # Clean up uploaded file
            if os.path.exists(session['file_path']):
                os.remove(session['file_path'])
            
            # Clean up output files
            output_dir = app.config['OUTPUT_FOLDER']
            for filename in os.listdir(output_dir):
                if file_id in filename:
                    os.remove(os.path.join(output_dir, filename))
            
            # Remove session
            del active_sessions[file_id]
        
        return jsonify({'success': True, 'message': 'Files cleaned up successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to watermark service'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('join_session')
def handle_join_session(data):
    """Join a watermarking session"""
    file_id = data.get('file_id')
    if file_id and file_id in active_sessions:
        room = active_sessions[file_id]['room']
        join_room(room)
        emit('session_joined', {
            'file_id': file_id,
            'watermarks': active_sessions[file_id]['watermarks']
        })
        print(f"Client {request.sid} joined session {file_id}")

@socketio.on('leave_session')
def handle_leave_session(data):
    """Leave a watermarking session"""
    file_id = data.get('file_id')
    if file_id and file_id in active_sessions:
        room = active_sessions[file_id]['room']
        leave_room(room)
        emit('session_left', {'file_id': file_id})

@socketio.on('update_watermark_position')
def handle_update_position(data):
    """Update watermark position in real-time"""
    file_id = data.get('file_id')
    watermark_id = data.get('watermark_id')
    position = data.get('position')
    
    if file_id in active_sessions:
        session = active_sessions[file_id]
        room = session['room']
        
        # Update watermark position
        for watermark in session['watermarks']:
            if watermark.get('id') == watermark_id:
                watermark['position'] = position
                break
        
        # Broadcast update to all clients in the room
        emit('watermark_position_updated', {
            'watermark_id': watermark_id,
            'position': position
        }, room=room)
        
        print(f"Updated position for watermark {watermark_id} in session {file_id}")

@socketio.on('add_watermark')
def handle_add_watermark(data):
    """Add a new watermark"""
    file_id = data.get('file_id')
    watermark_data = data.get('watermark')
    
    if file_id in active_sessions:
        session = active_sessions[file_id]
        room = session['room']
        
        # Add watermark to session
        session['watermarks'].append(watermark_data)
        
        # Broadcast to all clients in the room
        emit('watermark_added', {
            'watermark': watermark_data
        }, room=room)
        
        print(f"Added watermark to session {file_id}")

@socketio.on('remove_watermark')
def handle_remove_watermark(data):
    """Remove a watermark"""
    file_id = data.get('file_id')
    watermark_id = data.get('watermark_id')
    
    if file_id in active_sessions:
        session = active_sessions[file_id]
        room = session['room']
        
        # Remove watermark from session
        session['watermarks'] = [w for w in session['watermarks'] if w.get('id') != watermark_id]
        
        # Broadcast to all clients in the room
        emit('watermark_removed', {
            'watermark_id': watermark_id
        }, room=room)
        
        print(f"Removed watermark {watermark_id} from session {file_id}")

@socketio.on('update_watermark_properties')
def handle_update_properties(data):
    """Update watermark properties (text, color, size, etc.)"""
    file_id = data.get('file_id')
    watermark_id = data.get('watermark_id')
    properties = data.get('properties')
    
    if file_id in active_sessions:
        session = active_sessions[file_id]
        room = session['room']
        
        # Update watermark properties
        for watermark in session['watermarks']:
            if watermark.get('id') == watermark_id:
                watermark.update(properties)
                break
        
        # Broadcast to all clients in the room
        emit('watermark_properties_updated', {
            'watermark_id': watermark_id,
            'properties': properties
        }, room=room)
        
        print(f"Updated properties for watermark {watermark_id} in session {file_id}")

if __name__ == '__main__':
    print("Starting PDF Watermark Backend Server...")
    print("Backend Server running on: http://localhost:5001")
    print("WebSocket support enabled for real-time positioning")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
