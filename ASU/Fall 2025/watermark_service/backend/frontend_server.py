#!/usr/bin/env python3
"""
Frontend Server for PDF Watermark Service
Serves the web interface and communicates with the API server
"""

from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

# Serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# Serve main page
@app.route('/')
def index():
    return render_template('index.html')

# Serve favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')

if __name__ == '__main__':
    print("Starting PDF Watermark Frontend Server...")
    print("Frontend Server running on: http://localhost:5000")
    print("API Server should be running on: http://localhost:5001")
    app.run(host='0.0.0.0', port=5000, debug=True)
