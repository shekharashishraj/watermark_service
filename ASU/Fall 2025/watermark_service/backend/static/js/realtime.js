// Real-time WebSocket communication for PDF Watermark Service
class WatermarkRealtimeClient {
    constructor() {
        this.socket = null;
        this.currentFileId = null;
        this.currentSession = null;
        this.isConnected = false;
        this.apiBaseUrl = 'http://localhost:5001/api';
        this.wsUrl = 'http://localhost:5001';
        
        this.watermarks = [];
        this.watermarkCounter = 0;
        
        this.init();
    }
    
    init() {
        this.connectWebSocket();
        this.setupEventListeners();
    }
    
    connectWebSocket() {
        // Load Socket.IO client
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js';
        script.onload = () => {
            this.socket = io(this.wsUrl);
            this.setupSocketEvents();
        };
        document.head.appendChild(script);
    }
    
    setupSocketEvents() {
        this.socket.on('connect', () => {
            console.log('Connected to WebSocket server');
            this.isConnected = true;
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from WebSocket server');
            this.isConnected = false;
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('session_joined', (data) => {
            console.log('Joined session:', data);
            this.currentSession = data;
            this.watermarks = data.watermarks || [];
            this.renderWatermarks();
        });
        
        this.socket.on('watermark_position_updated', (data) => {
            console.log('Watermark position updated:', data);
            this.updateWatermarkPosition(data.watermark_id, data.position);
        });
        
        this.socket.on('watermark_added', (data) => {
            console.log('Watermark added:', data);
            this.watermarks.push(data.watermark);
            this.renderWatermarks();
        });
        
        this.socket.on('watermark_removed', (data) => {
            console.log('Watermark removed:', data);
            this.watermarks = this.watermarks.filter(w => w.id !== data.watermark_id);
            this.renderWatermarks();
        });
        
        this.socket.on('watermark_properties_updated', (data) => {
            console.log('Watermark properties updated:', data);
            this.updateWatermarkProperties(data.watermark_id, data.properties);
        });
    }
    
    setupEventListeners() {
        // File upload
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        }
        
        if (uploadArea) {
            uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
            uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        }
    }
    
    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) {
            await this.uploadFile(file);
        }
    }
    
    async uploadFile(file) {
        if (file.type !== 'application/pdf') {
            this.showError('Please select a PDF file.');
            return;
        }
        
        this.showLoading('Uploading file...');
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/upload`, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentFileId = data.file_id;
                this.joinSession(data.file_id);
                this.showSuccess('File uploaded successfully!');
                this.showConfigurationSection();
            } else {
                this.showError(data.error || 'Upload failed.');
            }
        } catch (error) {
            this.showError('Upload failed. Please try again.');
            console.error('Upload error:', error);
        } finally {
            this.hideLoading();
        }
    }
    
    joinSession(fileId) {
        if (this.socket && this.isConnected) {
            this.socket.emit('join_session', { file_id: fileId });
        }
    }
    
    leaveSession() {
        if (this.socket && this.isConnected && this.currentFileId) {
            this.socket.emit('leave_session', { file_id: this.currentFileId });
        }
    }
    
    addWatermark() {
        this.watermarkCounter++;
        const watermarkId = `watermark-${this.watermarkCounter}`;
        
        const watermark = {
            id: watermarkId,
            text: 'CONFIDENTIAL',
            position: 'center',
            font_size: 24,
            color: '#000000',
            opacity: 0.5,
            rotation: 0,
            custom_x: '',
            custom_y: ''
        };
        
        if (this.socket && this.isConnected && this.currentFileId) {
            this.socket.emit('add_watermark', {
                file_id: this.currentFileId,
                watermark: watermark
            });
        }
    }
    
    removeWatermark(watermarkId) {
        if (this.socket && this.isConnected && this.currentFileId) {
            this.socket.emit('remove_watermark', {
                file_id: this.currentFileId,
                watermark_id: watermarkId
            });
        }
    }
    
    updateWatermarkPosition(watermarkId, position) {
        if (this.socket && this.isConnected && this.currentFileId) {
            this.socket.emit('update_watermark_position', {
                file_id: this.currentFileId,
                watermark_id: watermarkId,
                position: position
            });
        }
    }
    
    updateWatermarkProperties(watermarkId, properties) {
        if (this.socket && this.isConnected && this.currentFileId) {
            this.socket.emit('update_watermark_properties', {
                file_id: this.currentFileId,
                watermark_id: watermarkId,
                properties: properties
            });
        }
    }
    
    async applyWatermarks() {
        if (!this.currentFileId || this.watermarks.length === 0) {
            this.showError('Please upload a file and add watermarks first.');
            return;
        }
        
        this.showLoading('Applying watermarks...');
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/watermark`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    file_id: this.currentFileId,
                    watermarks: this.watermarks
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showSuccess(data.message || 'Watermarks applied successfully!');
                this.showDownloadSection(data.output_file);
            } else {
                this.showError(data.error || 'Failed to apply watermarks.');
            }
        } catch (error) {
            this.showError('Failed to apply watermarks. Please try again.');
            console.error('Watermark error:', error);
        } finally {
            this.hideLoading();
        }
    }
    
    async downloadFile(filename) {
        window.open(`${this.apiBaseUrl}/download/${filename}`, '_blank');
    }
    
    async cleanupFiles() {
        if (!this.currentFileId) return;
        
        try {
            await fetch(`${this.apiBaseUrl}/cleanup`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ file_id: this.currentFileId })
            });
        } catch (error) {
            console.error('Cleanup error:', error);
        }
    }
    
    // UI Helper Methods
    showConfigurationSection() {
        const configSection = document.getElementById('configSection');
        if (configSection) {
            configSection.style.display = 'block';
        }
    }
    
    showDownloadSection(filename) {
        const downloadSection = document.getElementById('downloadSection');
        if (downloadSection) {
            downloadSection.style.display = 'block';
            // Store filename for download
            window.outputFilename = filename;
        }
    }
    
    renderWatermarks() {
        const watermarksList = document.getElementById('watermarksList');
        if (!watermarksList) return;
        
        watermarksList.innerHTML = '';
        
        if (this.watermarks.length === 0) {
            this.addWatermark(); // Add first watermark by default
            return;
        }
        
        this.watermarks.forEach((watermark, index) => {
            const watermarkElement = this.createWatermarkElement(watermark, index);
            watermarksList.appendChild(watermarkElement);
        });
    }
    
    createWatermarkElement(watermark, index) {
        const div = document.createElement('div');
        div.className = 'watermark-item';
        div.innerHTML = `
            <h3>
                <i class="fas fa-tag"></i>
                Watermark ${index + 1}
                ${this.watermarks.length > 1 ? `<button class="remove-btn" onclick="watermarkClient.removeWatermark('${watermark.id}')" title="Remove watermark">
                    <i class="fas fa-times"></i>
                </button>` : ''}
            </h3>
            <div class="config-grid">
                <!-- Text Configuration -->
                <div class="config-group">
                    <label for="watermarkText-${watermark.id}">Watermark Text</label>
                    <input type="text" id="watermarkText-${watermark.id}" 
                           placeholder="Enter watermark text..." 
                           value="${watermark.text}"
                           onchange="watermarkClient.updateWatermarkProperty('${watermark.id}', 'text', this.value)">
                </div>

                <!-- Font Size -->
                <div class="config-group">
                    <label for="fontSize-${watermark.id}">Font Size</label>
                    <input type="range" id="fontSize-${watermark.id}" 
                           min="8" max="72" value="${watermark.font_size}"
                           oninput="watermarkClient.updateWatermarkProperty('${watermark.id}', 'font_size', parseInt(this.value))">
                    <span class="range-value" id="fontSizeValue-${watermark.id}">${watermark.font_size}</span>
                </div>

                <!-- Color -->
                <div class="config-group">
                    <label for="color-${watermark.id}">Color</label>
                    <div class="color-input">
                        <input type="color" id="color-${watermark.id}" 
                               value="${watermark.color}"
                               onchange="watermarkClient.updateWatermarkProperty('${watermark.id}', 'color', this.value)">
                        <input type="text" id="colorText-${watermark.id}" 
                               value="${watermark.color}" placeholder="#000000"
                               onchange="watermarkClient.updateWatermarkProperty('${watermark.id}', 'color', this.value)">
                    </div>
                </div>

                <!-- Opacity -->
                <div class="config-group">
                    <label for="opacity-${watermark.id}">Opacity</label>
                    <input type="range" id="opacity-${watermark.id}" 
                           min="0.1" max="1.0" step="0.1" value="${watermark.opacity}"
                           oninput="watermarkClient.updateWatermarkProperty('${watermark.id}', 'opacity', parseFloat(this.value))">
                    <span class="range-value" id="opacityValue-${watermark.id}">${watermark.opacity}</span>
                </div>

                <!-- Rotation -->
                <div class="config-group">
                    <label for="rotation-${watermark.id}">Rotation (degrees)</label>
                    <input type="range" id="rotation-${watermark.id}" 
                           min="0" max="360" value="${watermark.rotation}"
                           oninput="watermarkClient.updateWatermarkProperty('${watermark.id}', 'rotation', parseInt(this.value))">
                    <span class="range-value" id="rotationValue-${watermark.id}">${watermark.rotation}°</span>
                </div>
            </div>
        `;
        
        return div;
    }
    
    updateWatermarkProperty(watermarkId, property, value) {
        const watermark = this.watermarks.find(w => w.id === watermarkId);
        if (watermark) {
            watermark[property] = value;
            this.updateWatermarkProperties(watermarkId, { [property]: value });
        }
    }
    
    // Drag and Drop Handlers
    handleDragOver(event) {
        event.preventDefault();
        const uploadArea = document.getElementById('uploadArea');
        if (uploadArea) {
            uploadArea.classList.add('dragover');
        }
    }
    
    handleDragLeave(event) {
        event.preventDefault();
        const uploadArea = document.getElementById('uploadArea');
        if (uploadArea) {
            uploadArea.classList.remove('dragover');
        }
    }
    
    handleDrop(event) {
        event.preventDefault();
        const uploadArea = document.getElementById('uploadArea');
        if (uploadArea) {
            uploadArea.classList.remove('dragover');
        }
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (file.type === 'application/pdf') {
                this.uploadFile(file);
            } else {
                this.showError('Please select a PDF file.');
            }
        }
    }
    
    // Utility Methods
    showLoading(message = 'Processing...') {
        const loadingOverlay = document.getElementById('loadingOverlay');
        const loadingText = document.getElementById('loadingText');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'flex';
        }
        if (loadingText) {
            loadingText.textContent = message;
        }
    }
    
    hideLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
    }
    
    showSuccess(message) {
        this.showNotification(message, 'success');
    }
    
    showError(message) {
        this.showNotification(message, 'error');
    }
    
    showNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
            <span>${message}</span>
        `;
        
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#48bb78' : '#f56565'};
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1001;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 500;
            animation: slideIn 0.3s ease-out;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, type === 'success' ? 3000 : 5000);
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            statusElement.textContent = connected ? '🟢 Connected' : '🔴 Disconnected';
            statusElement.className = connected ? 'connected' : 'disconnected';
        }
    }
}

// Initialize the real-time client when the page loads
let watermarkClient;
document.addEventListener('DOMContentLoaded', function() {
    watermarkClient = new WatermarkRealtimeClient();
});

// Global functions for HTML onclick handlers
function addWatermark() {
    if (watermarkClient) {
        watermarkClient.addWatermark();
    }
}

function applyWatermark() {
    if (watermarkClient) {
        watermarkClient.applyWatermarks();
    }
}

function downloadFile() {
    if (watermarkClient && window.outputFilename) {
        watermarkClient.downloadFile(window.outputFilename);
    }
}
