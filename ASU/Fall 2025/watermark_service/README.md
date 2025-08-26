# PDF Watermark Service

A professional PDF watermarking service with real-time drag-and-drop positioning, built with React frontend and Flask backend.

## 🚀 Features

- **Real-time Drag & Drop Positioning**: Visually position watermarks anywhere on your PDF
- **WebSocket Communication**: Instant updates and real-time collaboration
- **Multiple Watermarks**: Add and manage multiple watermarks simultaneously
- **Live Coordinate Tracking**: See exact PDF coordinates as you drag
- **Modern React UI**: Beautiful, responsive interface with animations
- **Professional PDF Processing**: High-quality watermark application
- **Session Management**: Support for multiple users and sessions

## 🏗️ Architecture

### Frontend (React)
- **Port**: 3000
- **Technology**: React 18, Socket.IO Client, Styled Components
- **Features**: Real-time UI, drag-and-drop, animations

### Backend (Flask)
- **Port**: 5001
- **Technology**: Flask, Socket.IO, PyPDF2, ReportLab
- **Features**: PDF processing, WebSocket server, session management

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm

### Quick Start
1. Clone the repository:
```bash
git clone <repository-url>
cd watermark_service
```

2. Run the startup script:
```bash
python3 start_servers.py
```

This will:
- Install all dependencies (both Python and Node.js)
- Start the backend server on port 5001
- Start the React frontend on port 3000
- Open the application in your browser

### Manual Installation

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
```

#### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## 🎯 Usage

1. **Upload PDF**: Drag and drop or click to upload a PDF file
2. **Configure Watermarks**: 
   - Add watermarks with custom text, size, color, and opacity
   - Use the visual positioning tool for precise placement
3. **Real-time Positioning**: Drag watermarks to exact positions on the PDF
4. **Apply & Download**: Process your PDF and download the watermarked version

## 🔧 API Endpoints

### REST API (Port 5001)
- `POST /api/upload` - Upload PDF file
- `POST /api/watermark` - Apply watermarks to PDF
- `GET /api/download/<filename>` - Download watermarked PDF
- `POST /api/cleanup` - Clean up temporary files
- `GET /api/health` - Health check

### WebSocket Events
- `connect` - Client connection
- `join_session` - Join watermarking session
- `add_watermark` - Add new watermark
- `remove_watermark` - Remove watermark
- `update_watermark_position` - Update watermark position
- `update_watermark_properties` - Update watermark properties

## 🎨 Frontend Components

- **FileUpload**: Drag-and-drop PDF upload with validation
- **WatermarkConfig**: Configure watermark properties (text, size, color, opacity)
- **VisualPositioning**: Real-time drag-and-drop positioning interface
- **DownloadSection**: Download processed PDFs

## 🔒 Security Features

- File type validation (PDF only)
- Secure file handling with unique IDs
- CORS configuration for frontend-backend communication
- Session-based watermark management

## 🛠️ Development

### Project Structure
```
watermark_service/
├── backend/
│   ├── app.py              # Main Flask application
│   ├── watermark_service.py # PDF processing logic
│   ├── requirements.txt    # Python dependencies
│   ├── uploads/           # Temporary uploaded files
│   └── outputs/           # Processed PDFs
├── frontend/
│   ├── src/
│   │   ├── App.js         # Main React component
│   │   └── components/    # React components
│   ├── public/
│   ├── package.json       # Node.js dependencies
│   └── README.md
└── start_servers.py       # Startup script
```

### Adding New Features

#### Backend
1. Add new routes in `backend/app.py`
2. Extend `watermark_service.py` for new functionality
3. Add WebSocket events for real-time features

#### Frontend
1. Create new components in `frontend/src/components/`
2. Update `App.js` to include new features
3. Add WebSocket event handlers for real-time updates

## 🚀 Deployment

### Production Setup
1. Build the React frontend:
```bash
cd frontend
npm run build
```

2. Configure production environment variables
3. Use a production WSGI server (Gunicorn, uWSGI)
4. Set up reverse proxy (Nginx)

### Docker Deployment
```dockerfile
# Backend Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
CMD ["python", "app.py"]
```

## 🐛 Troubleshooting

### Common Issues

1. **Port 5000/5001 in use**:
   ```bash
   lsof -ti:5000 | xargs kill -9
   lsof -ti:5001 | xargs kill -9
   ```

2. **Node.js not found**:
   - Install Node.js from https://nodejs.org/

3. **Python dependencies error**:
   ```bash
   pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```

4. **WebSocket connection issues**:
   - Check if backend is running on port 5001
   - Verify CORS configuration

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation

---

**Built with ❤️ using React, Flask, and WebSockets**
# watermark_service
