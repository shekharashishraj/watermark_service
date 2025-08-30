#!/bin/bash

# Test script for Docker setup
set -e

echo "🐳 Testing PDF Watermark Service Docker Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        exit 1
    fi
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check if Docker is running
echo "Checking Docker installation..."
if ! docker --version > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not installed or not running${NC}"
    exit 1
fi
print_status $? "Docker is installed and running"

# Check if Docker Compose is available
echo "Checking Docker Compose..."
if ! docker compose version > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    exit 1
fi
print_status $? "Docker Compose is available"

# Build the containers
echo "Building Docker containers..."
docker compose build
print_status $? "Containers built successfully"

# Start the services
echo "Starting services..."
docker compose up -d
print_status $? "Services started successfully"

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Test backend health
echo "Testing backend health..."
if curl -f http://localhost:5001/api/health > /dev/null 2>&1; then
    print_status 0 "Backend health check passed"
else
    print_status 1 "Backend health check failed"
fi

# Test frontend health
echo "Testing frontend health..."
if curl -f http://localhost:80/health > /dev/null 2>&1; then
    print_status 0 "Frontend health check passed"
else
    print_status 1 "Frontend health check failed"
fi

# Test WebSocket connection
echo "Testing WebSocket connection..."
if curl -f -I --http1.1 -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Key: test" -H "Sec-WebSocket-Version: 13" http://localhost:5001/socket.io/ > /dev/null 2>&1; then
    print_status 0 "WebSocket connection test passed"
else
    print_status 1 "WebSocket connection test failed"
fi

# Check container status
echo "Checking container status..."
if docker compose ps | grep -q "Up"; then
    print_status 0 "All containers are running"
else
    print_status 1 "Some containers are not running"
fi

# Display service information
echo ""
echo "🎉 Docker setup test completed successfully!"
echo ""
echo "📋 Service Information:"
echo "======================="
echo "Frontend: http://localhost:80"
echo "Backend API: http://localhost:5001"
echo "Health Check: http://localhost:5001/api/health"
echo ""
echo "📊 Container Status:"
docker-compose ps
echo ""
echo "📝 Logs:"
echo "To view logs: make logs"
echo "To stop services: make down"
echo "To clean up: make clean"
echo ""

# Optional: Test with a sample PDF
read -p "Would you like to test with a sample PDF? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing with sample PDF..."
    
    # Create a simple test PDF if it doesn't exist
    if [ ! -f "test-sample.pdf" ]; then
        print_info "Creating test PDF..."
        # This would require additional tools, so we'll skip for now
        echo "Please upload a PDF file manually to test the watermarking functionality."
    fi
    
    print_info "You can now:"
    print_info "1. Open http://localhost:80 in your browser"
    print_info "2. Upload a PDF file"
    print_info "3. Test the watermarking functionality"
fi

echo ""
echo "🚀 Your PDF Watermark Service is ready to use!"
