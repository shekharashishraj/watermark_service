# Docker Setup for PDF Watermark Service

This document provides comprehensive instructions for running the PDF Watermark Service using Docker and Docker Compose.

## 🐳 Quick Start

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 2GB of available RAM
- 5GB of available disk space

### Basic Usage

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd watermark_service
   ```

2. **Start the application:**
   ```bash
   # Development mode
   make dev
   
   # Or production mode
   make prod
   ```

3. **Access the application:**
   - Frontend: http://localhost:80 (production) or http://localhost:3000 (development)
   - Backend API: http://localhost:5001
   - Health Check: http://localhost:5001/api/health

## 📁 Docker Files Overview

### Core Files
- `Dockerfile.backend` - Flask backend container
- `Dockerfile.frontend` - React frontend container (production)
- `Dockerfile.frontend.dev` - React frontend container (development)
- `docker-compose.yml` - Base configuration
- `docker-compose.dev.yml` - Development overrides
- `docker-compose.prod.yml` - Production configuration
- `nginx.conf` - Nginx reverse proxy configuration

### Configuration Files
- `.dockerignore` - Excludes unnecessary files from build context
- `Makefile` - Convenient commands for Docker operations
- `docker-compose.override.yml` - Local development overrides

## 🚀 Deployment Options

### 1. Development Environment

**Features:**
- Hot reloading for both frontend and backend
- Volume mounts for live code changes
- Development-friendly logging
- Exposed ports for debugging

**Commands:**
```bash
# Start development environment
make dev

# Build development containers
make dev-build

# View development logs
make dev-logs

# Stop development environment
make dev-down
```

**Access:**
- Frontend: http://localhost:3000
- Backend: http://localhost:5001
- WebSocket: ws://localhost:5001

### 2. Production Environment

**Features:**
- Optimized builds with multi-stage Dockerfiles
- Nginx reverse proxy with SSL support
- Redis for session management
- Monitoring with Prometheus and Grafana
- Resource limits and health checks

**Commands:**
```bash
# Start production environment
make prod

# Build production containers
make prod-build

# View production logs
make prod-logs

# Stop production environment
make prod-down
```

**Access:**
- Frontend: http://localhost:80
- Backend API: http://localhost:5001
- Monitoring: http://localhost:9090 (Prometheus), http://localhost:3001 (Grafana)

## 🔧 Configuration

### Environment Variables

**Backend Environment Variables:**
```bash
FLASK_ENV=production          # Environment mode
PYTHONUNBUFFERED=1           # Python output buffering
REDIS_URL=redis://redis:6379/0  # Redis connection (production)
```

**Frontend Environment Variables:**
```bash
REACT_APP_API_URL=http://localhost:5001  # Backend API URL
CHOKIDAR_USEPOLLING=true                 # File watching (development)
```

### Volume Mounts

**Development:**
- `./backend:/app` - Live code changes for backend
- `./frontend:/app` - Live code changes for frontend
- `/app/node_modules` - Preserve node_modules in container

**Production:**
- `backend_uploads:/app/uploads` - Persistent file uploads
- `backend_outputs:/app/outputs` - Persistent processed files
- `backend_logs:/app/logs` - Persistent application logs
- `redis_data:/data` - Persistent Redis data

### Network Configuration

The application uses a custom bridge network (`watermark-network`) with subnet `172.20.0.0/16` for internal communication between services.

## 📊 Monitoring and Health Checks

### Health Checks

All services include health checks:

**Backend:**
```bash
curl -f http://localhost:5001/api/health
```

**Frontend:**
```bash
curl -f http://localhost:80/health
```

**Redis:**
```bash
docker-compose exec redis redis-cli ping
```

### Monitoring Stack (Production)

**Prometheus:**
- URL: http://localhost:9090
- Metrics collection for backend and frontend
- Custom metrics for watermark operations

**Grafana:**
- URL: http://localhost:3001
- Username: admin
- Password: admin
- Pre-configured dashboards for application metrics

## 🛠️ Common Operations

### Using Makefile Commands

```bash
# Show all available commands
make help

# Build all containers
make build

# Start all services
make up

# Stop all services
make down

# View logs
make logs

# Clean up everything
make clean

# Check service health
make health

# Monitor resource usage
make monitor

# Create backup
make backup

# Restore from backup
make restore
```

### Manual Docker Commands

```bash
# Build specific service
docker-compose build backend

# Start specific service
docker-compose up -d backend

# View logs for specific service
docker-compose logs -f backend

# Execute command in running container
docker-compose exec backend python -c "print('Hello from container')"

# Access Redis CLI
docker-compose exec redis redis-cli

# Scale services
docker-compose up -d --scale backend=3
```

## 🔒 Security Considerations

### Production Security

1. **SSL/TLS Configuration:**
   - Place SSL certificates in `./ssl/` directory
   - Update nginx configuration for HTTPS
   - Redirect HTTP to HTTPS

2. **Environment Variables:**
   - Use `.env` files for sensitive data
   - Never commit secrets to version control
   - Use Docker secrets for production deployments

3. **Network Security:**
   - Use custom networks for service isolation
   - Implement proper firewall rules
   - Consider using Docker Swarm for production

4. **Resource Limits:**
   - Set memory and CPU limits
   - Monitor resource usage
   - Implement rate limiting

### Example `.env` file:
```bash
# Backend
FLASK_SECRET_KEY=your-secret-key-here
REDIS_PASSWORD=your-redis-password

# Frontend
REACT_APP_API_URL=https://api.yourdomain.com

# Database
POSTGRES_PASSWORD=your-db-password
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Conflicts:**
   ```bash
   # Check what's using the port
   lsof -i :5001
   
   # Kill process using port
   sudo kill -9 <PID>
   ```

2. **Container Won't Start:**
   ```bash
   # Check container logs
   docker-compose logs backend
   
   # Check container status
   docker-compose ps
   ```

3. **Build Failures:**
   ```bash
   # Clean build cache
   docker-compose build --no-cache
   
   # Clean everything
   make clean
   ```

4. **Memory Issues:**
   ```bash
   # Check Docker resource usage
   docker stats
   
   # Increase Docker memory limit in Docker Desktop
   ```

### Debugging Commands

```bash
# Enter running container
docker-compose exec backend bash

# View real-time logs
docker-compose logs -f --tail=100

# Check network connectivity
docker-compose exec backend ping frontend

# Monitor resource usage
docker stats --no-stream

# Check volume mounts
docker volume ls
```

## 📈 Performance Optimization

### Production Optimizations

1. **Multi-stage Builds:**
   - Frontend uses multi-stage build to reduce image size
   - Backend optimized with slim Python image

2. **Caching:**
   - Nginx configured with aggressive caching
   - Static assets cached for 1 year
   - API responses cached appropriately

3. **Load Balancing:**
   - Multiple backend replicas in production
   - Nginx load balancing configuration
   - Health checks for automatic failover

4. **Resource Management:**
   - Memory and CPU limits set
   - Redis memory limits configured
   - Automatic cleanup of temporary files

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build and push Docker images
        run: |
          docker-compose -f docker-compose.prod.yml build
          docker-compose -f docker-compose.prod.yml push
      
      - name: Deploy to server
        run: |
          docker-compose -f docker-compose.prod.yml pull
          docker-compose -f docker-compose.prod.yml up -d
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Redis Documentation](https://redis.io/documentation)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

---

**Note:** This Docker setup is designed for both development and production use. The development configuration provides hot reloading and debugging capabilities, while the production configuration focuses on performance, security, and monitoring.

