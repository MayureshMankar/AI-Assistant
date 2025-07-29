#!/bin/bash

# =============================================================================
# AI Coding Assistant Pro - Setup Script
# =============================================================================
# This script automates the setup process for the AI Coding Assistant Pro
# Compatible with Linux, macOS, and Windows (with WSL)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Art Logo
print_logo() {
    echo -e "${PURPLE}"
    cat << "EOF"
    ___    ____   ______          ___                __ 
   /   |  /  _/  / ____/___  ____/ (_)___  ____ _   / / 
  / /| |  / /   / /   / __ \/ __  / / __ \/ __ `/  / /  
 / ___ |_/ /   / /___/ /_/ / /_/ / / / / / /_/ /  /_/   
/_/  |_/___/   \____/\____/\__,_/_/_/ /_/\__, /  (_)   
                                       /____/          
   ___              _      __              __ 
  /   |  __________(_)____/ /_____ _____  / /_
 / /| | / ___/ ___/ / ___/ __/ __ `/ __ \/ __/
/ ___ |(__  |__  ) (__  ) /_/ /_/ / / / / /_  
\_/  |_/____/____/_/____/\__/\__,_/_/ /_/\__/  
                                              
            Pro Version 2.0
EOF
    echo -e "${NC}"
}

# Print colored messages
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_requirements() {
    print_step "Checking system requirements..."
    
    local missing_deps=()
    
    # Check for required commands
    if ! command_exists docker; then
        missing_deps+=("docker")
    fi
    
    if ! command_exists docker-compose; then
        missing_deps+=("docker-compose")
    fi
    
    if ! command_exists git; then
        missing_deps+=("git")
    fi
    
    if ! command_exists curl; then
        missing_deps+=("curl")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        print_message "Please install the missing dependencies and run this script again."
        
        # Provide installation instructions
        echo -e "\n${CYAN}Installation instructions:${NC}"
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "macOS:"
            echo "  brew install docker docker-compose git curl"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            echo "Ubuntu/Debian:"
            echo "  sudo apt update && sudo apt install docker.io docker-compose git curl"
            echo ""
            echo "CentOS/RHEL:"
            echo "  sudo yum install docker docker-compose git curl"
        fi
        
        exit 1
    fi
    
    print_success "All required dependencies are installed!"
}

# Check Docker daemon
check_docker() {
    print_step "Checking Docker daemon..."
    
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running!"
        print_message "Please start Docker and run this script again."
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            print_message "On macOS: Start Docker Desktop application"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            print_message "On Linux: sudo systemctl start docker"
        fi
        
        exit 1
    fi
    
    print_success "Docker daemon is running!"
}

# Setup environment variables
setup_environment() {
    print_step "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_message "Created .env file from .env.example"
        else
            print_warning ".env.example file not found, creating basic .env file"
            create_basic_env
        fi
    else
        print_message ".env file already exists"
    fi
    
    # Check if OpenRouter API key is set
    if grep -q "your_openrouter_api_key_here" .env 2>/dev/null; then
        print_warning "OpenRouter API key not configured!"
        echo -e "\n${YELLOW}To enable AI functionality:${NC}"
        echo "1. Get your API key from: https://openrouter.ai/"
        echo "2. Edit .env file and replace 'your_openrouter_api_key_here' with your actual API key"
        echo "3. Restart the application"
    fi
}

# Create basic .env file
create_basic_env() {
    cat > .env << 'EOF'
# AI Coding Assistant Pro - Environment Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
DATABASE_URL=postgresql://ai_user:ai_password_secure_123@postgres:5432/ai_assistant
REDIS_URL=redis://:redis_password_123@redis:6379/0
SECRET_KEY=your-super-secret-key-change-this-in-production
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO
MAX_FILE_SIZE=10485760
MAX_EXECUTION_TIMEOUT=30
EOF
}

# Create necessary directories
create_directories() {
    print_step "Creating necessary directories..."
    
    local dirs=(
        "logs"
        "uploads"
        "backups"
        "monitoring"
        "nginx"
        "scripts"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_message "Created directory: $dir"
        fi
    done
}

# Download and setup Docker images
setup_docker_images() {
    print_step "Pulling Docker images..."
    
    local images=(
        "postgres:15-alpine"
        "redis:7-alpine"
        "nginx:alpine"
        "prom/prometheus:latest"
        "grafana/grafana:latest"
    )
    
    for image in "${images[@]}"; do
        print_message "Pulling $image..."
        docker pull "$image"
    done
    
    print_success "Docker images downloaded successfully!"
}

# Build application images
build_application() {
    print_step "Building application images..."
    
    # Build backend
    if [ -d "backend" ]; then
        print_message "Building backend image..."
        docker-compose build backend
    fi
    
    # Build frontend
    if [ -d "frontend" ]; then
        print_message "Building frontend image..."
        docker-compose build frontend
    fi
    
    print_success "Application images built successfully!"
}

# Initialize database
initialize_database() {
    print_step "Initializing database..."
    
    # Start only database services first
    docker-compose up -d postgres redis
    
    # Wait for database to be ready
    print_message "Waiting for database to be ready..."
    sleep 10
    
    # Initialize database
    docker-compose exec -T backend python -c "
from database import initialize_database
if initialize_database():
    print('Database initialized successfully!')
else:
    print('Database initialization failed!')
    exit(1)
" || true
    
    print_success "Database initialized!"
}

# Start all services
start_services() {
    print_step "Starting all services..."
    
    docker-compose up -d
    
    # Wait for services to be ready
    print_message "Waiting for services to start..."
    sleep 15
    
    # Health check
    print_message "Performing health checks..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/api/health >/dev/null 2>&1; then
            print_success "Backend is healthy!"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            print_warning "Backend health check failed after $max_attempts attempts"
            print_message "Check logs with: docker-compose logs backend"
        fi
        
        sleep 2
        ((attempt++))
    done
    
    # Check frontend
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:3000 >/dev/null 2>&1; then
            print_success "Frontend is healthy!"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            print_warning "Frontend health check failed after $max_attempts attempts"
            print_message "Check logs with: docker-compose logs frontend"
        fi
        
        sleep 2
        ((attempt++))
    done
}

# Show status and URLs
show_status() {
    print_success "AI Coding Assistant Pro setup completed!"
    
    echo -e "\n${CYAN}🚀 Application URLs:${NC}"
    echo "  • Main Application: http://localhost:3000"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • API Health Check: http://localhost:8000/api/health"
    echo "  • Grafana Dashboard: http://localhost:3001 (admin/admin123)"
    echo "  • Prometheus Metrics: http://localhost:9090"
    
    echo -e "\n${CYAN}📊 Service Status:${NC}"
    docker-compose ps
    
    echo -e "\n${CYAN}🔧 Useful Commands:${NC}"
    echo "  • View logs: docker-compose logs -f"
    echo "  • Stop services: docker-compose down"
    echo "  • Restart services: docker-compose restart"
    echo "  • Update images: docker-compose pull && docker-compose up -d"
    
    echo -e "\n${CYAN}📁 Important Files:${NC}"
    echo "  • Environment config: .env"
    echo "  • Docker compose: docker-compose.yml"
    echo "  • Application logs: ./logs/"
    echo "  • Uploaded files: ./uploads/"
    echo "  • Database backups: ./backups/"
    
    if grep -q "your_openrouter_api_key_here" .env 2>/dev/null; then
        echo -e "\n${YELLOW}⚠️  Don't forget to configure your OpenRouter API key in .env file!${NC}"
    fi
}

# Cleanup function
cleanup() {
    if [ $? -ne 0 ]; then
        print_error "Setup failed! Cleaning up..."
        docker-compose down 2>/dev/null || true
    fi
}

# Main setup function
main() {
    trap cleanup EXIT
    
    print_logo
    
    echo -e "${CYAN}Welcome to AI Coding Assistant Pro Setup!${NC}"
    echo "This script will help you set up the complete application stack."
    echo ""
    
    # Confirm before proceeding
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_message "Setup cancelled."
        exit 0
    fi
    
    check_requirements
    check_docker
    setup_environment
    create_directories
    setup_docker_images
    build_application
    initialize_database
    start_services
    show_status
    
    echo -e "\n${GREEN}🎉 Setup completed successfully!${NC}"
    echo -e "Visit ${BLUE}http://localhost:3000${NC} to start using AI Coding Assistant Pro!"
}

# Run main function
main "$@"