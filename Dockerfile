# ========================
# Stage 1: Frontend Build
# ========================
FROM node:22 AS frontend-build

# Set working directory for frontend
WORKDIR /app/frontend

# Copy package files for caching
COPY frontend/package*.json ./

# Install dependencies
RUN npm install --silent

# Copy all frontend source and build
COPY frontend/ ./
RUN npm run build


# ========================
# Stage 2: Backend (Python + GPU)
# ========================
# CUDA 12.8 + cuDNN 8.9 + Python 3.10
FROM nvidia/cuda:12.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory for backend
WORKDIR /app/backend

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git curl && \
    rm -rf /var/lib/apt/lists/*

# Ensure python3 points to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy backend requirements and install Python dependencies
COPY backend/requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./

# Copy built frontend from Stage 1
COPY --from=frontend-build /app/frontend/build ./frontend_build

# Expose Flask port
EXPOSE 5000

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Run Flask app
CMD ["python3", "-m", "app"]
