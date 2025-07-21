# Use official Python 3.11 image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies for dlib, opencv, etc.
RUN apt-get update && \
    apt-get install -y build-essential cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libx11-dev libopenblas-dev liblapack-dev libatlas-base-dev gfortran libhdf5-dev python3-dev libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY . .

# Expose ports for FastAPI (8000) and Flask (8001)
EXPOSE 8000 8001

# Default command (can be overridden)
CMD ["bash"] 