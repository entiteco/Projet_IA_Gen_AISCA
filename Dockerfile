# ==========================================
# Base Image
# ==========================================
# Use the official Python 3.10 slim image to minimize container size
FROM python:3.10-slim

# ==========================================
# Environment Variables
# ==========================================
# Prevents Python from writing .pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures stdout and stderr are flushed immediately (useful for logs)
ENV PYTHONUNBUFFERED=1

# ==========================================
# Working Directory
# ==========================================
# Set the working directory inside the container
WORKDIR /app

# ==========================================
# System Dependencies
# ==========================================
# Install essential build tools (gcc, make) required for some Python packages
# Clean up apt cache afterwards to reduce image size
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# Python Dependencies
# ==========================================
# Copy requirements file first to leverage Docker cache layering
COPY requirements.txt .
# Install Python dependencies without cache to save space
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# Application Code
# ==========================================
# Copy the rest of the application source code into the container
COPY . .

# ==========================================
# Network & Runtime
# ==========================================
# Expose the default Streamlit port
EXPOSE 8501

# Command to run the application on container start
# --server.address=0.0.0.0 is crucial for Docker networking
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]