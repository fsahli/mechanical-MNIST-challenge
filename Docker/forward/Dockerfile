# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for scientific computing)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file (create this if you have specific dependencies)
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Install essential Python packages for the model
RUN pip install --no-cache-dir \
    numpy>=1.21.0 \
    scipy>=1.7.0 \
    matplotlib>=3.4.0

# Copy the model script
COPY run_model_forward_docker.py .

# Make the script executable
RUN chmod +x run_model_forward_docker.py

# Set the entrypoint to python
ENTRYPOINT ["python", "run_model_forward_docker.py"]

# Default command (can be overridden)
CMD ["--help"]
