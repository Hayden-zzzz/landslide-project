# Use the official lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port SnapDeploy expects (8000)
EXPOSE 8000

# Run with Gunicorn (production WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]