# Use a multi-arch base image
FROM python:3.9-alpine

# Add metadata to the image
LABEL org.opencontainers.image.source=https://github.com/jtha/llm_infra
LABEL org.opencontainers.image.description='Multi-Arch API Gateway Service for LLM Infrastructure'

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip, install dependencies, and remove cache in a single layer
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy the rest of the application code
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV PYTHONPATH=/app

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]