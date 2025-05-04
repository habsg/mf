# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# Copy src, templates, static directories
COPY src/ ./src
COPY templates/ ./templates
COPY static/ ./static

# Make port 8000 available to the world outside this container
# Gunicorn will run on this port
EXPOSE 8000

# Define environment variable for Flask (optional, can be set in Render UI too)
ENV FLASK_ENV=production

# Run main.py when the container launches using Gunicorn
# Bind to 0.0.0.0 to allow external connections
CMD gunicorn --bind 0.0.0.0:$PORT src.main:app

