# Dockerfile for training
# Start from a base Python image
FROM python:3.11

# Changing the path to the working directory in the container
WORKDIR /app

# Copy data folder to Docker
COPY data/ /app/data

# Copy the code itself
COPY training /app/training

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python3 training/train.py