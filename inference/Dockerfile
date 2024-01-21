# Dockerfile for training
# Start from a base Python image
FROM python:3.11


WORKDIR /app

# Copy data folder to Docker
COPY data/ /app/data

# Copy models folder to Docker
COPY model/ /app/model

# Copy the code itself
COPY inference /app/inference

# Copy usefull scripts and configs

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python3 inference/prediction.py