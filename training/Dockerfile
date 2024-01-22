### Dockerfile for model training
# Get Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy data (should be generated first!)
COPY data/ /app/data

# Copy training script
COPY training /app/training

# Copy utility function(s)
COPY utils.py /app

# Install packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Run the training
RUN python3 training/train.py