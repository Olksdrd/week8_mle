### Dockerfile for inference
# Get Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy data (should be generated first!)
COPY data/ /app/data

# Copy model (should be trained first!)
COPY model/ /app/model

# Copy inference script
COPY inference /app/inference

# Install packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Run the inference
RUN python3 inference/prediction.py