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

# Copy utility function(s)
COPY utils.py /app

# Copy unit test(s)
COPY test.py /app

# Install packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Run unit tests and inference
RUN python3 test.py
RUN python3 inference/prediction.py 