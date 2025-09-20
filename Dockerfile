# Use a slim version of Python to keep the image small
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by OpenCV and MediaPipe
# This is crucial for solving the `libGL.so.1` and other related errors.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Copy the rest of your application code
COPY . .

# Set the entrypoint to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=true"]
