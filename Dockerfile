# Use Python 3.11.9 base image
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for faster caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your project
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Command to run your app
CMD ["streamlit", "run", "fitnessapp.py", "--server.port=8501", "--server.address=0.0.0.0"]
