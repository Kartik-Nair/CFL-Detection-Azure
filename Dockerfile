# Use an official Python runtime as a parent image
FROM python:3.12-slim

ENV PORT=8501

RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx \
    && apt-get install -y libglib2.0-dev \
    && ldconfig
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit runs on
EXPOSE 8501

# Run streamlit when the container launches
CMD streamlit run --server.port $PORT app.py
