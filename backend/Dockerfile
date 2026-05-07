# starting from a lightweight base image
# slim means a smaller, lightweight version of the Python Docker image
FROM python:3.10-slim

# setting the working directory in the container
# Inside the container, create/use a directory named /app and execute future commands there
WORKDIR /app

# copying the requirements file first. this leverages Docker's caching
# if no changes are made to the requirements file, docker skips this step in future builds, making the build process faster
COPY requirements.txt .

# installing dependencies
RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir flag is used to prevent pip from caching the installed packages, which helps reduce the image size

# copying the rest of code
COPY . .

# exposing the port
EXPOSE 8000

CMD [ "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000" ]