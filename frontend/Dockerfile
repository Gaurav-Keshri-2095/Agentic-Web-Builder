
# Start from a lightweight Node.js environment
FROM node:24-slim

# Set the working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Install Cloudflare's Wrangler CLI globally
RUN npm install -g wrangler

# Copy the rest of the frontend code
COPY . .

# INJECT THE BUILD VARIABLE DIRECTLY
ENV VITE_API_BASE=https://gauravkeshri.dev/api

# Build the application
RUN npm run build

# Expose port 8080 for the frontend
EXPOSE 8080

# Disable interactive prompts in Wrangler
ENV CI=true

# Start the Cloudflare simulator, binding to 0.0.0.0 to allow Docker traffic
CMD ["wrangler", "dev", "--ip", "0.0.0.0", "--port", "8080"]