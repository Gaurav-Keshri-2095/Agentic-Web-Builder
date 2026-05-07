# # the builder :
# # pulling a node.js image to install packages and compile the code
# FROM node:24-alpine AS builder

# WORKDIR /app

# # copying package files and installing dependencies.
# # Copy all matching files starting with package and ending with .json into the current container directory.
# COPY package*.json ./
# RUN npm install

# # Copy the frontend source code and build the static assets.
# COPY . .
# RUN npm run build
# # building a static asset as the frameworks like vite/react/vue do not run directly in production
# # as raw source code, they need to be compiled into static assets (html, css, js) that can
# # be served by a web server in production

# # React/Vite dev server is not meant for production hosting
# # so well will use a lightweight web server to serve the built static files in production runner
# FROM nginx:alpine

# # copying the compiled static files from the builder stage to nginx's server
# # dist means distribution built files, this is the default output directory for Vite builds.
# # it contains the compiled static assets ready for production deployment
# COPY --from=builder /app/dist /usr/share/nginx/html
# # /app/dist /usr/share/nginx/html : this the Nginx's default web root directory
# # files placed here are served to the users when they access the server

# # Exposing port 80
# EXPOSE 80

# # starting nginx server
# CMD ["nginx", "-g", "daemon off;"]
# # daemon is a background running service/process which run continuously in the background waiting for requrests
# # so nginx is a daemon process and it says .. i will go run in background now
# # but inside a docker container, we want the main process to run in foreground so that the
# # container stays alive and does not exit immediately after starting nginx
# # so we use the -g flag to pass a global configuration to nginx to tell it to run in foreground
# # instead of daemon mode and do daemon off


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
ENV VITE_API_BASE=http://146.190.219.20:8000

# Build the application
RUN npm run build

# Expose port 8080 for the frontend
EXPOSE 8080

# Disable interactive prompts in Wrangler
ENV CI=true

# Start the Cloudflare simulator, binding to 0.0.0.0 to allow Docker traffic
CMD ["wrangler", "dev", "--ip", "0.0.0.0", "--port", "8080"]