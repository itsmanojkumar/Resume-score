# Use Debian-based Node to fix Vite crypto issue
FROM node:20-bullseye-slim

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy all frontend code
COPY . .

# Build frontend
RUN npm run build

# Serve the built app
EXPOSE 5173
CMD ["npx", "serve", "-s", "dist", "-l", "5173"]
