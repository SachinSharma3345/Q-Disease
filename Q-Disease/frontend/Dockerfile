# Q-Disease frontend — React/Vite
# Two-stage build: Node build → Nginx serve

# ── Stage 1: build ───────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first (layer-cached unless package.json changes)
COPY package.json package-lock.json* ./
RUN npm ci --silent

# Copy source and build
COPY . .

# VITE_API_URL is the backend URL the BROWSER contacts at runtime.
# In docker-compose the default is http://localhost:8000 (Nginx proxy).
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# ── Stage 2: serve ───────────────────────────────────────────────
FROM nginx:1.25-alpine

# Copy the built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx config: serve static files and proxy /api → backend
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
