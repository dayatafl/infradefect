# InfraDefect — React Frontend

Modern, production-ready React frontend for infrastructure defect segmentation. Handles image upload, real-time inference, mask visualization, and defect statistics.

## 🚀 Setup & Run

### Prerequisites

- Node.js 16+ and npm

### Development

From the `frontend/` directory:

```bash
npm install
npm start
```

Opens at **http://localhost:3000**. The API must be running at **http://localhost:8000**.

### Run Both API & Frontend Together (from project root)

**Terminal 1 — API:**
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm install
npm start
```

Both should now be accessible:
- Frontend: http://localhost:3000
- API: http://localhost:8000/api/docs (Swagger UI)

## 🏗️ Build for Production

Generate optimized production bundle:

```bash
npm run build
```

Output in `build/`. Can be served as static files by the FastAPI backend.

### Option 1: Serve via FastAPI

Add to `backend/app.py` after other route definitions:

```python
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="../frontend/build", html=True), name="static")
```

Then all requests to `/` serve the React app, `/api/*` routes to the backend.

### Option 2: Serve via Nginx

Use Nginx reverse proxy:

```nginx
server {
    listen 80;
    server_name _;
    
    location /api {
        proxy_pass http://localhost:8000;
    }
    
    location / {
        root /var/www/infra-defect/frontend/build;
        try_files $uri /index.html;
    }
}
```

## 🎨 Features

- **Image Upload** — Drag-and-drop or file picker
- **Real-time Inference** — Sends to `/api/predict` endpoint
- **Mask Overlay** — Alpha-blended defect masks on original image
- **Per-Class Statistics** — Pixel count, percentage, severity level
- **Color-Coded Legend** — Visual mapping of defect classes to colors
- **Demo Mode Support** — Works with synthetic results if no model checkpoint

## 📁 Structure

```
frontend/
├── public/
│   ├── index.html      # HTML entry point
│   └── favicon.ico     # App icon
├── src/
│   ├── App.jsx         # Main React component
│   ├── App.css         # Styling
│   ├── index.js        # ReactDOM mount
│   └── ...             # Additional components
├── package.json        # Node.js dependencies
├── build/              # Production build output (after npm run build)
└── README.md           # This file
```

## 🔌 API Integration

Frontend communicates with backend via:

- **POST `/api/predict`** — Upload image, receive segmentation result
- **GET `/api/health`** — Check API status
- **GET `/api/model-info`** — Get model metadata
- **GET `/api/classes`** — Get defect class schema

Response format (from `/api/predict`):

```json
{
  "job_id": "a1b2c3d4",
  "overlay_b64": "iVBORw0KGgo...",
  "class_stats": {
    "crack": {
      "pixel_count": 15432,
      "percentage": 0.87,
      "color_hex": "#ff4646",
      "severity": "low"
    }
  },
  "inference_ms": 32.5,
  "engine_mode": "model",
  "image_size": [1920, 1080],
  "defects_found": ["crack"]
}
```

## 🌐 Proxy Configuration

The `package.json` includes:

```json
"proxy": "http://localhost:8000"
```

This routes all API calls during development. In production, update this or use environment variables.

## 🔧 Environment Variables

Create a `.env.local` file to configure:

```bash
REACT_APP_API_URL=http://localhost:8000
REACT_APP_API_TIMEOUT=30000  # ms
```

## 🚀 Deployment

### Docker

```dockerfile
FROM node:18 as builder
WORKDIR /app
COPY frontend .
RUN npm install && npm run build

FROM node:18
WORKDIR /app
COPY --from=builder /app/build ./public
# Serve with a simple Node server or Nginx
```

### Cloud Platforms

- **Vercel**: Push to GitHub, connect repo, auto-deploy on main branch
- **Netlify**: Similar to Vercel
- **AWS S3 + CloudFront**: Build locally, upload to S3, CDN in front

## 🔗 Links

- Backend API: [../README.md](../README.md)
- Architecture: [../ARCHITECTURE.md](../ARCHITECTURE.md)
- Model Training: [../model_training/README.md](../model_training/README.md)
