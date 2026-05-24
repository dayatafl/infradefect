# Deployment Guide — InfraDefect

## Architecture

```
┌─────────────────────┐        ┌──────────────────────────────┐
│   Vercel (free)     │        │  HuggingFace Spaces (free)   │
│                     │        │                              │
│   React Frontend    │──────▶ │   FastAPI Backend            │
│   (static build)    │  HTTPS │   SegFormer-B2 (CPU)         │
│                     │        │   ~250ms per image           │
└─────────────────────┘        └──────────────────────────────┘
                                            ▲
                                            │ download at startup
                                ┌───────────┴──────────┐
                                │  HuggingFace Hub      │
                                │  best_model.pth ~100MB│
                                │  (private model repo) │
                                └──────────────────────┘
```

**Why this split:**
- PyTorch + model needs ~2.5GB RAM — only HF Spaces free tier provides this (16GB RAM, 2 vCPU)
- Frontend is a static React build — Vercel is the fastest and simplest for this
- Checkpoint stored on HF Hub (free, supports large files via LFS) and downloaded once at container startup

---

## What You Need Before Starting

- [ ] GitHub account (free)
- [ ] HuggingFace account (free) — https://huggingface.co/join
- [ ] Vercel account (free) — https://vercel.com/signup
- [ ] Training finished: `checkpoints/best_model.pth` exists

---

## STEP 1 — Upload Checkpoint to HuggingFace Hub

The checkpoint (~100MB) is too large for GitHub. Upload it to HF Hub once.

### 1a. Install HF CLI

```bash
pip install huggingface_hub
```

### 1b. Login

```bash
hf auth login
# paste your HF token from https://huggingface.co/settings/tokens
```

### 1c. Run upload script

```bash
cd claude   # project root
python upload_model.py \
  --checkpoint checkpoints/best_model.pth \
  --repo YOUR_HF_USERNAME/infradefect-model
```

Replace `YOUR_HF_USERNAME` with your actual HuggingFace username.

This creates a **private** model repo and uploads `best_model.pth`. Keep note of the repo name — you'll need it in Step 3.

---

## STEP 2 — Push Project to GitHub

### 2a. Create .gitignore entries

Make sure these are in `.gitignore` (they should already be):

```
checkpoints/*.pth
roboflow_data/
uploads/
results/*.png
*.pyc
__pycache__/
node_modules/
frontend/build/
.env
```

### 2b. Create GitHub repo and push

```bash
cd claude   # project root
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/infradefect.git
git push -u origin main
```

---

## STEP 3 — Deploy Backend to HuggingFace Spaces

### 3a. Create a new Space

1. Go to https://huggingface.co/spaces
2. Click **Create new Space**
3. Fill in:
   - **Space name**: `infradefect-api`
   - **License**: MIT
   - **SDK**: Docker
   - **Visibility**: Public (required for the assessment URL)
4. Click **Create Space**

### 3b. Add deployment files to the Space

The Space needs these files. Copy them from `deploy/backend/`:

```
Space repo structure:
├── README.md          ← from deploy/backend/README.md
├── Dockerfile         ← from deploy/backend/Dockerfile
├── app.py             ← from deploy/backend/app.py
├── requirements.txt   ← from deploy/backend/requirements.txt
└── backend/
    └── __init__.py    ← copy from claude/backend/__init__.py
```

Clone the Space repo and copy files:

```bash
# Clone your Space repo
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/infradefect-api
cd infradefect-api

# Copy deployment files
copy ..\claude\deploy\backend\README.md .        # Windows
copy ..\claude\deploy\backend\Dockerfile .
copy ..\claude\deploy\backend\app.py .
copy ..\claude\deploy\backend\requirements.txt .
mkdir backend
copy ..\claude\backend\__init__.py backend\

# On Linux/Mac:
# cp ../claude/deploy/backend/* .
# mkdir backend && cp ../claude/backend/__init__.py backend/

git add .
git commit -m "Deploy InfraDefect API"
git push
```

### 3c. Set Space secrets

In your Space page → **Settings** → **Repository secrets**, add:

| Secret name | Value |
|---|---|
| `HF_MODEL_REPO` | `YOUR_HF_USERNAME/infradefect-model` |
| `HF_TOKEN` | Your HuggingFace token (so Space can download the private model) |

> **Important**: Without `HF_TOKEN`, the Space can't download from a private model repo.

### 3d. Wait for build

The Space will build the Docker image (5-10 minutes first time). Watch the build logs in the Space page.

When it says **Running**, your API is live at:
```
https://YOUR_HF_USERNAME-infradefect-api.hf.space
```

Test it:
```
https://YOUR_HF_USERNAME-infradefect-api.hf.space/api/health
https://YOUR_HF_USERNAME-infradefect-api.hf.space/api/docs
```

**Note on cold starts**: The free tier sleeps after 48h of inactivity. First request after sleep takes 2-3 minutes (downloads model, loads weights). Subsequent requests are fast (~250ms).

---

## STEP 4 — Deploy Frontend to Vercel

### 4a. Update files

**Replace** `frontend/src/hooks/useApi.js` with `deploy/frontend/useApi.js`

**Replace** `frontend/package.json` with `deploy/frontend/package.json` (removes the `proxy` field)

**Add** `frontend/vercel.json` from `deploy/frontend/vercel.json`

### 4b. Set environment variable

Create `frontend/.env.production`:

```
REACT_APP_API_URL=https://YOUR_HF_USERNAME-infradefect-api.hf.space
```

Replace with your actual HuggingFace Spaces URL from Step 3d.

Do NOT commit this file — add to `.gitignore`:
```
frontend/.env.production
```

Commit everything else:

```bash
git add frontend/src/hooks/useApi.js
git add frontend/package.json
git add frontend/vercel.json
git commit -m "Prepare frontend for production deployment"
git push
```

### 4c. Deploy on Vercel

1. Go to https://vercel.com/new
2. Click **Import Git Repository**
3. Select your `infradefect` GitHub repo
4. Configure project:
   - **Framework Preset**: Create React App
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`
5. Under **Environment Variables**, add:
   - `REACT_APP_API_URL` = `https://YOUR_HF_USERNAME-infradefect-api.hf.space`
6. Click **Deploy**

Vercel will build and deploy in ~2 minutes. Your frontend URL:
```
https://infradefect-frontend.vercel.app
```

### 4d. Update backend CORS (optional but good practice)

Once you have your Vercel URL, update the Space secret:

| Secret name | Value |
|---|---|
| `FRONTEND_URL` | `https://infradefect-frontend.vercel.app` |

The backend CORS currently allows `*` so this isn't strictly required, but it's good practice for production.

---

## STEP 5 — Verify Everything Works

Run through this checklist:

```
1. Health check
   GET https://YOUR_HF_USERNAME-infradefect-api.hf.space/api/health
   → should return {"status": "ok", "engine_mode": "model", ...}

2. Swagger UI
   https://YOUR_HF_USERNAME-infradefect-api.hf.space/api/docs
   → upload a test image via the /api/predict endpoint

3. Frontend
   https://infradefect-frontend.vercel.app
   → drop an image, should see overlay and class stats
```

---

## STEP 6 — Update Documentation

Update `README.md` to include the live URLs:

```markdown
## Live Demo

- **Frontend**: https://infradefect-frontend.vercel.app
- **API**: https://YOUR_HF_USERNAME-infradefect-api.hf.space
- **API Docs**: https://YOUR_HF_USERNAME-infradefect-api.hf.space/api/docs
```

---

## Optimisations Made for Deployment

### 1. CPU-only PyTorch

```
# requirements.txt (original)
torch==2.1.2          # 700MB with CUDA

# requirements.txt (deployment)
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.1.2+cpu      # ~170MB — saves 530MB from Docker image
```

### 2. Headless OpenCV

```
# original — requires libGL (display server)
opencv-python==4.8.1.78

# deployment — no display dependencies
opencv-python-headless==4.8.1.78
```

### 3. Stripped requirements

Training-only packages removed from backend requirements:
- `torchvision` — not used in inference
- `albumentations` — augmentation pipeline is training-only
- `scikit-multilearn` — stratified split is training-only
- `scikit-learn` — same
- `timm` — not used in backend
- `tqdm` — not used in backend

Result: Docker image goes from ~6GB to ~2.5GB.

### 4. Checkpoint downloaded once, cached

The checkpoint is not baked into the Docker image. At first startup:
1. `download_checkpoint()` runs
2. Downloads `best_model.pth` from your private HF Hub repo
3. Saves to `checkpoints/` in the container
4. On subsequent starts (if container is warm), file already exists — skips download

### 5. Remove development proxy

The `"proxy": "http://localhost:8000"` in `package.json` is a CRA development feature. It must be removed before production build or Vercel ignores it but it can cause confusion.

---

## Free Tier Limits to Know

| Service | Limit | Impact |
|---|---|---|
| HF Spaces CPU | 16GB RAM, 2 vCPU | Inference ~250ms per image — acceptable |
| HF Spaces sleep | Sleeps after 48h inactivity | First request cold start ~2-3 min |
| HF Hub storage | 10GB per repo (free) | best_model.pth is ~100MB — fine |
| Vercel bandwidth | 100GB/month | Static files only — very unlikely to hit |
| Vercel builds | 100/month | Fine for development |

### Preventing cold starts

To keep the Space warm, you can ping it periodically with a free cron service like cron-job.org:
- URL: `https://YOUR_HF_USERNAME-infradefect-api.hf.space/api/health`
- Interval: every 30 minutes

---

## Troubleshooting

**Space build fails: `libGL.so.1 not found`**
→ Make sure you're using `opencv-python-headless`, not `opencv-python`

**Space starts but `/api/health` returns 503**
→ Check Space logs — model download may have failed. Verify `HF_MODEL_REPO` and `HF_TOKEN` secrets are set correctly.

**Frontend shows "API OFFLINE"**
→ Space may be sleeping. Open the Space URL directly in browser first to wake it, then refresh the frontend.

**Vercel build fails: `Module not found`**
→ Check that `frontend/package.json` has all dependencies. Run `npm install` locally first to verify.

**CORS error in browser**
→ The backend `allow_origins=["*"]` should handle this. If you still see CORS errors, check that `REACT_APP_API_URL` in Vercel env vars doesn't have a trailing slash.

**Inference very slow (>10 seconds)**
→ Normal for first request after cold start. Subsequent requests should be ~250ms on CPU.
