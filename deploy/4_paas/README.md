# Type 4: Universal PaaS & Managed Web Services Deployment

Deploy this package to ANY Platform-as-a-Service (PaaS) or Git-integrated cloud hosting platform in the world.

## Supported Providers
- Render (render.yaml Blueprint included)
- Railway
- Fly.io
- Heroku (Procfile included)
- Koyeb
- PythonAnywhere
- AWS App Runner / Elastic Beanstalk

## Files in this Folder
- `Procfile`: Universal process file for Heroku, Railway, Render, etc.
- `render.yaml`: 1-click Render blueprint specification.
- `requirements.txt`: Python production dependencies.

## Standard Configuration for ANY PaaS
- **Runtime**: Python 3.11
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Health Check Path**: `/health`

## How to Deploy (e.g. Render, Railway)
1. In the platform dashboard, click **New Web Service**.
2. Connect your Git repository (`manjukp6-hue/tara`).
3. Set the Build and Start commands as listed above.
4. Select the Free or standard plan.
5. Click **Deploy**.
6. Once deployed, copy your service HTTPS URL and add it to `endpoints.txt`.
