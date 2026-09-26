# Type 3: Universal Container Deployment (Docker / OCI / CaaS / Kubernetes)

Deploy this package to ANY container engine, Docker host, or cloud container platform in the world.

## Supported Providers
- Docker / Docker Compose
- Hugging Face Spaces (SDK: docker)
- ModelScope Studio (Docker)
- Google Cloud Run
- AWS ECS / Fargate / App Runner
- Azure Container Apps
- Kubernetes (K8s) Clusters

## Files in this Folder
- `Dockerfile`: Production multi-architecture Python 3.11 container image.
- `.dockerignore`: Excludes build artifacts and temporary files.
- `requirements.txt`: Python dependencies.

## How to Build & Run Locally
```bash
docker build -t tara-ai-core -f deploy/3_container/Dockerfile .
docker run -p 7860:7860 tara-ai-core
```
Visit: `http://localhost:7860`

## How to Deploy to Cloud Containers (e.g. Google Cloud Run, AWS ECS, HF Docker Space)
1. Push this repository to your Git host or container registry.
2. Select Docker runtime.
3. Set container port to `7860` (or leave default).
4. Once deployed, copy the public HTTPS domain into `endpoints.txt`.
