# ==============================================================================
# TARA Universal Deployment Suite (6 Infrastructure Types + 4 Strategies)
# ==============================================================================

This directory provides universal, vendor-neutral deployment packages for TARA.
No matter what cloud, host, or server you use in the future, it fits into one
of the 6 fundamental infrastructure types below.

## The 6 Universal Infrastructure Types

```text
deploy/
├── 1_static/         → Static Web, CDN, Pages, Jamstack (GitHub Pages, Cloudflare Pages, S3, Netlify, HF Static)
├── 2_serverless/     → Edge Functions & Workers (Cloudflare Workers, Vercel Edge, AWS Lambda, Deno Deploy)
├── 3_container/      → Docker, OCI, Containers, Kubernetes (HF Docker Spaces, ModelScope, Cloud Run, AWS ECS)
├── 4_paas/           → Platform-as-a-Service (Render, Railway, Fly.io, Heroku, Koyeb)
├── 5_vps_iaas/       → Virtual Machines & Bare Metal (AWS EC2, DigitalOcean, Linode, Hetzner, Ubuntu VPS)
└── 6_local_device/   → On-Premise, Local PC, Workstation, Edge IoT (Windows, Linux, macOS, Raspberry Pi)
```

---

## Universal 3-Step Deployment Workflow

For ANY site or cloud provider in the world:

1. **Pick the matching type** from `deploy/1_*` to `deploy/6_*`.
2. **Deploy/Upload** following the 1-page `README.md` in that folder.
3. **Add the resulting HTTPS URL** into `endpoints.txt`:
   ```text
   https://your-new-service-url.com
   ```

**That is all.** 
All TARA clients automatically discover the new provider concurrently, test latency, and failover seamlessly. Zero code modification required.

---

## The 4 Release Strategies

When updating a deployed service, choose one of these 4 strategies:

1. **Rolling Update (Zero Downtime)**:
   Update container/node instances one by one. Traffic keeps flowing to healthy nodes.
2. **Blue-Green Deployment (Instant Switch)**:
   Deploy the new version on a parallel environment ("Green"). Verify health. Switch the entry in `endpoints.txt` to Green.
3. **Canary Deployment (Gradual Verification)**:
   Add the new endpoint to `endpoints.txt`. TARA's concurrent probing evaluates latency and health across clients before full traffic migration.
4. **Recreate (Clean Restart)**:
   Stop the old process and start the new version. Best for local development and single-node setups.
