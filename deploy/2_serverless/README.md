# Type 2: Universal Serverless & Edge Compute Deployment

Deploy this package to ANY serverless edge runtime or edge function provider in the world.

## Supported Providers
- Cloudflare Workers
- Vercel Edge Middleware / Functions
- AWS Lambda@Edge
- Deno Deploy
- Supabase Edge Functions
- Fastly Compute@Edge

## Files in this Folder
- `worker.js`: Universal Edge Handler (serves the UI, handles `/health`, `/endpoints.txt`, and proxies chat requests with zero mock AI).
- `wrangler.toml`: Standard configuration for Cloudflare Workers.
- `endpoints.txt`: The dynamic provider registry.

## How to Deploy (e.g. Cloudflare Workers)
### Option A: Web Dashboard (Zero Install)
1. Go to [dash.cloudflare.com](https://dash.cloudflare.com) -> **Workers & Pages** -> **Create Worker**.
2. Click **Quick Edit**.
3. Paste the contents of `worker.js`.
4. Click **Save and Deploy**.
5. Copy your worker HTTPS URL and add it to `endpoints.txt`.

### Option B: Terminal CLI
```bash
npx wrangler login
npx wrangler deploy
```
