# Type 1: Universal Static Web Deployment

Deploy this package to ANY static website host, CDN, or Jamstack provider in the world.

## Supported Providers
- Cloudflare Pages
- GitHub Pages
- Hugging Face Spaces (SDK: static)
- Vercel (Static)
- Netlify
- AWS S3 + CloudFront
- Firebase Hosting
- Any web server serving HTML files

## Files in this Folder
- `index.html`: The canonical TARA frontend (minimal centered UI, voice recognition, concurrent probing engine).
- `endpoints.txt`: The dynamic provider registry.

## How to Deploy
1. Upload both `index.html` and `endpoints.txt` to your static host root directory.
2. In Hugging Face Spaces, ensure `sdk: static` in `README.md`.
3. In GitHub Pages, enable Pages on the branch serving this folder.
4. Your static site is now live!
5. Whenever you add or remove backend providers, just update `endpoints.txt`.
