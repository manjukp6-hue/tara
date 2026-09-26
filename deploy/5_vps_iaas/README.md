# Type 5: Universal VPS & Bare Metal Server Deployment (IaaS)

Deploy this package to ANY dedicated Linux server, Virtual Private Server (VPS), or cloud compute instance in the world.

## Supported Providers
- AWS EC2
- DigitalOcean Droplets
- Linode / Akamai
- Hetzner Cloud
- Google Compute Engine (GCE)
- Oracle Cloud Free Tier Ampere ARM/x86 Instances
- Any Ubuntu / Debian / RHEL / AlmaLinux server

## Files in this Folder
- `install.sh`: Automated 1-step installer script.
- `tara.service`: Systemd service unit (24/7 background process with automatic crash recovery & reboot persistence).
- `nginx_tara.conf`: Production Nginx reverse-proxy configuration with SSL support.

## How to Deploy on a Clean Linux Server
1. Clone the repository on your server:
   ```bash
   git clone https://github.com/manjukp6-hue/tara.git /opt/tara
   cd /opt/tara
   ```
2. Run the automated installer:
   ```bash
   sudo bash deploy/5_vps_iaas/install.sh
   ```
3. (Optional) Set up Nginx & Free SSL with Certbot:
   ```bash
   sudo cp deploy/5_vps_iaas/nginx_tara.conf /etc/nginx/sites-available/tara
   sudo ln -s /etc/nginx/sites-available/tara /etc/nginx/sites-enabled/
   sudo certbot --nginx -d your-domain.com
   sudo systemctl restart nginx
   ```
4. Copy your server domain or public IP HTTPS URL into `endpoints.txt`.
