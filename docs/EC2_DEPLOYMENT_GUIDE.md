# AWS EC2 Provisioning & Deployment Guide

This guide covers setting up an AWS EC2 virtual machine, configuring security and IAM roles, running the containerized FastAPI prediction service with automatic restart policies, and connecting the Streamlit frontend.

---

## 1. AWS EC2 Instance Provisioning

### Recommended Specs:
- **AMI**: Ubuntu Server 22.04 LTS (HVM), SSD Volume Type
- **Instance Type**: `t3.medium` (or `t3.small` / `t2.medium` with $\ge 2$ GB RAM)
- **Storage**: 20 GiB gp3 root volume

### Security Group Configuration (Inbound Rules):
| Type | Protocol | Port Range | Source | Description |
| :--- | :--- | :--- | :--- | :--- |
| **SSH** | TCP | 22 | Your IP (`My IP`) | Administrative SSH access |
| **Custom TCP** | TCP | 8000 | `0.0.0.0/0` | FastAPI Prediction Service endpoint |
| **Custom TCP** | TCP | 8501 | `0.0.0.0/0` | Streamlit Dashboard (optional if hosted on EC2) |

---

## 2. Deploying via Script or Docker Compose

SSH into your EC2 instance:
```bash
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### Option A: Using the Automated Deployment Script (`scripts/deploy_ec2.sh`)
Clone the repository or copy the deployment script:
```bash
git clone https://github.com/Devershika/E-Commerce-MLOPs.git
cd E-Commerce-MLOPs

# Make script executable and run
chmod +x scripts/deploy_ec2.sh
./scripts/deploy_ec2.sh
```

### Option B: Using Production Docker Compose (`docker-compose.prod.yml`)
```bash
# Pull the latest image
docker compose -f docker-compose.prod.yml pull

# Start service in the background with auto-restart enabled
docker compose -f docker-compose.prod.yml up -d
```

---

## 3. High Availability & Auto-Restart Verification

The container is configured with `--restart always`. To verify resilience:
1. **Reboot the EC2 instance**:
   ```bash
   sudo reboot
   ```
2. Wait 30 seconds, then check if Docker automatically restored the container:
   ```bash
   docker ps
   curl http://localhost:8000/health
   ```
   *Expected Response:*
   ```json
   {"status":"healthy","model":"Holt-Winters (Fine-Tuned)","version":"1.0.0","model_loaded":true}
   ```

---

## 4. Connecting the Streamlit Frontend

In the Streamlit dashboard (`apps/app.py`):
1. In the sidebar under **Configuration**, enter your EC2 endpoint:
   ```
   http://<EC2-PUBLIC-IP>:8000
   ```
2. The indicator badge will turn green: `● Backend API Connected`.
3. Selecting any SKU and clicking predict will now serve real-time predictions directly from your cloud-hosted model service!
