# IoT Cybersecurity Analyst Chatbot
### CISC 886 — Cloud Computing | Queen's University
**NetID:** 20596360 | **Region:** `us-east-1` | **Live URL:** `http://3.92.74.249:8501`

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Repository Structure](#repository-structure)
4. [Phase 1 — VPC & Networking](#phase-1--vpc--networking)
5. [Phase 2 — S3 Buckets](#phase-2--s3-buckets)
6. [Phase 3 — EMR Data Preprocessing](#phase-3--emr-data-preprocessing)
7. [Phase 4 — Model Fine-Tuning (Google Colab)](#phase-4--model-fine-tuning-google-colab)
8. [Phase 5 — EC2 Deployment](#phase-5--ec2-deployment)
9. [Phase 6 — Web Interface](#phase-6--web-interface)
10. [Phase 7 — Teardown](#phase-7--teardown)
11. [Cost Summary](#cost-summary)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

An end-to-end cloud-based IoT cybersecurity chatbot deployed on AWS. The system:

- Preprocesses the **Edge-IIoTset** dataset (2.2M IoT traffic records) using **Apache Spark on EMR**
- Fine-tunes **TinyLlama-1.1B** with **LoRA (Unsloth)** on 100,000 labelled samples in **Google Colab**
- Serves the fine-tuned model via **Ollama** on an **EC2 t3.large** instance
- Exposes a **Streamlit** chat interface accessible from any browser on port 8501

```
User Browser
     │ HTTPS:8501
     ▼
EC2 t3.large (3.92.74.249)
  ├── Streamlit  :8501   ← web interface
  └── Ollama     :11434  ← LLM runner (localhost only)
           │
     cybersecurity-model (TinyLlama-1.1B + LoRA adapter)
           │
    S3: 20596360-models (adapter weights)
           ▲
    Google Colab (fine-tuning)
           ▲
    S3: 20596360-processed-data (JSONL)
           ▲
    EMR Cluster (PySpark preprocessing)
           ▲
    S3: 20596360-raw-data (Edge-IIoTset CSV)
```

---

## Prerequisites

### Accounts & Access
| Requirement | Notes |
|---|---|
| AWS Account | Access to EC2, EMR, S3, VPC in `us-east-1`. Shared course account — prefix all resources with `20596360-` |
| AWS CLI | v2.x configured with `aws configure` (access key, secret, region `us-east-1`, output `json`) |
| Google Account | For Google Colab (free tier sufficient; Pro recommended for faster T4 GPU allocation) |
| GitHub Account | Public repository for code submission |
| SSH Key Pair | Create `20596360-key` in EC2 Console → Key Pairs; download `.pem` file |

### Local Tools
| Tool | Version | Install |
|---|---|---|
| AWS CLI | v2.x | https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html |
| Python | 3.9+ | https://www.python.org/downloads/ |
| Git | any | https://git-scm.com/ |
| SSH client | any | Built-in on macOS/Linux; use PuTTY or Windows Terminal on Windows |
| Web browser | any | To access Streamlit on port 8501 |

### AWS CLI Configuration
```bash
aws configure
# AWS Access Key ID:     <from course account>
# AWS Secret Access Key: <from course account>
# Default region name:   us-east-1
# Default output format: json
```

### Verify Access
```bash
aws sts get-caller-identity
aws s3 ls
aws ec2 describe-vpcs --region us-east-1
```

---

## Repository Structure

```
cisc886-iot-chatbot/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies for EC2
├── preprocessing/
│   └── pyspark.py                     # PySpark EMR preprocessing pipeline
├── fine-tuning/
│   └── CISC886_fine_tuning.ipynb      # Annotated Colab fine-tuning notebook
├── deployment/
│   ├── app.py                         # Streamlit web application
│   ├── Modelfile                      # Ollama model configuration
│   └── install_script.sh              # One-command EC2 setup script
├── terraform/
│   └── main.tf                        # (Optional) Terraform VPC provisioning
└── report/
    └── CISC886_Project_Report.pdf     # Final submission report
```

---

## Phase 1 — VPC & Networking

**Goal:** Create an isolated network environment for all project resources.

### Option A — AWS Console (Manual)

#### Step 1.1 — Create VPC
```
AWS Console → VPC → Your VPCs → Create VPC
  Name:        20596360-vpc
  IPv4 CIDR:   10.0.0.0/16
  Tenancy:     Default
→ Create VPC
```

#### Step 1.2 — Create Public Subnet
```
VPC → Subnets → Create subnet
  VPC:               20596360-vpc
  Subnet name:       20596360-subnet-public
  Availability Zone: us-east-1a
  IPv4 CIDR:         10.0.1.0/24
→ Create subnet

# Enable auto-assign public IPv4:
Select subnet → Actions → Edit subnet settings
  ✓ Enable auto-assign public IPv4 address
→ Save
```

#### Step 1.3 — Create and Attach Internet Gateway
```
VPC → Internet Gateways → Create internet gateway
  Name: 20596360-igw
→ Create

Select 20596360-igw → Actions → Attach to VPC
  VPC: 20596360-vpc
→ Attach
```

#### Step 1.4 — Configure Route Table
```
VPC → Route Tables → Create route table
  Name: 20596360-rt-public
  VPC:  20596360-vpc
→ Create

Select route table → Routes → Edit routes → Add route
  Destination: 0.0.0.0/0
  Target:      20596360-igw
→ Save

→ Subnet associations → Edit → Select 20596360-subnet-public → Save
```

#### Step 1.5 — Create Security Group
```
VPC → Security Groups → Create security group
  Name:        20596360-chatbot-sg
  Description: IoT chatbot security group
  VPC:         20596360-vpc

Inbound rules → Add rule:
  Type: SSH         Port: 22    Source: My IP         (admin access only)
  Type: Custom TCP  Port: 8501  Source: 0.0.0.0/0     (Streamlit web interface)
  Type: Custom TCP  Port: 11434 Source: 127.0.0.1/32  (Ollama — localhost only)

Outbound rules: All traffic 0.0.0.0/0 (default)
→ Create security group
```

### Option B — AWS CLI
```bash
# Create VPC
VPC_ID=$(aws ec2 create-vpc \
  --cidr-block 10.0.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=20596360-vpc}]' \
  --query 'Vpc.VpcId' --output text)
echo "VPC: $VPC_ID"

# Create subnet
SUBNET_ID=$(aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 10.0.1.0/24 \
  --availability-zone us-east-1a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=20596360-subnet-public}]' \
  --query 'Subnet.SubnetId' --output text)
echo "Subnet: $SUBNET_ID"

# Enable auto-assign public IP
aws ec2 modify-subnet-attribute \
  --subnet-id $SUBNET_ID \
  --map-public-ip-on-launch

# Create and attach internet gateway
IGW_ID=$(aws ec2 create-internet-gateway \
  --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=20596360-igw}]' \
  --query 'InternetGateway.InternetGatewayId' --output text)
aws ec2 attach-internet-gateway --internet-gateway-id $IGW_ID --vpc-id $VPC_ID

# Create route table with default route
RT_ID=$(aws ec2 create-route-table \
  --vpc-id $VPC_ID \
  --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=20596360-rt-public}]' \
  --query 'RouteTable.RouteTableId' --output text)
aws ec2 create-route --route-table-id $RT_ID \
  --destination-cidr-block 0.0.0.0/0 --gateway-id $IGW_ID
aws ec2 associate-route-table --route-table-id $RT_ID --subnet-id $SUBNET_ID

# Create security group
MY_IP=$(curl -s https://checkip.amazonaws.com)/32
SG_ID=$(aws ec2 create-security-group \
  --group-name 20596360-chatbot-sg \
  --description "IoT chatbot security group" \
  --vpc-id $VPC_ID \
  --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $SG_ID \
  --ip-permissions \
  "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$MY_IP,Description=SSH-admin}]" \
  "IpProtocol=tcp,FromPort=8501,ToPort=8501,IpRanges=[{CidrIp=0.0.0.0/0,Description=Streamlit}]" \
  "IpProtocol=tcp,FromPort=11434,ToPort=11434,IpRanges=[{CidrIp=127.0.0.1/32,Description=Ollama-localhost}]"

echo "SG: $SG_ID"
```

---

## Phase 2 — S3 Buckets

**Goal:** Create storage buckets for raw data, processed data, model weights, and scripts.

```bash
# Create all four project buckets
aws s3 mb s3://20596360-raw-data       --region us-east-1
aws s3 mb s3://20596360-processed-data --region us-east-1
aws s3 mb s3://20596360-models         --region us-east-1
aws s3 mb s3://20596360-scripts        --region us-east-1

# Upload PySpark script
aws s3 cp preprocessing/pyspark.py s3://20596360-scripts/pyspark.py

# Upload Edge-IIoTset dataset (download from Kaggle first)
# Dataset: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot
aws s3 cp DNN-EdgeIIoT-dataset.csv s3://20596360-raw-data/DNN-EdgeIIoT-dataset.csv

# Verify uploads
aws s3 ls s3://20596360-raw-data/
aws s3 ls s3://20596360-scripts/
```

---

## Phase 3 — EMR Data Preprocessing

**Goal:** Run PySpark on EMR to preprocess 2.2M IoT records into 100,000 JSONL training pairs.

> ⚠️ **Cost warning:** EMR m5.xlarge costs ~$0.24/hr per node. Terminate the cluster immediately after the job completes. A missing teardown screenshot will be treated as an incomplete submission.

### Step 3.1 — Create EMR Cluster

```
AWS Console → EMR → Create cluster
  Name:            20596360-emr-cluster
  EMR Release:     emr-7.1.0
  Applications:    Spark, Hadoop
  Instance type:   m5.xlarge (master + 1 core node)
  EC2 key pair:    20596360-key
  VPC:             20596360-vpc
  Subnet:          20596360-subnet-public
→ Create cluster
```

### Step 3.2 — Submit PySpark Job

```
EMR → Clusters → 20596360-emr-cluster → Steps → Add step
  Step type:      Spark application
  Name:           preprocess-edgeiiotset
  Application:    s3://20596360-scripts/pyspark.py
  Arguments:      (none)
  Action on failure: Continue
→ Add
```

### Step 3.3 — PySpark Script Reference

The full script is at `preprocessing/pyspark.py`. Key operations:

```python
from pyspark.sql import SparkSession
import json

spark = SparkSession.builder.appName("IoT-Preprocess").getOrCreate()

# Load raw CSV from S3
df = spark.read.csv(
    "s3://20596360-raw-data/DNN-EdgeIIoT-dataset.csv",
    header=True, inferSchema=False
)

# Drop rows with nulls in key feature columns
FEATURES = ["frame.len", "ip.proto", "tcp.flags",
            "Attack_label", "Attack_type"]
df = df.dropna(subset=FEATURES)

# Cast label to integer
from pyspark.sql.functions import col
df = df.withColumn("Attack_label", col("Attack_label").cast("integer"))

# Sample 100,000 records (stratified)
df_sample = df.sample(False, fraction=0.05, seed=42).limit(100000)

# Format as instruction-following JSONL
def format_record(row):
    label = "Yes, this is an attack." if row["Attack_label"] == 1 \
            else "No, this is normal traffic."
    record = {
        "user": {"content":
            f"Analyze this IoT network traffic. "
            f"Protocol ARP opcode: {row.get('arp.opcode','0.0')}, "
            f"TCP flags: {row.get('tcp.flags','0.0')}, "
            f"TCP packet size: {row.get('frame.len','0.0')}. "
            f"Is this an attack?"},
        "assistant": {"content": label}
    }
    return json.dumps(record)

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
format_udf = udf(format_record, StringType())
df_jsonl = df_sample.withColumn("jsonl", format_udf(struct(*df_sample.columns)))

# Write output to S3
df_jsonl.select("jsonl").coalesce(1).write.mode("overwrite").text(
    "s3://20596360-processed-data/"
)
```

### Step 3.4 — Verify Output

```bash
# Check output landed in S3
aws s3 ls s3://20596360-processed-data/
# Expected: fine_tuning_dataset.jsonl (~19.5 MB)

# Preview first 2 records
aws s3 cp s3://20596360-processed-data/fine_tuning_dataset.jsonl - | head -2
```

### Step 3.5 — Terminate Cluster (REQUIRED)

```
EMR Console → Clusters → 20596360-emr-cluster
→ Terminate  (confirm)
→ Wait for status: Terminated
→ Take screenshot of Terminated status
```

---

## Phase 4 — Model Fine-Tuning (Google Colab)

**Goal:** Fine-tune TinyLlama-1.1B with LoRA on the preprocessed JSONL dataset.

> Hardware used: NVIDIA Tesla T4 GPU (16 GB VRAM) on Google Colab Pro.  
> Runtime: ~4 minutes for 200 training steps.

### Step 4.1 — Download Dataset from S3

```bash
# On your local machine — download the processed JSONL
aws s3 cp s3://20596360-processed-data/fine_tuning_dataset.jsonl ./fine_tuning_dataset.jsonl
```

### Step 4.2 — Open Colab Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `fine-tuning/CISC886_fine_tuning.ipynb`
3. Runtime → Change runtime type → **GPU (T4)**
4. Run all cells in order (`Runtime → Run all`)

### Step 4.3 — Key Hyperparameters

| Hyperparameter | Value |
|---|---|
| Base model | `unsloth/tinyllama-bnb-4bit` |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Learning rate | 2e-4 |
| Batch size | 2 (effective: 8 with grad accum = 4) |
| Max steps | 200 |
| Optimiser | adamw_8bit |
| Trainable params | 4,505,600 / 1,104,553,984 (0.41%) |

### Step 4.4 — Download and Upload Adapter

After training completes, the notebook auto-downloads `cybersecurity_chatbot.zip` (~11 MB).

```bash
# Upload adapter weights to S3
aws s3 cp cybersecurity_chatbot.zip s3://20596360-models/cybersecurity_chatbot.zip

# Verify
aws s3 ls s3://20596360-models/
```

---

## Phase 5 — EC2 Deployment

**Goal:** Launch an EC2 instance, install Ollama, load the fine-tuned model, and serve it.

### Step 5.1 — Launch EC2 Instance

```
EC2 Console → Instances → Launch instances
  Name:          20596360-ec2
  AMI:           Ubuntu Server 22.04 LTS (ami-0c7217cdde317cfec)
  Instance type: t3.large
  Key pair:      20596360-key
  VPC:           20596360-vpc
  Subnet:        20596360-subnet-public
  Security group: 20596360-chatbot-sg
  Storage:       30 GB gp3
→ Launch instance
```

### Step 5.2 — Connect via SSH

```bash
# Set correct permissions on key file (Linux/macOS)
chmod 400 20596360-key.pem

# SSH into instance (replace with your actual public IP)
ssh -i 20596360-key.pem ubuntu@3.92.74.249
```

### Step 5.3 — Install Ollama

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install Ollama (official installer)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server in background
ollama serve &

# Wait for server to be ready
sleep 5

# Pull base TinyLlama model weights
ollama pull tinyllama

# Verify model is available
ollama list
```

### Step 5.4 — Create and Register Fine-Tuned Model

```bash
# Download adapter weights from S3
aws s3 cp s3://20596360-models/cybersecurity_chatbot.zip ./
unzip cybersecurity_chatbot.zip

# Create Modelfile with cybersecurity system prompt
cat > Modelfile << 'EOF'
FROM tinyllama
PARAMETER temperature 0.1
SYSTEM Classify IoT network traffic as an attack or normal. Respond concisely: "Yes, this is an attack." or "No, this is normal traffic."
EOF

# Register the fine-tuned model with Ollama
ollama create cybersecurity-model -f Modelfile

# Verify registration
ollama list
# Expected output: cybersecurity-model listed
```

### Step 5.5 — Test Model API

```bash
# Test via curl (take screenshot of this output for Section 6)
curl http://localhost:11434/api/generate -d '{
  "model": "cybersecurity-model",
  "prompt": "Analyze this IoT network traffic. Protocol ARP opcode: 0.0, TCP flags: 2.0, TCP packet size: 40.0. Is this an attack?",
  "stream": false
}'
# Expected response: {"response":"Yes, this is an attack.",...}
```

### Step 5.6 — Install Python Dependencies

```bash
# Install pip if not present
sudo apt-get install python3-pip -y

# Install Streamlit and requests
pip3 install streamlit requests

# Verify installation
streamlit --version
```

### Step 5.7 — Deploy Streamlit Application

```bash
# Create the app file
cat > app.py << 'EOF'
import streamlit as st
import requests
import json

st.set_page_config(page_title="IoT Cybersecurity Chatbot", page_icon="🔒")
st.title("🔒 IoT Cybersecurity Analyst")
st.caption("Model: cybersecurity-model (TinyLlama-1.1B + LoRA) · Powered by Ollama")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Describe network traffic to classify..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "cybersecurity-model", "prompt": prompt, "stream": False}
            )
            answer = response.json().get("response", "Error: no response")
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
EOF

# Run Streamlit (accessible on port 8501)
streamlit run app.py --server.address 0.0.0.0 --server.port 8501 &
```

---

## Phase 6 — Web Interface

**Goal:** Configure the Streamlit interface to start automatically on server reboot.

### Step 6.1 — Create systemd Service

```bash
# Create systemd unit file for auto-start
sudo tee /etc/systemd/system/streamlit.service > /dev/null << 'EOF'
[Unit]
Description=Streamlit IoT Cybersecurity Chatbot
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=/home/ubuntu/.local/bin/streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 8501
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd, enable, and start service
sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl start streamlit

# Verify service is running
sudo systemctl status streamlit
```

### Step 6.2 — Create systemd Service for Ollama

```bash
# Ollama installer creates its own service automatically
# Verify it is enabled
sudo systemctl status ollama
sudo systemctl enable ollama   # enable if not already

# Restart both services after reboot test
sudo reboot
# After SSH reconnect:
sudo systemctl status ollama
sudo systemctl status streamlit
```

### Step 6.3 — Access the Interface

Open a browser and navigate to:
```
http://3.92.74.249:8501
```

Sample test queries:
```
TCP SYN flood from 192.168.1.100 — flags: 2, size: 40
ARP opcode 0, TCP flags 18, packet size 0
UDP packet size 1400, no TCP flags
```

---

## Phase 7 — Teardown

> ⚠️ Always tear down resources after grading to avoid depleting the shared account budget.

```bash
# Stop and disable services on EC2
sudo systemctl stop streamlit
sudo systemctl stop ollama

# Terminate EC2 instance via CLI
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=20596360-ec2" \
  --query "Reservations[0].Instances[0].InstanceId" \
  --output text)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID

# Delete S3 bucket contents and buckets
aws s3 rm s3://20596360-raw-data       --recursive
aws s3 rm s3://20596360-processed-data --recursive
aws s3 rm s3://20596360-models         --recursive
aws s3 rm s3://20596360-scripts        --recursive
aws s3 rb s3://20596360-raw-data
aws s3 rb s3://20596360-processed-data
aws s3 rb s3://20596360-models
aws s3 rb s3://20596360-scripts

# Delete security group, route table, subnet, IGW, VPC
# (Must delete in reverse dependency order)
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=20596360-chatbot-sg" \
  --query "SecurityGroups[0].GroupId" --output text)
aws ec2 delete-security-group --group-id $SG_ID

SUBNET_ID=$(aws ec2 describe-subnets \
  --filters "Name=tag:Name,Values=20596360-subnet-public" \
  --query "Subnets[0].SubnetId" --output text)
aws ec2 delete-subnet --subnet-id $SUBNET_ID

RT_ID=$(aws ec2 describe-route-tables \
  --filters "Name=tag:Name,Values=20596360-rt-public" \
  --query "RouteTables[0].RouteTableId" --output text)
aws ec2 delete-route-table --route-table-id $RT_ID

IGW_ID=$(aws ec2 describe-internet-gateways \
  --filters "Name=tag:Name,Values=20596360-igw" \
  --query "InternetGateways[0].InternetGatewayId" --output text)
VPC_ID=$(aws ec2 describe-vpcs \
  --filters "Name=tag:Name,Values=20596360-vpc" \
  --query "Vpcs[0].VpcId" --output text)
aws ec2 detach-internet-gateway --internet-gateway-id $IGW_ID --vpc-id $VPC_ID
aws ec2 delete-internet-gateway --internet-gateway-id $IGW_ID
aws ec2 delete-vpc --vpc-id $VPC_ID

echo "✅ All resources deleted."
```

---

## Cost Summary

All costs below are approximate and based on actual usage during development.

| AWS Service | Resource | Usage | Unit Price | Approx. Cost |
|---|---|---|---|---|
| **EMR** | 2× m5.xlarge (master + core) | ~2 hours | $0.24/hr/node | **$0.96** |
| **EC2** | t3.large (2 vCPU, 8 GB RAM) | ~10 hours | $0.0832/hr | **$0.83** |
| **S3 Storage** | 4 buckets (~25 GB total) | 1 month | $0.023/GB/month | **$0.58** |
| **S3 Requests** | PUT/GET operations | ~5,000 requests | $0.0004/1000 | **$0.002** |
| **Data Transfer** | EC2 → Internet (Streamlit traffic) | ~1 GB | $0.09/GB | **$0.09** |
| **EBS Storage** | 30 GB gp3 (EC2 root volume) | ~10 hours | $0.08/GB/month | **$0.003** |
| | | | **Total** | **~$2.47** |

> **Note:** The EMR cluster was terminated immediately after the PySpark job completed (~2 hours total including provisioning). The EC2 instance was kept running only during development and screenshot collection. Costs on the shared course account will vary depending on how long resources remain active.

### Cost Optimisation Decisions

- **t3.large instead of t3.xlarge for EC2:** TinyLlama-1.1B at 4-bit quantisation requires ~1.1 GB RAM. A t3.large (8 GB) is sufficient and costs ~50% less than t3.xlarge.
- **m5.xlarge × 2 for EMR instead of larger cluster:** The preprocessing job (100k records) completes in under 8 minutes on a 2-node cluster. Adding more nodes would increase cost with negligible time savings at this data scale.
- **No NAT Gateway:** Using a public subnet for all resources avoids the $0.045/hr NAT Gateway charge while still allowing Internet access for package installation.
- **save_strategy="no" in fine-tuning:** Disabling checkpoint saves avoided repeated S3 write costs during training iteration.

---

## Troubleshooting

### Ollama not responding
```bash
# Check if Ollama service is running
sudo systemctl status ollama

# Restart if needed
sudo systemctl restart ollama
sleep 5

# Test locally
curl http://localhost:11434/api/tags
```

### Streamlit not accessible on port 8501
```bash
# Verify security group allows port 8501
aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=20596360-chatbot-sg" \
  --query "SecurityGroups[0].IpPermissions"

# Check Streamlit service status
sudo systemctl status streamlit
sudo journalctl -u streamlit -n 50

# Restart
sudo systemctl restart streamlit
```

### EMR job fails
```bash
# Check step logs in S3 (EMR writes logs automatically)
aws s3 ls s3://aws-logs-<account-id>-us-east-1/elasticmapreduce/

# Common causes:
# 1. S3 bucket permissions — ensure EMR role has s3:GetObject / s3:PutObject
# 2. CSV header mismatch — verify column name 'Attack_label' exists with head -1
aws s3 cp s3://20596360-raw-data/DNN-EdgeIIoT-dataset.csv - | head -1
```

### SSH connection refused
```bash
# Verify instance is running and has public IP
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=20596360-ec2" \
  --query "Reservations[0].Instances[0].{State:State.Name,IP:PublicIpAddress}"

# Check key file permissions
ls -l 20596360-key.pem
# Should be: -r-------- (400)
chmod 400 20596360-key.pem
```

### Model produces generic (non-cybersecurity) responses
```bash
# Verify the cybersecurity-model is registered (not just tinyllama)
ollama list

# If only tinyllama is listed, re-create the model
ollama create cybersecurity-model -f Modelfile
ollama list   # should now show cybersecurity-model
```

---

## Quick Replication Checklist

```
Phase 1 — VPC & Networking
  [ ] 20596360-vpc created (10.0.0.0/16)
  [ ] 20596360-subnet-public created (10.0.1.0/24)
  [ ] 20596360-igw attached, route table configured
  [ ] 20596360-chatbot-sg created (ports 22, 8501, 11434)

Phase 2 — S3
  [ ] 4 buckets created with 20596360- prefix
  [ ] Dataset uploaded to 20596360-raw-data
  [ ] pyspark.py uploaded to 20596360-scripts

Phase 3 — EMR
  [ ] 20596360-emr-cluster created and job submitted
  [ ] fine_tuning_dataset.jsonl in 20596360-processed-data
  [ ] Cluster terminated (screenshot taken)

Phase 4 — Fine-Tuning
  [ ] Colab notebook run to completion
  [ ] cybersecurity_chatbot.zip downloaded
  [ ] Adapter uploaded to 20596360-models

Phase 5 — EC2 Deployment
  [ ] 20596360-ec2 launched (t3.large, Ubuntu 22.04)
  [ ] Ollama installed and tinyllama pulled
  [ ] cybersecurity-model created via Modelfile
  [ ] curl test screenshot taken

Phase 6 — Web Interface
  [ ] app.py deployed on port 8501
  [ ] systemd service enabled (auto-start)
  [ ] Browser screenshot taken showing model name
  [ ] Sample conversation screenshot taken

Phase 7 — Submission
  [ ] All code pushed to GitHub (public repo)
  [ ] README.md complete with replication steps
  [ ] Report PDF exported and uploaded to OnQ
```

---

*CISC 886 Cloud Computing — Queen's University | NetID: 20596360*
