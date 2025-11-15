# 📌 Sentiment Analysis on IMDB Reviews — End-to-End MLOps Project

This repository contains a complete **end-to-end MLOps pipeline** built for **Sentiment Analysis using the IMDB Movie Review dataset**.  
The project demonstrates industry-level workflows including:

- Experiment tracking with **MLflow (Dagshub)**
- Version control for data & models using **DVC**
- Model packaging using **Flask & Docker**
- CI/CD using **GitHub Actions**
- Deployment on **AWS ECR + EKS (Kubernetes)**
- Monitoring with **Prometheus & Grafana**

---

## 📂 Project Workflow Summary

### **1. Project Structure Setup**
```
conda create -n atlas python=3.10
conda activate atlas
pip install cookiecutter
cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
```

- Renamed: `src/models` → `src/model`
- Initial GitHub commit & push

---

## 🧪 MLflow Experiment Tracking (Dagshub)

1. Create repository on Dagshub
2. Connect GitHub repository
3. Copy MLflow tracking URI from Dagshub
4. Install required libraries:
```
pip install dagshub mlflow
```
5. Run experiments & push changes to GitHub

---

## 🗃️ DVC Setup (Local → S3)

### Initialize DVC:
```
dvc init
mkdir local_s3
dvc remote add -d mylocal local_s3
```

### Add core modules inside `src/`
- `logger/`
- `data_ingestion.py`
- `data_preprocessing.py`
- `feature_engineering.py`
- `model_building.py`
- `model_evaluation.py`
- `register_model.py`

Add config files:
- `dvc.yaml`
- `params.yaml`

Run pipeline:
```
dvc repro
dvc status
```

### Add S3 Storage
```
pip install dvc[s3] awscli
aws configure
dvc remote add -d myremote s3://<bucket-name>
```

---

## 🔥 Flask Application

Inside `flask_app/`:
```
pip install flask
pip freeze > requirements.txt
```

- Build REST API for model prediction.
- Push code to GitHub.

---

## 🔄 CI/CD Pipeline (GitHub Actions)

Add secrets:

| Secret Name | Description |
|-------------|-------------|
| AWS_ACCESS_KEY_ID | IAM user key |
| AWS_SECRET_ACCESS_KEY | IAM user secret |
| AWS_REGION | Your AWS region |
| ECR_REPOSITORY | ECR repo name |
| AWS_ACCOUNT_ID | AWS account number |
| CAPSTONE_TEST | Dagshub MLFlow token |

Add workflow file:
```
.github/workflows/ci.yaml
```

---

## 🐳 Docker Setup

Generate minimal requirements file:
```
pip install pipreqs
cd flask_app
pipreqs . --force
```

Build Docker image:
```
docker build -t capstone-app:latest .
docker run -p 8888:5000 -e CAPSTONE_TEST=your_token capstone-app:latest
```

(Optional) Push to DockerHub.

---

## ☸️ AWS EKS Deployment (Kubernetes)

### Install CLIs
Ensure:
```
aws --version
kubectl version --client
eksctl version
```

### Create EKS Cluster:
```
eksctl create cluster --name flask-app-cluster --region us-east-1 --nodegroup-name flask-app-nodes --node-type t3.small --nodes 1 --managed
```

Update config:
```
aws eks --region us-east-1 update-kubeconfig --name flask-app-cluster
```

Check:
```
kubectl get nodes
kubectl get pods
kubectl get svc
```

### Deploy via GitHub CI/CD pipeline

LoadBalancer IP:
```
kubectl get svc flask-app-service
```

Test:
```
http://<external-ip>:5000
```

---

## 📊 Monitoring with Prometheus & Grafana

### Prometheus Setup
- Launch Ubuntu EC2 (Port 9090 open)
- Install Prometheus:
```
wget https://github.com/prometheus/prometheus/releases/download/v2.46.0/prometheus-2.46.0.linux-amd64.tar.gz
tar -xvzf ...
sudo mv prometheus /etc/prometheus
sudo mv /etc/prometheus/prometheus /usr/local/bin/
```

Update config:
```
/etc/prometheus/prometheus.yml
```

Add Flask app endpoint:
```
targets: ["<loadbalancer-dns>:5000"]
```

Run:
```
/usr/local/bin/prometheus --config.file=/etc/prometheus/prometheus.yml
```

---

### Grafana Setup
- Launch Ubuntu EC2 (Port 3000 open)
- Install Grafana:
```
wget https://dl.grafana.com/oss/release/grafana_10.1.5_amd64.deb
sudo apt install ./grafana_10.1.5_amd64.deb
```

Access UI:
```
http://<grafana-ec2-ip>:3000
```

Add Prometheus datasource:
```
http://<prometheus-ip>:9090
```

Build dashboards for monitoring ML API performance.

---

## 🧹 AWS Cleanup (Recommended)

```
kubectl delete deployment flask-app
kubectl delete service flask-app-service
kubectl delete secret capstone-secret
eksctl delete cluster --name flask-app-cluster --region us-east-1
```

Also remove:
- S3 bucket
- ECR repository
- EC2 instances
- CloudFormation stacks

---

## 🛠️ Tech Stack

### **Machine Learning**
- Python, NumPy, Pandas, Scikit-learn  
- IMDB Sentiment Dataset

### **MLOps Tools**
- MLflow (Dagshub)
- DVC
- GitHub Actions
- Docker
- AWS ECR
- AWS EKS
- Kubernetes
- Prometheus
- Grafana

### **Deployment**
- Flask REST API

---

## 👤 Author  
**Karan Singh**  
AI/ML Developer • Data Scientist • MLOps Practitioner  
