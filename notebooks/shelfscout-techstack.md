# ShelfScout: GCP Tech Stack

## Overview
This tech stack represents a fully managed, production-ready implementation of ShelfScout - a computer vision system for retail product detection using Google Cloud Platform's ML services, with an emphasis on Vertex AI managed services.

## Data Management
- **Cloud Storage**: Central repository for image datasets and exported models
- **Vertex AI Datasets**: Managed dataset versioning and labeling for image data
- **Vertex AI Feature Store**: Centralized feature repository for consistent training and serving
- **BigQuery**: Analytics and metadata storage for model performance and business metrics

## Development & Experimentation
- **Vertex AI Workbench**: Interactive Jupyter environment for model development
- **Vertex AI Experiments**: Experiment tracking and comparison
- **Model Garden**: Pre-trained foundation models for computer vision
- **Vertex AI TensorBoard**: Visualization of training metrics and model performance

## Model Training
- **Vertex AI AutoML Vision**: Low-code model development for baseline models
- **Vertex AI Training**: Custom training service for specialized vision models
- **Vertex AI Hyperparameter Tuning**: Automated optimization of model parameters
- **GPU/TPU Resources**: Accelerated hardware for deep learning model training

## Model Deployment & Serving
- **Vertex AI Model Registry**: Centralized model management and versioning
- **Vertex AI Endpoints**: Fully managed online prediction service with auto-scaling
- **Vertex AI Batch Prediction**: Managed large-scale batch inference
- **Vertex AI Multi-model Serving**: Efficient deployment of multiple model versions

## MLOps & Orchestration
- **Vertex AI Pipelines**: End-to-end ML workflow orchestration
- **Cloud Build**: CI/CD automation for model deployment
- **Vertex ML Metadata**: Tracking and auditing of model lineage and artifacts
- **Predefined Pipeline Components**: Accelerate development with ready-to-use TFX components

## Monitoring & Maintenance
- **Vertex AI Model Monitoring**: Automated drift detection and monitoring
- **Cloud Monitoring**: System metrics and dashboards
- **Cloud Logging**: Centralized logging for all components
- **Vertex AI Continuous Evaluation**: Ongoing validation of model performance

## Responsible AI & Evaluation
- **Vertex Explainable AI**: Model interpretability and feature attribution
- **Vertex AI Evaluation**: Systematic model evaluation and comparison
- **Fairness Indicators**: Bias detection and mitigation tools
- **Model Cards**: Documentation of model characteristics and limitations

## Security & Governance
- **IAM (Identity and Access Management)**: Fine-grained access control
- **Secret Manager**: Secure management of credentials and secrets
- **VPC Service Controls**: Network security perimeter
- **Data Loss Prevention (DLP)**: Protection of sensitive information in images

## Specialized Vision Services
- **Vertex AI Vision API**: Pre-built API for general computer vision tasks
- **Document AI**: Processing text in product images
- **Vision AI**: Additional product detection capabilities
- **Vertex AI Agent Builder**: Optional conversational interface for system interaction

## System Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Cloud Storage  │     │  Vertex AI      │     │  Vertex AI       │
│  (Image Data)   │────►│  AutoML Vision  │────►│  Custom Training │
└─────────────────┘     └─────────────────┘     └──────────┬───────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Vertex AI      │     │  Vertex AI      │     │  Vertex AI       │
│  Feature Store  │◄────┤  Pipelines      │◄────┤  Experiments     │
└────────┬────────┘     └─────────────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Vertex AI      │     │  Vertex AI      │     │  Vertex AI       │
│  Endpoints      │◄────┤  Model Registry │◄────┤  Evaluation      │
└────────┬────────┘     └─────────────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Vertex AI      │     │  Vertex AI      │     │  Vertex          │
│  Model Monitoring│────►│  Explainable AI │────►│  Continuous Eval │
└─────────────────┘     └─────────────────┘     └──────────────────┘
```

This fully managed, Vertex AI-centric tech stack enables ShelfScout to demonstrate Professional ML Engineering excellence by leveraging Google Cloud's purpose-built services across the complete ML lifecycle while minimizing operational overhead.
