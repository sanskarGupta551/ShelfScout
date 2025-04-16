# Week 1: Project Setup & Data Preparation

## Project Setup

### GCP Project Creation
- Created project: `shelfscout-portfolio`
- Enabled necessary APIs:
  - Vertex AI API
  - Cloud Storage API
  - BigQuery API
  - Notebooks API
  - Cloud Build API
  - Container Registry API

### IAM Configuration
- Primary account: Owner role (solo developer project)
- Created service account: `shelfscout-sa@shelfscout-portfolio.iam.gserviceaccount.com`
- Service account roles:
  - `roles/aiplatform.user`
  - `roles/storage.objectAdmin`
  - `roles/bigquery.dataEditor`

## Data Infrastructure

### Cloud Storage Configuration
- Bucket name: `shelfscout-data`
- Location type: US Multi-region
- Default storage class: Standard
- Access control: Uniform
- Public access prevention: Enabled
- Soft delete policy: 7 days (default)
- Object versioning: Enabled
- Lifecycle configuration:
  - Delete object when noncurrent (5+ newer versions)
  - Delete object 3+ days since becoming noncurrent

### Labels
- `project`: `shelfscout`
- `data-type`: `image-dataset`
- `purpose`: `ml-training`
- `dataset`: `sku110k`

## Dataset Preparation

### Dataset Acquisition
- Downloaded SKU-110K dataset from Kaggle
- Dataset specs: 11,762 retail shelf images with ~1.7 million annotated product bounding boxes
- Average of 147 objects per image

### Dataset Structure
The dataset was uploaded to GCS with the following structure:
```
SKU110K_Kaggle/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Current Progress

- [x] Create GCP project with appropriate IAM permissions
- [x] Set up Cloud Storage buckets for SKU-110K dataset
- [x] Import SKU-110K dataset to GCS
- [x] Validate SKU-110K dataset
- [ ] Configure Vertex AI Datasets
- [ ] Implement data preprocessing pipeline
- [ ] Create initial Vertex AI Workbench notebooks