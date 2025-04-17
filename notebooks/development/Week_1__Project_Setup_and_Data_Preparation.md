# Week 1: Project Setup & Data Preparation

## Project Setup

### GCP Project Creation
- Created project: `shelfscout`
- Enabled necessary APIs:
  - Vertex AI API
  - Cloud Storage API
  - BigQuery API
  - Notebooks API
  - Cloud Build API
  - Container Registry API

### IAM Configuration
- Primary account: Owner role (solo developer project)
- Created service account: `universal-development@shelfscout.iam.gserviceaccount.com`
- Service account roles:
  - `roles/aiplatform.user`
  - `roles/storage.objectAdmin`
  - `roles/bigquery.dataEditor`

## Data Infrastructure

### Cloud Storage Configuration
- Bucket name: `sku-110k-dataset`
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

## Data Analysis

### Data Distribution
- The dataset has a 70/5/25 split for Train/Validation/Test
- This distribution is acceptable for our implementation needs

### Annotation Format
- Annotations stored as text files with normalized coordinates (0-1 range)
- Each line follows the format: [x1 y1 x2 y2 confidence]
- High object density with ~100-180 products per image
- Bounding boxes represent individual retail products on shelves

### Dataset Characteristics
- Images show densely packed retail shelves
- Various product sizes and shapes
- Challenging detection scenario due to visual similarity of products
- Complex retail environments with varying lighting conditions

## Current Progress

- [x] Create GCP project with appropriate IAM permissions
- [x] Set up Cloud Storage buckets for SKU-110K dataset
- [x] Import SKU-110K dataset to GCS
- [x] Create initial Vertex AI Workbench notebooks
- [x] Validate SKU-110K dataset
- [x] Analyze data distribution and annotation format
- [ ] Configure Vertex AI Datasets
- [ ] Implement data preprocessing pipeline