# ShelfScout: Project Setup Documentation

## GCP Project Creation
- Created project: `shelfscout`
- Enabled necessary APIs:
  - Vertex AI API
  - Cloud Storage API
  - BigQuery API
  - Notebooks API
  - Cloud Build API
  - Container Registry API

## IAM Configuration
- Primary account: Owner role (solo developer project)
- Created service account: `universal-development@shelfscout.iam.gserviceaccount.com`
- Service account roles:
  - `roles/aiplatform.user`
  - `roles/storage.objectAdmin`
  - `roles/bigquery.dataEditor`

## Cloud Storage Configuration
- Bucket name: `sku-110k-dataset`
- Location type: US Multi-region
- Default storage class: Standard
- Access control: Uniform
- Public access prevention: Enabled

## Labels
- `project`: `shelfscout`
- `data-type`: `image-dataset`
- `purpose`: `ml-training`
- `dataset`: `sku110k`

## Status Verification
- Verified all services are operational
- Confirmed service account permissions are working
- Validated bucket access and permissions
- Tested API availability