# ShelfScout: Data Ingestion Documentation

## Data Acquisition & Analysis
- Successfully downloaded SKU-110K dataset from Kaggle
- Dataset contains 11,762 retail shelf images with ~1.7 million annotated product bounding boxes
- Average of 147 objects per image
- Conducted comprehensive analysis of dataset characteristics

## Dataset Consolidation
- Reorganized dataset into unified structure:
  ```
  SKU110K_consolidated/
  ├── images/         # All images with numerical names (0.jpg, 1.jpg, etc.)
  ├── annotations/    # All annotations with matching numerical names
  └── annotations.jsonl  # Annotations in Vertex AI format
  ```
- Implemented numerical renaming system for all images and annotations
- Maintained original YOLO format for annotation files
- Generated JSONL file with proper GCS paths for Vertex AI integration

## Data Analysis Results
- **Image Properties**:
  - Resolution range: Primarily 600-800 pixels in each dimension
  - Aspect ratio distribution: Mostly portrait orientation (0.7-1.5 range)
  - High variety of lighting conditions and angles
- **Annotation Analysis**:
  - Average of ~147 objects per image
  - High object density (products tightly packed on shelves)
  - Small object size (most products occupy <5% of image area)
  - Single class ("product")
- **Key Challenges Identified**:
  1. Small objects (tiny product footprint in images)
  2. Dense packing of products on shelves
  3. Visual similarity between many products
  4. Lighting variations across different shelf sections

## GCP Storage & Integration
- Uploaded entire consolidated dataset to GCS bucket
- Final GCS structure:
  ```
  gs://sku-110k-dataset/
  └── SKU110K_consolidated/
      ├── images/
      │   ├── 0.jpg
      │   ├── 1.jpg
      │   └── ...
      ├── annotations/
      │   ├── 0.txt
      │   ├── 1.txt
      │   └── ...
      └── annotations.jsonl
  ```
- Verified upload integrity and completeness

## Vertex AI Dataset Creation
- Created Vertex AI dataset: `ShelfScout_SKU110K_Datasets`
- Import source: `gs://sku-110k-dataset/SKU110K_consolidated/annotations.jsonl`
- Successfully imported 11,689 out of 11,762 images (99.4% success rate)
- All images properly labeled with "product" class and bounding box annotations
- Applied appropriate metadata labels to the dataset