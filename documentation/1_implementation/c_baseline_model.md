# ShelfScout: Baseline Model Development

## AutoML Training Configuration

- **Dataset**: ShelfScout_SKU110K_Datasets (11,689 images)
- **Annotation Set**: ShelfScout_SKU110K_Datasets_iod
- **Training Method**: AutoML (high-accuracy mode)
- **Model Name**: ShelfScout_AutoML_Baseline
- **Model ID**: shelfscout-automl-baseline-20250419
- **Description**: Product Object Detection

## Data Configuration
- **Split Method**: Random (80/10/10)
  - Training: 80% (~9,351 images)
  - Validation: 10% (~1,169 images)
  - Test: 10% (~1,169 images)

## Performance Settings
- **Optimization Target**: Higher accuracy (new)
- **Expected Latency**: 150ms-180ms
- **Compute Resources**: 26 node hours
- **Estimated Completion**: ~2 hours

## Evaluation Results
- **Mean Average Precision (mAP)**: 0.673 (67.3%)
- **Precision**: 95.1%
- **Recall**: 47%
- **Total Test Images**: 1,218 

## Model Characteristics
- **High precision / lower recall trade-off**: The model correctly identifies products with high confidence (few false positives) but misses a significant portion of products (more false negatives)
- **Effective for clear, well-defined products**: Given the high precision, detections are reliable when made
- **Challenges with dense product arrangements**: The lower recall suggests difficulty with tightly packed products on shelves
- **Deployed endpoint**: Successfully deployed to "shelfscout_baseline_model" endpoint in us-central1
- **Computing resources**: 1 node allocated for serving