# ShelfScout: SKU110K Dataset Documentation

## 1. Dataset Overview

The SKU-110K dataset serves as the foundation for the ShelfScout retail product detection system. This comprehensive dataset captures the complexity and density of retail shelf environments, providing an ideal basis for training accurate product detection models.

### 1.1 Dataset Origin

| Attribute | Details |
|-----------|---------|
| **Name** | SKU-110K |
| **Source** | Published by Cohen et al. (2019) |
| **Domain** | Retail product detection |
| **License** | MIT License |
| **GCS Location** | `gs://sku-110k-dataset/` |

### 1.2 Dataset Scale

| Metric | Value |
|--------|-------|
| **Total Images** | 11,762 |
| **Total Objects** | ~1.7 million bounding boxes |
| **Average Objects Per Image** | ~147 |
| **Object Class** | Single class ("product") |

## 2. Raw Data Characteristics

The original SKU-110K dataset provides a realistic representation of retail shelves with dense product arrangements and varying conditions.

### 2.1 Image Properties

| Property | Description |
|----------|-------------|
| **Resolution** | Primarily 600-800 pixels in each dimension |
| **Aspect Ratio** | Mostly portrait orientation (0.7-1.5 range) |
| **Format** | JPEG |
| **Channels** | RGB (3 channels) |
| **Bit Depth** | 8-bit per channel |

### 2.2 Annotation Format

| Property | Description |
|----------|-------------|
| **Format** | YOLO-style text files (.txt) |
| **Coordinates** | Normalized [0-1] coordinates (class_id, x_center, y_center, width, height) |
| **Class ID** | Single class (0 = product) |
| **Density** | High density with ~147 annotations per image |

### 2.3 Data Characteristics

| Characteristic | Description |
|----------------|-------------|
| **Object Size** | Small objects (most products occupy <5% of image area) |
| **Object Density** | High density with significant overlap between products |
| **Lighting Conditions** | Varied lighting across store environments |
| **Viewpoint Variation** | Multiple angles and perspectives |
| **Background Complexity** | Consistent retail environment backgrounds |

## 3. Managed Data Organization

The managed dataset represents the consolidated, organized version of SKU-110K used as the input to the preprocessing pipeline.

### 3.1 Directory Structure

```
SKU110K_consolidated/
├── images/         # All images with numerical names (0.jpg, 1.jpg, etc.)
├── annotations/    # All annotations with matching numerical names
└── annotations.jsonl  # Annotations in Vertex AI format
```

### 3.2 Managed Data Format

| Component | Format | Description |
|-----------|--------|-------------|
| **Images** | JPEG | Renamed with sequential numerical identifiers |
| **Annotations** | YOLO TXT | Matching numerical identifiers with original format |
| **Vertex AI Annotations** | JSONL | GCS paths and bounding box annotations for Vertex AI |

### 3.3 Data Validation

| Validation Aspect | Details |
|-------------------|---------|
| **Filename Matching** | 100% match between image and annotation files |
| **Format Consistency** | Verified YOLO format correctness |
| **Import Success Rate** | 99.4% (11,689 of 11,762 images) |
| **GCS Location** | `gs://sku-110k-dataset/SKU110K_consolidated/` |

## 4. Processed Data Structure

The processed dataset represents the augmented, TFRecord-formatted version optimized for efficient model training.

### 4.1 TFRecord Structure

Each example in the TFRecord file contains the following features:

| Feature | Type | Description |
|---------|------|-------------|
| `image/height` | int64 | Image height in pixels |
| `image/width` | int64 | Image width in pixels |
| `image/channels` | int64 | Number of channels (3 for RGB) |
| `image/encoded` | bytes | JPEG encoded image data |
| `image/format` | bytes | Image format ('jpeg') |
| `image/object/bbox/xmin` | float list | Normalized xmin coordinates [0-1] |
| `image/object/bbox/xmax` | float list | Normalized xmax coordinates [0-1] |
| `image/object/bbox/ymin` | float list | Normalized ymin coordinates [0-1] |
| `image/object/bbox/ymax` | float list | Normalized ymax coordinates [0-1] |
| `image/object/class/label` | int64 list | Class labels (1 for 'product') |

### 4.2 Augmentation Diversity

The processed dataset includes multiple augmentation types to improve model robustness:

| Category | Augmentation Types | Implementation Details |
|----------|-------------------|------------------------|
| **Color Variations** | Brightness, Contrast, Hue, Saturation | Enhanced ranges for stronger variations |
| **Geometric Transformations** | Horizontal Flip, Rotation, Random Crop | Coordinate-aware transformations |
| **Advanced Augmentations** | Intelligent Cutout, Mosaic | Optimized placement algorithms |
| **Combined Augmentations** | 7 paired combinations | Strategic pairings of complementary augmentations |

### 4.3 Example Distribution

| Metric | Value |
|--------|-------|
| **Original Examples** | 11,762 |
| **Augmented Examples** | ~30,000-35,000 |
| **Augmentations Per Image** | 3-4 (including original) |
| **Selection Strategy** | Smart sampling to prevent overfitting |

### 4.4 Supporting Files

| File | Purpose | Format |
|------|---------|--------|
| `sku110k_complete.tfrecord` | Single consolidated dataset | TFRecord |
| `label_map.pbtxt` | Class mapping file | Protocol Buffer Text |
| `dataset_info.json` | Dataset statistics and metadata | JSON |

### 4.5 GCS Location

```
gs://sku-110k-dataset/SKU110K_tfrecords/
├── sku110k_complete.tfrecord
├── label_map.pbtxt
└── dataset_info.json
```

## 5. Data Quality Metrics

### 5.1 Annotation Quality

| Metric | Value |
|--------|-------|
| **Annotation Coverage** | 100% of visible products |
| **Annotation Precision** | Tight bounding boxes around products |
| **Annotation Consistency** | High consistency across dataset |
| **Box Validation** | Invalid and too-small boxes filtered |

### 5.2 Augmentation Quality

| Aspect | Implementation |
|--------|----------------|
| **Color Range** | Extended range (0.6-1.4) for stronger visual diversity |
| **Geometric Fidelity** | Proper coordinate transformation in all geometric augmentations |
| **Occlusion Handling** | 70% threshold for removing heavily occluded boxes |
| **Visible Area Requirement** | 60% minimum visibility for cropped objects |

### 5.3 Processing Integrity

| Verification Method | Result |
|---------------------|--------|
| **Box Coordinate Validation** | All coordinates within [0-1] range |
| **TFRecord Schema Verification** | Complete feature presence in all examples |
| **Visual Inspection** | Random samples verified for rendering correctness |
| **Size Validation** | No empty or corrupted examples |

## 6. Dataset Usage Guidelines

### 6.1 Recommended Training Practice

| Practice | Recommendation |
|----------|----------------|
| **Train/Val/Test Split** | 80/10/10 recommended during training |
| **Batch Size** | 8-16 depending on GPU memory |
| **Shuffling** | Enable shuffling with buffer size >1000 |
| **Dataset Caching** | Recommended for faster training if memory permits |
| **Prefetching** | Use tf.data prefetch for optimal training speed |

### 6.2 Data Loading Code Example

```python
# Pseudocode example (no implementation)
def create_dataset(tfrecord_path, batch_size=8):
    # Parse TFRecord
    # Apply any additional preprocessing
    # Create batches
    # Enable prefetching
    return dataset
```

### 6.3 Dataset Limitations

| Limitation | Description |
|------------|-------------|
| **Single Product Class** | No distinction between product types |
| **Retail-Specific** | May not generalize to non-retail shelves |
| **Dense Packing Bias** | Models may struggle with sparsely arranged products |
| **Image Quality Variance** | Some images have lower quality or lighting issues |

### 6.4 Future Dataset Enhancements

| Enhancement | Description |
|-------------|-------------|
| **Multi-Class Annotations** | Potential to add product category labels |
| **Text Recognition** | Adding OCR for product text could enhance capabilities |
| **3D Shelf Mapping** | Potential to add shelf structure information |
| **Instance Segmentation** | Future extension to pixel-level segmentation |

## 7. Conclusion

The processed SKU110K dataset represents a high-quality, augmented dataset optimized for training robust retail product detection models. With approximately 30,000-35,000 augmented examples derived from 11,762 original images, the dataset provides substantial diversity while maintaining annotation quality. The intelligent augmentation strategy and single TFRecord format make it ideally suited for efficient model training in the ShelfScout product detection system.