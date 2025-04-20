#!/usr/bin/env python3
"""
ShelfScout YOLOv8 Fine-tuning Script

This script fine-tunes a YOLOv8 model on the SKU-110K retail product
detection dataset using Vertex AI Custom Training with proper Ultralytics implementation.

Example usage:
    python train.py \
        --data-dir gs://sku-110k-dataset/SKU110K_tfrecords \
        --model-dir gs://sku-110k-dataset/models/yolov8-finetuned \
        --config-path config.yaml
"""

import os
import argparse
import logging
import yaml
import json
import time
import shutil
from datetime import datetime
import tensorflow as tf
from google.cloud import storage
from ultralytics import YOLO
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for ShelfScout')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='GCS directory containing dataset files')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='GCS directory for saving model artifacts')
    parser.add_argument('--config-path', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--job-dir', type=str, default='',
                        help='Vertex AI job directory (used internally by Vertex AI)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_from_gcs(gcs_path, local_path):
    """Download files from GCS to local filesystem."""
    if not gcs_path.startswith('gs://'):
        return gcs_path  # Already local path
    
    logger.info(f"Downloading {gcs_path} to {local_path}...")
    
    # Parse GCS path
    bucket_name = gcs_path.replace('gs://', '').split('/')[0]
    blob_path = '/'.join(gcs_path.replace(f'gs://{bucket_name}/', '').split('/'))
    
    # Download directory or file
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    
    if '*' in blob_path:  # Handle wildcards
        prefix = blob_path.split('*')[0]
        blobs = bucket.list_blobs(prefix=prefix)
        os.makedirs(local_path, exist_ok=True)
        
        for blob in blobs:
            filename = os.path.basename(blob.name)
            local_file_path = os.path.join(local_path, filename)
            blob.download_to_filename(local_file_path)
    else:  # Single file
        if blob_path.endswith('/'):  # Directory
            blobs = bucket.list_blobs(prefix=blob_path)
            os.makedirs(local_path, exist_ok=True)
            
            for blob in blobs:
                rel_path = blob.name[len(blob_path):]
                local_file_path = os.path.join(local_path, rel_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                blob.download_to_filename(local_file_path)
        else:  # File
            blob = bucket.blob(blob_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
    
    logger.info(f"Download completed: {gcs_path}")
    return local_path

def upload_to_gcs(local_path, gcs_path, executor=None):
    """
    Upload files to GCS, with option for asynchronous upload.
    
    Args:
        local_path: Path to local file or directory
        gcs_path: Destination GCS path
        executor: Optional ThreadPoolExecutor for async uploads
    
    Returns:
        Future object if async, or GCS path if synchronous
    """
    if not gcs_path.startswith('gs://'):
        return gcs_path  # Not a GCS path
    
    # Parse GCS path
    bucket_name = gcs_path.replace('gs://', '').split('/')[0]
    blob_path = '/'.join(gcs_path.replace(f'gs://{bucket_name}/', '').split('/'))
    
    def _upload_file(local_file_path, blob_name):
        """Helper function to upload a single file to GCS."""
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_file_path)
        logger.info(f"Uploaded {local_file_path} to gs://{bucket_name}/{blob_name}")
        return f"gs://{bucket_name}/{blob_name}"
    
    def _upload_directory(dir_path, base_path):
        """Helper function to upload a directory to GCS."""
        upload_futures = []
        for root, _, files in os.walk(dir_path):
            for filename in files:
                local_file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(local_file_path, start=dir_path)
                dest_blob_name = os.path.join(base_path, rel_path)
                
                if executor:
                    future = executor.submit(_upload_file, local_file_path, dest_blob_name)
                    upload_futures.append(future)
                else:
                    _upload_file(local_file_path, dest_blob_name)
                    
        return upload_futures
    
    logger.info(f"Uploading {local_path} to {gcs_path}...")
    
    if os.path.isdir(local_path):
        if executor:
            return _upload_directory(local_path, blob_path)
        else:
            _upload_directory(local_path, blob_path)
    else:
        if executor:
            return executor.submit(_upload_file, local_path, blob_path)
        else:
            _upload_file(local_path, blob_path)
    
    logger.info(f"Upload initiated: {gcs_path}")
    return gcs_path

def convert_tfrecords_to_yolo(tfrecord_paths, output_dir):
    """Convert TFRecord format to YOLO format."""
    logger.info(f"Converting TFRecords to YOLO format in {output_dir}")
    
    # Create directories
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Feature description for parsing TFRecords
    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    }
    
    def _parse_function(example_proto):
        """Parse TFRecord example."""
        return tf.io.parse_single_example(example_proto, feature_description)
    
    # Process each TFRecord file
    image_count = 0
    for tfrecord_path in tfrecord_paths:
        logger.info(f"Processing TFRecord: {tfrecord_path}")
        
        # Read TFRecord file
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        
        # Process each example
        for example in dataset:
            parsed = _parse_function(example)
            
            # Get image data
            image_data = parsed['image/encoded'].numpy()
            height = parsed['image/height'].numpy()
            width = parsed['image/width'].numpy()
            
            # Get bounding boxes
            xmin = tf.sparse.to_dense(parsed['image/object/bbox/xmin']).numpy()
            xmax = tf.sparse.to_dense(parsed['image/object/bbox/xmax']).numpy()
            ymin = tf.sparse.to_dense(parsed['image/object/bbox/ymin']).numpy()
            ymax = tf.sparse.to_dense(parsed['image/object/bbox/ymax']).numpy()
            class_ids = tf.sparse.to_dense(parsed['image/object/class/label']).numpy()
            
            # Save image
            image_filename = f"{image_count:08d}.jpg"
            image_path = os.path.join(images_dir, image_filename)
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            # Convert boxes to YOLO format (class_id, x_center, y_center, width, height)
            # All values normalized to [0, 1]
            label_filename = f"{image_count:08d}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for i in range(len(xmin)):
                    # For SKU-110K, we have a single class (product), so class_id is 0
                    class_id = 0  # YOLOv8 format uses 0-indexed classes
                    
                    # Convert to YOLO format
                    x_center = (xmin[i] + xmax[i]) / 2.0
                    y_center = (ymin[i] + ymax[i]) / 2.0
                    box_width = xmax[i] - xmin[i]
                    box_height = ymax[i] - ymin[i]
                    
                    # Write to file (class_id, x_center, y_center, width, height)
                    f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")
            
            image_count += 1
            
            # Log progress
            if image_count % 100 == 0:
                logger.info(f"Processed {image_count} images")
    
    logger.info(f"Conversion complete. Total images: {image_count}")
    return images_dir, labels_dir

def create_yolo_dataset_config(output_dir, train_dir=None, val_dir=None, test_dir=None):
    """Create YOLO dataset configuration file."""
    logger.info("Creating YOLO dataset configuration")
    
    # Define class names
    class_names = ['product']
    
    # Create dataset.yaml
    dataset_config = {
        'path': output_dir,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images' if test_dir else '',
        'nc': len(class_names),  # number of classes
        'names': class_names
    }
    
    # Write dataset.yaml
    dataset_yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    logger.info(f"Created dataset configuration: {dataset_yaml_path}")
    return dataset_yaml_path

def prepare_dataset(data_dir, config):
    """Download and prepare dataset for YOLOv8 training."""
    # Local directories
    local_data_dir = './data'
    local_tfrecord_dir = os.path.join(local_data_dir, 'tfrecords')
    local_yolo_dir = os.path.join(local_data_dir, 'yolo')
    
    # Create directories
    os.makedirs(local_tfrecord_dir, exist_ok=True)
    os.makedirs(local_yolo_dir, exist_ok=True)
    
    # Download TFRecord files from GCS
    train_pattern = os.path.join(data_dir, 'train-*.tfrecord')
    val_pattern = os.path.join(data_dir, 'val-*.tfrecord')
    test_pattern = os.path.join(data_dir, 'test-*.tfrecord')
    
    # Download using wildcards
    download_from_gcs(train_pattern, local_tfrecord_dir)
    download_from_gcs(val_pattern, local_tfrecord_dir)
    download_from_gcs(test_pattern, local_tfrecord_dir)
    
    # Get local TFRecord paths
    local_train_tfrecords = [os.path.join(local_tfrecord_dir, f) for f in os.listdir(local_tfrecord_dir) if f.startswith('train-')]
    local_val_tfrecords = [os.path.join(local_tfrecord_dir, f) for f in os.listdir(local_tfrecord_dir) if f.startswith('val-')]
    local_test_tfrecords = [os.path.join(local_tfrecord_dir, f) for f in os.listdir(local_tfrecord_dir) if f.startswith('test-')]
    
    # Create YOLO dataset directories
    train_yolo_dir = os.path.join(local_yolo_dir, 'train')
    val_yolo_dir = os.path.join(local_yolo_dir, 'val')
    test_yolo_dir = os.path.join(local_yolo_dir, 'test')
    
    os.makedirs(train_yolo_dir, exist_ok=True)
    os.makedirs(val_yolo_dir, exist_ok=True)
    os.makedirs(test_yolo_dir, exist_ok=True)
    
    # Convert TFRecords to YOLO format
    convert_tfrecords_to_yolo(local_train_tfrecords, train_yolo_dir)
    convert_tfrecords_to_yolo(local_val_tfrecords, val_yolo_dir)
    
    if local_test_tfrecords:
        convert_tfrecords_to_yolo(local_test_tfrecords, test_yolo_dir)
    
    # Create YOLO dataset configuration
    dataset_yaml_path = create_yolo_dataset_config(local_yolo_dir, train_yolo_dir, val_yolo_dir, test_yolo_dir)
    
    return dataset_yaml_path

def check_for_checkpoint(model_dir):
    """
    Check for existing checkpoints to resume training.
    
    Args:
        model_dir: GCS or local directory where checkpoints might exist
    
    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check if there's a checkpoint in GCS
    if model_dir.startswith('gs://'):
        try:
            checkpoint_gcs_path = os.path.join(model_dir, 'checkpoints/last.pt')
            local_checkpoint_path = os.path.join(checkpoint_dir, 'last.pt')
            download_from_gcs(checkpoint_gcs_path, local_checkpoint_path)
            
            if os.path.exists(local_checkpoint_path):
                logger.info(f"Found checkpoint at {checkpoint_gcs_path}")
                return local_checkpoint_path
        except Exception as e:
            logger.info(f"No checkpoint found to resume: {e}")
            return None
    else:
        # Local path
        checkpoint_path = os.path.join(model_dir, 'checkpoints/last.pt')
        if os.path.exists(checkpoint_path):
            logger.info(f"Found checkpoint at {checkpoint_path}")
            return checkpoint_path
    
    return None

def train_yolov8(dataset_yaml_path, config, model_dir, resume_from=None):
    """
    Train YOLOv8 model with checkpoint support.
    
    Args:
        dataset_yaml_path: Path to YOLO dataset config
        config: Training configuration dictionary
        model_dir: Directory to save model artifacts
        resume_from: Optional path to checkpoint for resuming training
    
    Returns:
        Path to best model weights
    """
    logger.info("Starting YOLOv8 training")
    
    # Set up checkpoint directory
    local_checkpoint_dir = os.path.join('./checkpoints')
    os.makedirs(local_checkpoint_dir, exist_ok=True)
    
    # Get YOLOv8 variant
    yolo_variant = config['model']['variant']
    logger.info(f"Using YOLOv8 variant: {yolo_variant}")
    
    # Load model - either resume from checkpoint or start with pre-trained weights
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming training from checkpoint: {resume_from}")
        model = YOLO(resume_from)
    else:
        logger.info(f"Starting training with pre-trained weights: {yolo_variant}.pt")
        model = YOLO(f"{yolo_variant}.pt")
    
    # Log GPU info
    if torch.cuda.is_available():
        logger.info(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Configure checkpoint saving
    save_period = min(5, config['training']['epochs'] // 10)  # Save every 5 epochs or 10% of total
    
    # Train the model with enhanced parameters
    results = model.train(
        data=dataset_yaml_path,
        epochs=config['training']['epochs'],
        batch=config['training']['batch_size'],
        imgsz=config['model']['image_size'],
        patience=config['training']['early_stopping_patience'],
        device=0,  # Use GPU
        project='shelfscout',
        name='yolov8_retail',
        pretrained=True,
        optimizer=config['training'].get('optimizer', 'Adam'),
        lr0=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0005),
        conf=config['model']['confidence_threshold'],
        iou=config['model']['iou_threshold'],
        
        # Enhanced augmentation for retail product detection
        mosaic=config['training'].get('mosaic', 1.0),
        mixup=config['training'].get('mixup', 0.15),
        copy_paste=config['training'].get('copy_paste', 0.3),
        
        # Validation settings
        val=config['validation'].get('frequency', 1),
        save_json=config['validation'].get('save_json', True),
        
        # Checkpoint settings
        save_period=save_period,  # Save checkpoints periodically
        exist_ok=True,
        
        # Additional training parameters for dense object detection
        overlap_mask=True,  # Helps with overlapping products
        task='detect',
        augment=True,
        verbose=True
    )
    
    # Paths to model artifacts
    best_model_path = os.path.join('shelfscout', 'yolov8_retail', 'weights', 'best.pt')
    last_model_path = os.path.join('shelfscout', 'yolov8_retail', 'weights', 'last.pt')
    results_path = os.path.join('shelfscout', 'yolov8_retail')
    
    # Create thread pool for asynchronous uploads
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Upload best model
        gcs_best_model_path = os.path.join(model_dir, 'best.pt')
        best_future = upload_to_gcs(best_model_path, gcs_best_model_path, executor)
        
        # Upload last checkpoint
        gcs_checkpoint_path = os.path.join(model_dir, 'checkpoints/last.pt')
        os.makedirs(os.path.dirname(gcs_checkpoint_path), exist_ok=True)
        checkpoint_future = upload_to_gcs(last_model_path, gcs_checkpoint_path, executor)
        
        # Upload all results asynchronously
        results_future = upload_to_gcs(results_path, os.path.join(model_dir, 'results'), executor)
        
        # Create and upload model card
        model_card_path = os.path.join(model_dir, 'model_card.md')
        create_model_card(model, config, model_card_path)
        
        # Wait for critical uploads to complete
        logger.info("Waiting for model uploads to complete...")
        best_future.result()
        checkpoint_future.result()
        
    logger.info(f"Model artifacts uploaded to {model_dir}")
    return best_model_path

def create_model_card(model, config, model_card_path):
    """Create model card with information about the trained model."""
    logger.info(f"Creating model card: {model_card_path}")
    
    model_card_content = f"""# ShelfScout YOLOv8 Object Detection Model

## Model Overview
- **Model Type**: YOLOv8 {config['model']['variant']}
- **Creation Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Classes**: Product detection (single class)
- **Input Size**: {config['model']['image_size']}x{config['model']['image_size']} pixels

## Model Architecture
YOLOv8 is a one-stage object detection model that achieves real-time performance on a single GPU. 
It features an anchor-free design that directly predicts bounding boxes, making it more efficient 
for detecting densely packed retail products.

## Performance Metrics
- **Precision**: {model.metrics.get('precision', 'N/A'):.3f}
- **Recall**: {model.metrics.get('recall', 'N/A'):.3f}
- **mAP@0.5**: {model.metrics.get('map50', 'N/A'):.3f}
- **mAP@0.5:0.95**: {model.metrics.get('map', 'N/A'):.3f}

## Training Configuration
- **Epochs**: {config['training']['epochs']}
- **Batch Size**: {config['training']['batch_size']}
- **Learning Rate**: {config['training']['learning_rate']}
- **Image Size**: {config['model']['image_size']}
- **Confidence Threshold**: {config['model']['confidence_threshold']}
- **IoU Threshold**: {config['model']['iou_threshold']}

## Usage
This model detects retail products on store shelves. It outputs bounding boxes
around each detected product with associated confidence scores. YOLOv8's anchor-free 
approach makes it particularly effective for the dense product arrangements in retail environments.

## Deployment
The model is saved in PyTorch format (.pt) and can be converted to ONNX or TFLite
for deployment on various platforms, including Vertex AI Endpoints.
"""
    
    # Write model card
    with open('./model_card.md', 'w') as f:
        f.write(model_card_content)
    
    # Upload to GCS if path is GCS path
    if model_card_path.startswith('gs://'):
        upload_to_gcs('./model_card.md', model_card_path)
    
    logger.info(f"Model card created: {model_card_path}")

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config_path)
        logger.info(f"Loaded configuration from {args.config_path}")
        
        # Log GPU information if available
        if tf.config.list_physical_devices('GPU'):
            logger.info(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")
        else:
            logger.warning("No GPUs found. Training may be slow.")
        
        # Check for existing checkpoint if resuming
        checkpoint_path = None
        if args.resume:
            checkpoint_path = check_for_checkpoint(args.model_dir)
            if checkpoint_path:
                logger.info(f"Will resume training from {checkpoint_path}")
            else:
                logger.warning("Resume requested but no checkpoint found. Starting fresh.")
        
        # Prepare dataset
        dataset_yaml_path = prepare_dataset(args.data_dir, config)
        
        # Train YOLOv8 model
        best_model_path = train_yolov8(dataset_yaml_path, config, args.model_dir, checkpoint_path)
        
        # Complete
        logger.info(f"YOLOv8 training completed successfully. Best model saved to: {best_model_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()