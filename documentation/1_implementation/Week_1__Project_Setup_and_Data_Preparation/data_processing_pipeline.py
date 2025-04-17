#!/usr/bin/env python3
"""
SKU-110K Dataset Preprocessing Pipeline

This script preprocesses the SKU-110K dataset for object detection:
- Resizes images to a target size
- Normalizes pixel values to [0-1] range
- Converts annotations to appropriate format
- Creates TFRecord files for efficient training

Usage:
  python preprocess_sku110k.py --project_id=shelfscout \
                              --bucket_name=sku-110k-dataset \
                              --input_path=SKU110K_Kaggle \
                              --output_path=processed_data
"""

import os
import io
import json
import argparse
import logging
import sys
import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
from google.cloud import storage
from datetime import datetime
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('preprocessing.log')
    ]
)
logger = logging.getLogger('sku110k-preprocessing')

class SKU110KPreprocessor:
    """SKU-110K dataset preprocessor for object detection."""
    
    def __init__(self, project_id, bucket_name, input_path, output_path, target_size=(640, 640)):
        """Initialize the preprocessor.
        
        Args:
            project_id: GCP project ID
            bucket_name: GCS bucket name
            input_path: Path to raw data in GCS bucket
            output_path: Path for processed data in GCS bucket
            target_size: Target image dimensions as (width, height)
        """
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.input_path = input_path
        self.output_path = output_path
        self.target_size = target_size
        
        # Initialize GCS client
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.get_bucket(bucket_name)
        
        logger.info(f"Initialized preprocessor for project: {project_id}, bucket: {bucket_name}")
        logger.info(f"Input path: {input_path}, Output path: {output_path}")
        logger.info(f"Target image size: {target_size}")
    
    def _ensure_output_directory(self, path):
        """Ensure output directory exists in GCS."""
        blob = self.bucket.blob(f"{path}/")
        if not blob.exists():
            blob.upload_from_string('')
            logger.info(f"Created directory: gs://{self.bucket_name}/{path}/")
    
    def _list_images(self, split):
        """List all images for a given split."""
        prefix = f"{self.input_path}/images/{split}/"
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        image_blobs = [blob for blob in blobs if blob.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
        logger.info(f"Found {len(image_blobs)} images in {split} split")
        return image_blobs
    
    def _resize_and_normalize_image(self, img):
        """Resize image to target size and normalize pixel values."""
        img_resized = img.resize(self.target_size, Image.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        return img_array
    
    def _scale_bounding_boxes(self, boxes, original_size):
        """Scale bounding boxes to match the resized image dimensions."""
        orig_width, orig_height = original_size
        scaled_boxes = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            # Convert to absolute pixels in original image
            x1_px = x1 * orig_width
            y1_px = y1 * orig_height
            x2_px = x2 * orig_width
            y2_px = y2 * orig_height
            
            # Scale to new dimensions and normalize
            x1_new = x1_px * self.target_size[0] / orig_width / self.target_size[0]
            y1_new = y1_px * self.target_size[1] / orig_height / self.target_size[1]
            x2_new = x2_px * self.target_size[0] / orig_width / self.target_size[0]
            y2_new = y2_px * self.target_size[1] / orig_height / self.target_size[1]
            
            scaled_boxes.append([x1_new, y1_new, x2_new, y2_new])
        
        return scaled_boxes
    
    def _create_tf_example(self, image_array, boxes, image_id):
        """Create a TensorFlow Example from an image and its bounding boxes."""
        # Convert image to bytes
        encoded_image = tf.io.encode_jpeg(tf.cast(image_array * 255.0, tf.uint8))
        
        # Prepare features
        xmins, ymins, xmaxs, ymaxs = [], [], [], []
        for box in boxes:
            xmins.append(box[0])
            ymins.append(box[1])
            xmaxs.append(box[2])
            ymaxs.append(box[3])
        
        # Create feature dictionary
        feature = {
            'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.target_size[1]])),
            'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.target_size[0]])),
            'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_id.encode('utf8')])),
            'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_id.encode('utf8')])),
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image.numpy()])),
            'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpeg'])),
            'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
            'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
            'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
            'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
            'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'product'] * len(boxes))),
            'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1] * len(boxes))),
        }
        
        return tf.train.Example(features=tf.train.Features(feature=feature))
    
    def process_split(self, split, num_shards):
        """Process a dataset split into TFRecord shards.
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            num_shards: Number of shards to create
            
        Returns:
            GCS path to the processed TFRecord files
        """
        start_time = datetime.now()
        logger.info(f"Started processing {split} split at {start_time}")
        
        # Create output directory
        output_dir = f"{self.output_path}/{split}"
        self._ensure_output_directory(output_dir)
        
        # List all images in the split
        image_blobs = self._list_images(split)
        num_images = len(image_blobs)
        
        # Calculate items per shard
        images_per_shard = int(np.ceil(num_images / num_shards))
        logger.info(f"Processing {num_images} images into {num_shards} shards ({images_per_shard} images per shard)")
        
        # Track statistics
        processed_images = 0
        skipped_images = 0
        error_images = 0
        total_boxes = 0
        
        # Process each shard
        for shard_id in range(num_shards):
            # Set shard range
            start_idx = shard_id * images_per_shard
            end_idx = min((shard_id + 1) * images_per_shard, num_images)
            
            # Create TFRecord file locally
            output_file = f"shard_{split}_{shard_id:03d}.tfrecord"
            
            try:
                with tf.io.TFRecordWriter(output_file) as writer:
                    # Process each image in shard
                    for idx in tqdm(range(start_idx, end_idx), desc=f"Shard {shard_id+1}/{num_shards}"):
                        try:
                            # Get image data
                            image_blob = image_blobs[idx]
                            image_name = os.path.basename(image_blob.name)
                            image_id = os.path.splitext(image_name)[0]
                            
                            # Get corresponding annotation
                            annotation_path = f"{self.input_path}/labels/{split}/{image_id}.txt"
                            annotation_blob = self.bucket.blob(annotation_path)
                            
                            if not annotation_blob.exists():
                                logger.warning(f"Skipping {image_name}: no annotation")
                                skipped_images += 1
                                continue
                            
                            # Read image and annotation
                            image_data = image_blob.download_as_bytes()
                            img = Image.open(io.BytesIO(image_data))
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Read annotation and parse bounding boxes
                            annotation_text = annotation_blob.download_as_string().decode('utf-8')
                            boxes = []
                            for line in annotation_text.strip().split('\n'):
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    # Parse according to dataset format
                                    x1, y1, width, height = map(float, parts[:4])
                                    x2, y2 = x1 + width, y1 + height
                                    boxes.append([x1, y1, x2, y2])
                            
                            # Count total boxes
                            total_boxes += len(boxes)
                            
                            # Preprocess image - resize and normalize
                            img_array = self._resize_and_normalize_image(img)
                            
                            # Scale bounding boxes to new dimensions
                            scaled_boxes = self._scale_bounding_boxes(boxes, img.size)
                            
                            # Create TF Example
                            tf_example = self._create_tf_example(img_array, scaled_boxes, image_id)
                            writer.write(tf_example.SerializeToString())
                            
                            processed_images += 1
                            
                        except Exception as e:
                            logger.error(f"Error processing {image_name}: {str(e)}")
                            logger.debug(traceback.format_exc())
                            error_images += 1
                
                # Upload to GCS
                logger.info(f"Uploading shard {shard_id+1}/{num_shards} to GCS...")
                self.bucket.blob(f"{output_dir}/{output_file}").upload_from_filename(output_file)
                os.remove(output_file)
            
            except Exception as e:
                logger.error(f"Error processing shard {shard_id}: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Log statistics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Completed processing {split} split in {duration:.2f} seconds")
        logger.info(f"Processed: {processed_images}, Skipped: {skipped_images}, Errors: {error_images}")
        logger.info(f"Total bounding boxes: {total_boxes} (avg {total_boxes/processed_images:.1f} per image)")
        
        return f"gs://{self.bucket_name}/{output_dir}/"
    
    def create_metadata(self, split_shards):
        """Create metadata file with preprocessing information.
        
        Args:
            split_shards: Dictionary of splits and their shard counts
            
        Returns:
            Path to the metadata file in GCS
        """
        logger.info("Creating preprocessing metadata...")
        
        metadata = {
            "dataset": "SKU-110K",
            "preprocessing": {
                "image_size": self.target_size,
                "normalization": "0-1 scale",
                "resize_method": "LANCZOS",
                "format": "TFRecord"
            },
            "splits": {split: {"shards": shards} for split, shards in split_shards.items()},
            "created": datetime.now().strftime("%Y-%m-%d"),
            "version": "1.0"
        }
        
        # Save metadata locally and to GCS
        with open('preprocessing_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        metadata_path = f"{self.output_path}/metadata.json"
        self.bucket.blob(metadata_path).upload_from_filename('preprocessing_metadata.json')
        logger.info(f"Metadata saved to gs://{self.bucket_name}/{metadata_path}")
        
        return metadata_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess SKU-110K dataset for object detection')
    
    parser.add_argument('--project_id', required=True,
                        help='GCP project ID')
    parser.add_argument('--bucket_name', required=True,
                        help='GCS bucket name')
    parser.add_argument('--input_path', default='SKU110K_Kaggle',
                        help='Path to raw data in GCS bucket')
    parser.add_argument('--output_path', default='processed_data',
                        help='Path for processed data in GCS bucket')
    parser.add_argument('--target_width', type=int, default=640,
                        help='Target image width')
    parser.add_argument('--target_height', type=int, default=640,
                        help='Target image height')
    parser.add_argument('--train_shards', type=int, default=10,
                        help='Number of shards for train split')
    parser.add_argument('--val_shards', type=int, default=5,
                        help='Number of shards for validation split')
    parser.add_argument('--test_shards', type=int, default=5,
                        help='Number of shards for test split')
    parser.add_argument('--log_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set log level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Initialize preprocessor
    preprocessor = SKU110KPreprocessor(
        project_id=args.project_id,
        bucket_name=args.bucket_name,
        input_path=args.input_path,
        output_path=args.output_path,
        target_size=(args.target_width, args.target_height)
    )
    
    # Process each split
    split_shards = {
        'train': args.train_shards,
        'val': args.val_shards,
        'test': args.test_shards
    }
    
    for split, num_shards in split_shards.items():
        output_path = preprocessor.process_split(split, num_shards)
        logger.info(f"Processed {split} split: {output_path}")
    
    # Create metadata
    preprocessor.create_metadata(split_shards)
    
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()