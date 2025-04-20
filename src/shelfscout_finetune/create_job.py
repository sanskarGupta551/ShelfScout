# Save this as create_job.py
from google.cloud import aiplatform

aiplatform.init(project='shelfscout', location='us-central1')

job = aiplatform.CustomJob(
    display_name='shelfscout-yolov8-finetuning',
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "n1-standard-8",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1
        },
        "replica_count": 1,
        "python_package_spec": {
            "executor_image_uri": "gcr.io/cloud-aiplatform/training/pytorch-gpu:latest",
            "python_module": "train",
            "package_uris": [
                "gs://sku-110k-dataset/YOLOv8_Finetuning/requirements.txt",
                "gs://sku-110k-dataset/YOLOv8_Finetuning/config.yaml",
                "gs://sku-110k-dataset/YOLOv8_Finetuning/train.py"
            ]
        }
    }],
    args=[
        "--data-dir=gs://sku-110k-dataset/SKU110K_tfrecords/",
        "--config-path=gs://sku-110k-dataset/YOLOv8_Finetuning/config.yaml",
        "--model-dir=gs://sku-110k-dataset/YOLOv8_Finetuning/Model/"
    ],
    service_account="vertexai-custom-job@shelfscout.iam.gserviceaccount.com"
)

response = job.run()
print(f"Created job: {response.name}")