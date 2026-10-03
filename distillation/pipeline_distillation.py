#!/usr/bin/env python3
"""
Vertex AI LLM Distillation Step-by-Step Pipeline Launcher
Orchestrates training of lightweight student language models using rationales
and predictions extracted from larger foundation teacher models on Google Cloud Vertex AI.

Reference: "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes"
"""

import os
import sys
import json
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a Step-by-Step Distillation Pipeline Job on Vertex AI."
    )
    parser.add_argument(
        "--project_id",
        type=str,
        default=os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID"),
        help="Google Cloud Project ID",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=os.environ.get("GOOGLE_CLOUD_REGION", "us-central1"),
        help="Vertex AI region (e.g. us-central1, europe-west4, asia-southeast1)",
    )
    parser.add_argument(
        "--gcs_output_dir",
        type=str,
        default=os.environ.get("GCS_OUTPUT_DIR", "gs://YOUR_GCS_BUCKET/distillation_output"),
        help="GCS bucket URI for pipeline job outputs",
    )
    parser.add_argument(
        "--train_dataset_uri",
        type=str,
        default="gs://YOUR_GCS_BUCKET/data/train_data.jsonl",
        help="GCS URI to the training dataset containing source prompts and targets",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="text-bison@002",
        help="Teacher foundation model identifier (e.g. text-bison@002, gemini-1.5-pro)",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="t5-base",
        choices=["t5-small", "t5-base", "t5-large"],
        help="Student model architecture to train",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=1000,
        help="Number of distillation training steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Student training learning rate",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weighting factor balancing label prediction loss vs rationale generation loss (0.0 to 1.0)",
    )
    parser.add_argument(
        "--service_account",
        type=str,
        default=os.environ.get("SERVICE_ACCOUNT", None),
        help="IAM service account for pipeline execution",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print the configured pipeline specification JSON without submitting to Vertex AI",
    )

    return parser.parse_args()


def build_pipeline_spec(args):
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    job_id = f"distillation-stepbystep-{args.student_model}-{timestamp}"

    pipeline_parameters = {
        "project": args.project_id,
        "location": args.location,
        "train_dataset_uri": args.train_dataset_uri,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "train_steps": args.train_steps,
        "learning_rate": args.learning_rate,
        "multitask_alpha": args.alpha,
        "output_dir": args.gcs_output_dir,
    }

    pipeline_spec = {
        "displayName": job_id,
        "runtimeConfig": {
            "gcsOutputDirectory": args.gcs_output_dir,
            "parameterValues": pipeline_parameters,
        },
        "templateUri": "https://us-kfp.pkg.dev/vertex-evaluation/pipeline-templates/distillation-step-by-step/1.0.0",
    }
    if args.service_account:
        pipeline_spec["runtimeConfig"]["serviceAccount"] = args.service_account

    return job_id, pipeline_spec


def main():
    args = parse_args()
    print("=" * 70)
    print("  Vertex AI LLM Distillation Step-by-Step Pipeline Launcher")
    print("=" * 70)

    job_id, spec = build_pipeline_spec(args)

    print(f"Job Name:        {job_id}")
    print(f"Project ID:      {args.project_id}")
    print(f"Location:        {args.location}")
    print(f"Teacher Model:   {args.teacher_model}")
    print(f"Student Model:   {args.student_model}")
    print(f"Multi-Task Alpha:{args.alpha} (Loss = α*L_task + (1-α)*L_rationale)")
    print(f"Training Steps:  {args.train_steps}")
    print(f"Train Data URI:  {args.train_dataset_uri}")
    print(f"GCS Output:      {args.gcs_output_dir}\n")

    if args.dry_run:
        print("[DRY-RUN] Pipeline Specification Payload:")
        print(json.dumps(spec, indent=2))
        print("\n[DRY-RUN] To submit this job via gcloud CLI:")
        print(f"gcloud ai pipeline-jobs create --region={args.location} --display-name={job_id} --template-uri={spec['templateUri']} --pipeline-parameters-file=<params.json>")
        return

    try:
        from google.cloud import aiplatform

        print("[Vertex AI] Initializing Vertex AI client...")
        aiplatform.init(project=args.project_id, location=args.location)

        job = aiplatform.PipelineJob(
            display_name=job_id,
            template_path=spec["templateUri"],
            pipeline_root=args.gcs_output_dir,
            parameter_values=spec["runtimeConfig"]["parameterValues"],
        )

        print(f"[Vertex AI] Submitting pipeline job '{job_id}' to Vertex AI...")
        job.submit(service_account=args.service_account)
        print(f"[Vertex AI] Job successfully submitted!")
        print(f"[Vertex AI] Pipeline resource: {job.resource_name}")
        print(f"[Vertex AI] Console Link: https://console.cloud.google.com/vertex-ai/pipelines/locations/{args.location}/runs/{job.resource_name.split('/')[-1]}?project={args.project_id}")

    except ImportError:
        print("[Error] 'google-cloud-aiplatform' is not installed.")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Failed to submit pipeline job: {e}")
        print("Tip: Run with --dry_run to validate the payload configuration without active GCP connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
