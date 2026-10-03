#!/usr/bin/env python3
"""
Vertex AI LLM Task Evaluation Pipeline Launcher
Submits managed Vertex AI Evaluation Pipeline jobs for automated quantitative
benchmarking of LLMs across QA, Summarization, Text Generation, and Classification.
"""

import os
import sys
import json
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a Vertex AI LLM Quantitative Evaluation Pipeline."
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
        help="Vertex AI region",
    )
    parser.add_argument(
        "--gcs_output_dir",
        type=str,
        default=os.environ.get("GCS_OUTPUT_DIR", "gs://YOUR_GCS_BUCKET/eval_output"),
        help="GCS bucket URI for evaluation outputs and metrics",
    )
    parser.add_argument(
        "--dataset_uri",
        type=str,
        default="gs://YOUR_GCS_BUCKET/data/qa_eval_dataset.jsonl",
        help="GCS URI to evaluation dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="text-bison@001",
        help="Model ID or Vertex Endpoint resource name to evaluate",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="question-answering",
        choices=["question-answering", "summarization", "text-generation", "classification"],
        help="Evaluation task type",
    )
    parser.add_argument(
        "--machine_type",
        type=str,
        default="e2-highmem-16",
        help="Compute machine type for evaluation workers",
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
        help="Generate pipeline request specification without submitting to Vertex AI",
    )

    return parser.parse_args()


TASK_TEMPLATES = {
    "question-answering": "https://us-kfp.pkg.dev/vertex-evaluation/pipeline-templates/evaluation-llm-text-generation-pipeline/1.0.1",
    "summarization": "https://us-kfp.pkg.dev/vertex-evaluation/pipeline-templates/evaluation-llm-text-generation-pipeline/1.0.1",
    "text-generation": "https://us-kfp.pkg.dev/vertex-evaluation/pipeline-templates/evaluation-llm-text-generation-pipeline/1.0.1",
    "classification": "https://us-kfp.pkg.dev/vertex-evaluation/pipeline-templates/evaluation-llm-classification-pipeline/1.0.1",
}


def build_task_eval_spec(args):
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    job_id = f"vertex-eval-{args.task_type}-{timestamp}"

    pipeline_parameters = {
        "project": args.project_id,
        "location": args.location,
        "batch_predict_gcs_source_uris": [args.dataset_uri],
        "batch_predict_gcs_destination_output_uri": args.gcs_output_dir,
        "model_name": args.model_name,
        "evaluation_task": args.task_type,
        "batch_predict_instances_format": "jsonl",
        "batch_predict_predictions_format": "jsonl",
        "machine_type": args.machine_type,
    }

    template_uri = TASK_TEMPLATES.get(args.task_type, TASK_TEMPLATES["question-answering"])

    pipeline_spec = {
        "displayName": job_id,
        "runtimeConfig": {
            "gcsOutputDirectory": args.gcs_output_dir,
            "parameterValues": pipeline_parameters,
        },
        "templateUri": template_uri,
    }
    if args.service_account:
        pipeline_spec["runtimeConfig"]["serviceAccount"] = args.service_account

    return job_id, pipeline_spec


def main():
    args = parse_args()
    print("=" * 70)
    print("  Vertex AI Quantitative LLM Task Evaluation Pipeline")
    print("=" * 70)

    job_id, spec = build_task_eval_spec(args)

    print(f"Job Name:      {job_id}")
    print(f"Project ID:    {args.project_id}")
    print(f"Location:      {args.location}")
    print(f"Task Type:     {args.task_type}")
    print(f"Target Model:  {args.model_name}")
    print(f"Dataset URI:   {args.dataset_uri}")
    print(f"GCS Output:    {args.gcs_output_dir}\n")

    if args.dry_run:
        print("[DRY-RUN] Task Evaluation Pipeline Specification:")
        print(json.dumps(spec, indent=2))
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

        print(f"[Vertex AI] Submitting Task Evaluation job '{job_id}'...")
        job.submit(service_account=args.service_account)
        print(f"[Vertex AI] Job successfully submitted!")
        print(f"[Vertex AI] Resource: {job.resource_name}")
        print(f"[Vertex AI] Console: https://console.cloud.google.com/vertex-ai/pipelines/locations/{args.location}/runs/{job.resource_name.split('/')[-1]}?project={args.project_id}")

    except ImportError:
        print("[Error] 'google-cloud-aiplatform' is not installed.")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Failed to submit task evaluation job: {e}")
        print("Tip: Use --dry_run to validate configuration without active GCP credentials.")
        sys.exit(1)


if __name__ == "__main__":
    main()
