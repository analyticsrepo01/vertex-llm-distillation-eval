#!/usr/bin/env python3
"""
Vertex AI AutoSxS (Automated Side-by-Side) Evaluation Pipeline
Executes automated model comparison between two candidate LLMs using an LLM-as-a-Judge (Arbiter)
on Google Cloud Vertex AI Managed Pipelines.

Supported Tasks:
- Question Answering (factual accuracy, coherence, grounding)
- Text Summarization (conciseness, coverage, hallucination detection)
- General Instruction Following & Text Generation
"""

import os
import sys
import json
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch an AutoSxS Side-by-Side Model Evaluation Job on Vertex AI."
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
        help="Vertex AI region (e.g. us-central1, europe-west4)",
    )
    parser.add_argument(
        "--gcs_output_dir",
        type=str,
        default=os.environ.get("GCS_OUTPUT_DIR", "gs://YOUR_GCS_BUCKET/autosxs_eval_output"),
        help="GCS bucket URI for evaluation artifacts and metric logs",
    )
    parser.add_argument(
        "--eval_dataset_uri",
        type=str,
        default="gs://YOUR_GCS_BUCKET/data/eval_prompts.jsonl",
        help="GCS URI to evaluation dataset containing prompts and references",
    )
    parser.add_argument(
        "--model_a",
        type=str,
        default="text-bison@002",
        help="Model A identifier or Vertex Endpoint resource name (Baseline model)",
    )
    parser.add_argument(
        "--model_b",
        type=str,
        default="YOUR_FINE_TUNED_MODEL_ENDPOINT",
        help="Model B identifier or Vertex Endpoint resource name (Candidate model)",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="question-answering",
        choices=["question-answering", "summarization", "general-text-generation"],
        help="Evaluation task category",
    )
    parser.add_argument(
        "--judgement_model",
        type=str,
        default="gemini-1.5-pro",
        help="Foundation arbiter model acting as LLM-as-a-judge",
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


def build_autosxs_spec(args):
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    job_id = f"autosxs-{args.task_type}-{timestamp}"

    pipeline_parameters = {
        "project": args.project_id,
        "location": args.location,
        "evaluation_dataset": args.eval_dataset_uri,
        "model_a": args.model_a,
        "model_b": args.model_b,
        "task": args.task_type,
        "judgement_model": args.judgement_model,
        "output_dir": args.gcs_output_dir,
    }

    pipeline_spec = {
        "displayName": job_id,
        "runtimeConfig": {
            "gcsOutputDirectory": args.gcs_output_dir,
            "parameterValues": pipeline_parameters,
        },
        "templateUri": "https://us-kfp.pkg.dev/vertex-evaluation/pipeline-templates/autosxs-pipeline/1.0.0",
    }
    if args.service_account:
        pipeline_spec["runtimeConfig"]["serviceAccount"] = args.service_account

    return job_id, pipeline_spec


def main():
    args = parse_args()
    print("=" * 70)
    print("  Vertex AI AutoSxS (Automated Side-by-Side) Evaluation Pipeline")
    print("=" * 70)

    job_id, spec = build_autosxs_spec(args)

    print(f"Job Name:        {job_id}")
    print(f"Project ID:      {args.project_id}")
    print(f"Location:        {args.location}")
    print(f"Task Category:   {args.task_type}")
    print(f"Model A (Base):  {args.model_a}")
    print(f"Model B (Cand):  {args.model_b}")
    print(f"Arbiter Judge:   {args.judgement_model}")
    print(f"Eval Dataset:    {args.eval_dataset_uri}")
    print(f"GCS Output:      {args.gcs_output_dir}\n")

    if args.dry_run:
        print("[DRY-RUN] AutoSxS Pipeline Specification Payload:")
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

        print(f"[Vertex AI] Submitting AutoSxS pipeline job '{job_id}'...")
        job.submit(service_account=args.service_account)
        print(f"[Vertex AI] Job successfully submitted!")
        print(f"[Vertex AI] Resource: {job.resource_name}")
        print(f"[Vertex AI] Console: https://console.cloud.google.com/vertex-ai/pipelines/locations/{args.location}/runs/{job.resource_name.split('/')[-1]}?project={args.project_id}")

    except ImportError:
        print("[Error] 'google-cloud-aiplatform' is not installed.")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Failed to submit AutoSxS job: {e}")
        print("Tip: Use --dry_run to inspect payload format without active GCP connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
