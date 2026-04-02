#!/usr/bin/env python3
"""Batch inference using vLLM API server for CG-PRM teacher model."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

from cg_prm.data.schema import read_jsonl, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


async def send_request(
    client: httpx.AsyncClient,
    server_url: str,
    request_data: dict[str, Any],
    timeout: int = 300,
) -> dict[str, Any] | None:
    """Send single inference request to vLLM server."""
    try:
        # Extract prompt string from CG-PRM format (prompt is a dict with 'user' key)
        prompt_text = request_data["prompt"]
        if isinstance(prompt_text, dict):
            prompt_text = prompt_text.get("user", "")
        
        response = await client.post(
            f"{server_url}/generate",
            json={
                "prompt": prompt_text,
                "max_tokens": request_data.get("max_tokens", 1024),
                "temperature": request_data.get("temperature", 0.2),
                "top_p": request_data.get("top_p", 0.95),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        
        # Handle vLLM v0.17.x response format (text is a list)
        raw_text = result.get("text", "")
        if isinstance(raw_text, list) and len(raw_text) > 0:
            generated_text = raw_text[0]
        else:
            generated_text = raw_text if isinstance(raw_text, str) else ""
        
        # Get original request data if available
        original = request_data.get("_original", {})
        example_data = original.get("example", {})
        config = original.get("config", {})
        
        return {
            "request_id": request_data["request_id"],
            "example_id": request_data["example_id"],
            "benchmark": example_data.get("benchmark", "gqa" if "clevr" in str(request_data.get("prompt", "")) else "docvqa"),
            "image_path": example_data.get("image_path", ""),
            "question": example_data.get("question", ""),
            "answer": example_data.get("answer", ""),
            "generated_text": generated_text,  # This will be converted to raw_text when saving
            "finish_reason": result.get("finish_reason", "unknown"),
            "config": config,
        }
    except Exception as e:
        print(f"Error for request {request_data['request_id']}: {e}")
        return None


async def batch_inference(
    requests_path: Path,
    output_path: Path,
    server_url: str,
    batch_size: int = 64,
    max_concurrent: int = 32,
    checkpoint_interval: int = 1000,
) -> int:
    """Process teacher requests in batches through vLLM server."""
    
    # Load requests
    print(f"Loading requests from {requests_path}")
    requests_list = list(read_jsonl(requests_path))
    print(f"Loaded {len(requests_list)} requests")
    
    # Convert CG-PRM format requests to vLLM format if needed
    vllm_requests = []
    request_lookup = {}  # Map request_id to original request data
    for req in requests_list:
        # Check if this is CG-PRM format (has 'example' key)
        if "example" in req:
            vllm_req = {
                "request_id": req["example"]["example_id"],
                "example_id": req["example"]["example_id"],
                "prompt": req["prompt"]["user"],
                "max_tokens": req["config"].get("max_tokens", 1024),
                "temperature": req["config"].get("temperature", 0.2),
                "top_p": req["config"].get("top_p", 0.95),
                # Store original data for later
                "_original": req,
            }
            request_lookup[vllm_req["request_id"]] = req
        else:
            # Already in vLLM format
            vllm_req = {
                "request_id": req.get("request_id", req.get("example_id")),
                "example_id": req.get("example_id"),
                "prompt": req["prompt"],
                "max_tokens": req.get("max_tokens", 1024),
                "temperature": req.get("temperature", 0.2),
                "top_p": req.get("top_p", 0.95),
                "_original": req,
            }
            request_lookup[vllm_req["request_id"]] = req
        vllm_requests.append(vllm_req)
    
    requests = vllm_requests
    
    # Check for existing checkpoint
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    completed_ids = set()
    outputs = []
    
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}")
        checkpoint_data = json.loads(checkpoint_path.read_text())
        outputs = checkpoint_data.get("outputs", [])
        completed_ids = set(checkpoint_data.get("completed_ids", []))
        print(f"Resuming from checkpoint: {len(completed_ids)} completed")
    
    # Filter out completed requests
    pending_requests = [r for r in requests if r["request_id"] not in completed_ids]
    print(f"Pending requests: {len(pending_requests)}")
    
    if not pending_requests:
        print("All requests already completed!")
        write_jsonl(output_path, outputs)
        return len(outputs)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process in batches
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(req):
        async with semaphore:
            return await send_request(client, server_url, req)
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10)) as client:
        for batch_start in tqdm(range(0, len(pending_requests), batch_size), desc="Batches"):
            batch_end = min(batch_start + batch_size, len(pending_requests))
            batch = pending_requests[batch_start:batch_end]
            
            # Process batch concurrently
            tasks = [process_with_semaphore(req) for req in batch]
            results = await asyncio.gather(*tasks)
            
            # Collect results
            for result in results:
                if result is not None:
                    outputs.append(result)
                    completed_ids.add(result["request_id"])
            
            # Save checkpoint
            if (batch_start // batch_size + 1) % (checkpoint_interval // batch_size) == 0:
                checkpoint_data = {
                    "outputs": outputs,
                    "completed_ids": list(completed_ids),
                    "total": len(requests),
                }
                checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))
                print(f"\nCheckpoint saved: {len(completed_ids)}/{len(requests)}")
    
    # Remove duplicates (in case of checkpoint overlap)
    seen = set()
    unique_outputs = []
    for out in outputs:
        if out["request_id"] not in seen:
            seen.add(out["request_id"])
            # Convert to TeacherOutput format expected by pipeline
            teacher_output = {
                "request_id": out["request_id"],
                "example_id": out["example_id"],
                "benchmark": out.get("benchmark", "unknown"),
                "image_path": out.get("image_path", ""),
                "question": out.get("question", ""),
                "answer": out.get("answer", ""),
                "raw_text": out.get("generated_text", ""),  # Convert generated_text to raw_text
                "finish_reason": out.get("finish_reason", "unknown"),
                "config": out.get("config", {}),
            }
            unique_outputs.append(teacher_output)
    
    # Save final outputs
    write_jsonl(output_path, unique_outputs)
    
    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    print(f"\nInference complete!")
    print(f"Processed: {len(unique_outputs)}/{len(requests)}")
    print(f"Output saved to: {output_path}")
    
    return len(unique_outputs)


def convert_requests_for_vllm(
    teacher_requests_path: Path,
    vllm_requests_path: Path,
) -> int:
    """Convert CG-PRM teacher requests to vLLM format."""
    
    print(f"Converting requests from {teacher_requests_path}")
    teacher_requests = list(read_jsonl(teacher_requests_path))
    
    vllm_requests = []
    for req in teacher_requests:
        vllm_req = {
            "request_id": req.get("request_id", req.get("example_id")),
            "example_id": req["example_id"],
            "prompt": req["prompt"],
            "max_tokens": req.get("generation_config", {}).get("max_tokens", 1024),
            "temperature": req.get("generation_config", {}).get("temperature", 0.2),
            "top_p": req.get("generation_config", {}).get("top_p", 0.95),
        }
        vllm_requests.append(vllm_req)
    
    write_jsonl(vllm_requests_path, vllm_requests)
    print(f"Converted {len(vllm_requests)} requests to vLLM format")
    print(f"Saved to: {vllm_requests_path}")
    
    return len(vllm_requests)


def convert_outputs_to_teacher_format(
    vllm_outputs_path: Path,
    teacher_outputs_path: Path,
    original_requests_path: Path,
) -> int:
    """Convert vLLM outputs back to CG-PRM teacher output format."""
    
    print(f"Converting outputs from {vllm_outputs_path}")
    vllm_outputs = {o["example_id"]: o for o in read_jsonl(vllm_outputs_path)}
    original_requests = {r["example_id"]: r for r in read_jsonl(original_requests_path)}
    
    teacher_outputs = []
    for example_id, vllm_out in vllm_outputs.items():
        orig_req = original_requests.get(example_id, {})
        
        # Convert to CG-PRM teacher output format
        teacher_out = {
            "example_id": vllm_out["example_id"],
            "request_id": vllm_out["request_id"],
            "benchmark": orig_req.get("benchmark", "gqa" if "clevr" in str(original_requests_path) else "docvqa"),
            "image_path": orig_req.get("image_path", ""),
            "question": orig_req.get("question", ""),
            "gold_answer": orig_req.get("gold_answer", ""),
            "generated_text": vllm_out["generated_text"],
            "finish_reason": vllm_out["finish_reason"],
            "metadata": {
                "model": "Qwen3VL-4B",
                "temperature": vllm_out.get("temperature", 0.2),
            },
        }
        teacher_outputs.append(teacher_out)
    
    write_jsonl(teacher_outputs_path, teacher_outputs)
    print(f"Converted {len(teacher_outputs)} outputs to teacher format")
    print(f"Saved to: {teacher_outputs_path}")
    
    return len(teacher_outputs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch inference using vLLM server for CG-PRM teacher model."
    )
    parser.add_argument("--requests", required=True, help="Teacher requests JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--server-url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-concurrent", type=int, default=32, help="Max concurrent requests")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Save checkpoint every N samples")
    parser.add_argument("--mode", choices=["convert", "infer", "convert-back"], default="infer")
    parser.add_argument("--original-requests", help="Original requests (for convert-back mode)")
    args = parser.parse_args()
    
    requests_path = Path(args.requests)
    output_path = Path(args.output)
    
    if args.mode == "convert":
        # Convert CG-PRM format to vLLM format
        vllm_requests_path = output_path
        return convert_requests_for_vllm(requests_path, vllm_requests_path)
    
    elif args.mode == "infer":
        # Run batch inference
        return asyncio.run(batch_inference(
            requests_path=requests_path,
            output_path=output_path,
            server_url=args.server_url,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            checkpoint_interval=args.checkpoint_interval,
        ))
    
    elif args.mode == "convert-back":
        # Convert vLLM outputs back to CG-PRM format
        if not args.original_requests:
            print("ERROR: --original-requests required for convert-back mode")
            return 1
        original_requests_path = Path(args.original_requests)
        return convert_outputs_to_teacher_format(
            output_path,
            requests_path,  # This is actually the teacher output path in convert-back mode
            original_requests_path,
        )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
