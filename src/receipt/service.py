"""Local upload API. Uncalibrated predictions require review, never auto-approved."""
from __future__ import annotations
import argparse
import base64
import uuid

import requests
from receipt.inference import MAX_UPLOAD_BYTES, DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, OpenAIReceiptClient
from receipt.metrics import parse_response
from receipt.normalization import normalize_document
from receipt.schema import RECEIPT_FIELDS
from receipt.validation import validate_document


def create_app(client=None):
    from flask import Flask, jsonify, request
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 1024 * 1024
    backend = client or OpenAIReceiptClient()

    @app.errorhandler(413)
    def too_large(error):
        return jsonify({"status": "failed", "error": "upload_too_large"}), 413

    @app.get("/health")
    def health():
        ready = backend.health()
        return jsonify({"status": "ok" if ready else "degraded", "api_alive": True, "backend_ready": ready}), 200 if ready else 503

    @app.post("/extract")
    def extract():
        request_id = str(uuid.uuid4())
        try:
            fields = None
            if request.is_json:
                payload = request.get_json()
                if not isinstance(payload, dict) or not isinstance(payload.get("image"), str):
                    raise ValueError("JSON body needs a base64 image string")
                encoded = payload["image"].split(",", 1)[-1]
                if len(encoded) > 4 * MAX_UPLOAD_BYTES // 3 + 8:
                    raise ValueError("Image too large")
                image_bytes = base64.b64decode(encoded, validate=True)
                fields = payload.get("fields")
                schema_version = payload.get("schema_version", "receipt_v1")
            else:
                if "image" not in request.files:
                    raise ValueError("Multipart form needs image file")
                image_bytes = request.files["image"].read(MAX_UPLOAD_BYTES + 1)
                schema_version = request.form.get("schema_version", "receipt_v1")
                if request.form.get("fields"):
                    fields = [field.strip() for field in request.form["fields"].split(",")]
            if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
                return jsonify({"request_id": request_id, "status": "failed", "error": "upload_too_large_or_empty"}), 413
            # Invoice field contracts must be explicitly configured; no arbitrary server-side keys.
            if schema_version != "receipt_v1":
                raise ValueError("API currently exposes receipt_v1 only; invoice batches use an explicit field contract")
            if fields is not None and (not isinstance(fields, list) or not fields or any(not isinstance(field, str) or field not in RECEIPT_FIELDS for field in fields)):
                raise ValueError("Unknown requested fields")
            result = backend.extract(image_bytes, schema_version=schema_version, fields=fields)
            parsed = parse_response(result["response"])
            truncated = result.get("finish_reason") in {"length", "max_tokens"}
            if parsed is None:
                validation = {"json_valid": False, "schema_valid": False, "issues": ["invalid_json"], "applicable_checks": []}
            else:
                validation = validate_document(parsed, schema_version=schema_version, fields=fields)
                validation["json_valid"] = True
            if truncated:
                validation["issues"].append("truncated_output")
            failed = parsed is None or truncated or not validation.get("schema_valid", False)
            status = "failed" if failed else "review_required"
            # No GT, model confidence or calibrated risk policy available at request time.
            if not failed:
                validation["issues"].append("uncalibrated_requires_review")
            body = {"request_id": request_id, "schema_version": schema_version, "model_version": result.get("model_version", "unknown"),
                    "status": status, "extracted": parsed, "normalized": normalize_document(parsed) if parsed is not None else None,
                    "validation": validation, "metrics": {"latency_ms": result.get("latency_ms"), "retry_count": 0,
                    "finish_reason": result.get("finish_reason"), "image": result.get("image"), "usage": result.get("usage", {})}}
            return jsonify(body), 422 if failed else 200
        except (ValueError, TypeError) as exc:
            return jsonify({"request_id": request_id, "status": "failed", "error": "invalid_input", "message": str(exc)}), 400
        except requests.Timeout:
            return jsonify({"request_id": request_id, "status": "failed", "error": "backend_timeout"}), 504
        except requests.RequestException:
            return jsonify({"request_id": request_id, "status": "failed", "error": "backend_unavailable"}), 503
        except (KeyError, IndexError):
            return jsonify({"request_id": request_id, "status": "failed", "error": "backend_protocol_error"}), 502
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=58080)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="document-extraction")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)
    create_app(OpenAIReceiptClient(args.base_url, args.model, timeout=args.timeout, max_pixels=args.max_pixels, min_pixels=args.min_pixels)).run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
