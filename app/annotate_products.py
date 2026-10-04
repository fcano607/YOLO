"""Local browser review of COCO proposals; exports only explicitly reviewed labels."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.product_data import RAW, catalog, group_summary, identifier, read_json, review_task, save_review, write_json


def make_handler(raw, manifest, report_path):
    raw = Path(raw)
    samples = {s["image_id"]: s for s in manifest["samples"]}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, status, content, kind="application/json; charset=utf-8"):
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def sample_for(self, path, prefix):
            image_id = identifier(path[len(prefix):])
            if image_id not in samples:
                raise ValueError("Unknown image ID")
            return samples[image_id]

        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    return self.reply(200, (ROOT / "app/product_review.html").read_bytes(),
                                      "text/html; charset=utf-8")
                if path == "/api/state":
                    with lock:
                        summary = group_summary(raw, manifest)
                    return self.reply(200, {"classes": catalog()["classes"],
                                           "samples": manifest["samples"], "summary": summary})
                if path.startswith("/api/sample/"):
                    sample = self.sample_for(path, "/api/sample/")
                    draft = read_json(raw / "annotations/drafts" / (sample["image_id"] + ".json"))
                    review_path = raw / "annotations/reviewed" / (sample["image_id"] + ".json")
                    with lock:
                        review = read_json(review_path) if review_path.exists() else None
                        working_path = raw / "annotations/in_progress" / (sample["image_id"] + ".json")
                        working = read_json(working_path) if working_path.exists() else None
                    return self.reply(200, {"sample": sample, "draft": draft, "review": review, "working": working})
                if path.startswith("/api/image/"):
                    sample = self.sample_for(path, "/api/image/")
                    return self.reply(200, (raw / sample["image_path"]).read_bytes(), "image/png")
                return self.reply(404, {"error": "Unknown endpoint"})
            except (ValueError, KeyError, FileNotFoundError) as error:
                return self.reply(400, {"error": str(error)})

        def do_POST(self):
            path = urlsplit(self.path).path
            origin = self.headers.get("Origin")
            expected = "http://" + self.headers.get("Host", "")
            if origin != expected or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.reply(403, {"error": "Only local browser JSON requests are accepted"})
            try:
                if not path.startswith("/api/review/"):
                    return self.reply(404, {"error": "Unknown endpoint"})
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("Invalid request length")
                payload = json.loads(self.rfile.read(length))
                sample = self.sample_for(path, "/api/review/")
                draft = read_json(raw / "annotations/drafts" / (sample["image_id"] + ".json"))
                with lock:
                    working_path = raw / "annotations/in_progress" / (sample["image_id"] + ".json")
                    try:
                        record = save_review(raw, sample, draft, payload)
                    except ValueError as error:
                        # Preserve the attempted edit as UNREVIEWED working data, never a label.
                        write_json(working_path, {"status": "working_unreviewed", "image_id": sample["image_id"],
                                                  "payload": payload, "error": str(error)})
                        raise
                    working_path.unlink(missing_ok=True)
                    summary = group_summary(raw, manifest)
                    write_json(report_path, summary)
                return self.reply(200, {"review": record, "summary": summary})
            except (ValueError, KeyError, FileNotFoundError) as error:
                return self.reply(400, {"error": str(error)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True)
    parser.add_argument("--port", type=int, default=0, help="0 selects an unused local port")
    args = parser.parse_args()
    identifier(args.group)
    catalog()
    manifest = read_json(RAW / "sessions" / (args.group + ".json"))
    if not manifest["samples"]:
        parser.error("Group has no originals")
    for sample in manifest["samples"]:
        draft = RAW / "annotations/drafts" / (sample["image_id"] + ".json")
        if not draft.is_file():
            parser.error("Run annotate_product_drafts.py draft --group first")
    report = ROOT / "reports/data" / (review_task(manifest) + "_" + args.group + ".json")
    write_json(report, group_summary(RAW, manifest))
    # On Windows address reuse can let a second app appear to bind an occupied port.
    class LocalServer(ThreadingHTTPServer):
        allow_reuse_address = False
    server = LocalServer(("127.0.0.1", args.port), make_handler(RAW, manifest, report))
    print("Review UI: http://127.0.0.1:" + str(server.server_port), flush=True)
    print("No predictions are training labels until you confirm and save them. Ctrl+C stops server.", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
