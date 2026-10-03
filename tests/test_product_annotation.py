"""Engineering fixtures only: label geometry, review gate, provenance and HTTP scope."""

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.annotate_products import make_handler
from app.product_data import digest, group_summary, read_json, save_review, validate_polygon, write_json


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="yolo_annotation_test_")
        self.raw = Path(self.temp.name)
        self.image = self.raw / "frames/test_001.png"
        self.image.parent.mkdir()
        self.image.write_bytes(b"synthetic fixture; never a project dataset")
        self.points = [[0.1, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]]
        self.sample = {"image_id": "test_001", "group_id": "test",
                       "image_path": "frames/test_001.png", "image_sha256": digest(self.image),
                       "width": 1280, "height": 720}
        self.draft = {"candidates": [{"id": "c000", "points": self.points}]}
        write_json(self.raw / "annotations/drafts/test_001.json", self.draft)
        self.manifest = {"group_id": "test", "samples": [self.sample]}
        self.payload = {"confirmed": True, "status": "reviewed", "active_seconds": 2.5,
                        "polygons": [{"class_id": 1, "points": self.points,
                                      "source_candidate_id": "c000"}]}

    def tearDown(self):
        self.temp.cleanup()

    def save(self, payload=None):
        return save_review(self.raw, self.sample, self.draft, payload or self.payload)

    def test_export_has_product_id_and_only_normalized_geometry(self):
        record = self.save()
        tokens = (self.raw / "annotations/reviewed/test_001.txt").read_text().split()
        self.assertEqual(tokens[0], "1")
        self.assertEqual(len(tokens), 9)  # class + four xy pairs; no confidence.
        self.assertEqual(record["stats"]["retained_unchanged"], 1)
        self.assertEqual(group_summary(self.raw, self.manifest)["instances"], 1)

    def test_unreviewed_or_unassigned_draft_cannot_become_a_label(self):
        for patch in ({"confirmed": False}, {"polygons": [
                {"class_id": None, "points": self.points, "source_candidate_id": "c000"}]}):
            with self.assertRaises(ValueError):
                self.save({**self.payload, **patch})
        self.assertFalse((self.raw / "annotations/reviewed").exists())
        with self.assertRaises(ValueError):
            save_review(self.raw, self.sample, self.draft, [])

    def test_negative_is_explicit_and_excluded_has_no_label(self):
        with self.assertRaises(ValueError):
            self.save({**self.payload, "polygons": []})
        self.save({**self.payload, "status": "negative", "polygons": []})
        label = self.raw / "annotations/reviewed/test_001.txt"
        self.assertEqual(label.read_text(), "")
        self.save({**self.payload, "status": "excluded", "polygons": [], "reason": "ambiguous"})
        self.assertFalse(label.exists())
        self.assertEqual(group_summary(self.raw, self.manifest)["excluded_images"], 1)

    def test_repeated_save_replaces_instead_of_appending(self):
        self.save()
        self.save()
        self.assertEqual(len((self.raw / "annotations/reviewed/test_001.txt").read_text().splitlines()), 1)

    def test_near_duplicate_manual_closing_point_does_not_block_save(self):
        # Minimized from the actual failing browser request, not a made-up crossing.
        points = [[0.3257599157102251, 0.226234190126479],
                  [0.3575836611202618, 0.1246883579618863],
                  [0.33228783858449834, 0.18706650346797227],
                  [0.3265758967312051, 0.23058615531075752]]
        record = self.save({**self.payload, "polygons": [
            {"class_id": 2, "points": points, "source_candidate_id": None}]})
        self.assertEqual(len(record["polygons"][0]["points"]), 3)
        self.assertEqual(record["polygons"][0]["geometry_cleanup"],
                         "removed_near_duplicate_closing_vertex_within_5_original_pixels")

    def test_real_crossing_reports_the_polygon_and_edges(self):
        payload = {**self.payload, "polygons": [
            {"class_id": 1, "points": self.points, "source_candidate_id": None},
            {"class_id": 0, "points": [[0, 0], [1, 1], [0, 1], [0.8, 0.2]],
             "source_candidate_id": None}]}
        with self.assertRaisesRegex(ValueError, "\u7b2c 2.*edges"):
            self.save(payload)
        self.assertFalse((self.raw / "annotations/reviewed/test_001.txt").exists())

    def test_failed_http_save_preserves_work_without_exporting_bad_labels(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(
            self.raw, self.manifest, self.raw / "report.json"))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = "http://127.0.0.1:" + str(server.server_port)
        invalid = {**self.payload, "polygons": [
            {"class_id": 0, "points": [[0, 0], [1, 1], [0, 1], [0.8, 0.2]],
             "source_candidate_id": None}]}
        try:
            def submit(payload):
                request = Request(base + "/api/review/test_001", data=json.dumps(payload).encode(),
                                  headers={"Origin": base, "Content-Type": "application/json"})
                return urlopen(request, timeout=3)
            with self.assertRaises(HTTPError) as caught:
                submit(invalid)
            self.assertEqual(caught.exception.code, 400)
            working = self.raw / "annotations/in_progress/test_001.json"
            self.assertEqual(read_json(working)["payload"]["polygons"], invalid["polygons"])
            self.assertFalse((self.raw / "annotations/reviewed/test_001.txt").exists())
            with urlopen(base + "/api/sample/test_001", timeout=3) as response:
                self.assertEqual(json.load(response)["working"]["status"], "working_unreviewed")
            with submit(self.payload) as response:
                self.assertEqual(response.status, 200)
            self.assertFalse(working.exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_geometry_rejects_nan_out_of_bounds_degenerate_and_crossing(self):
        invalid = [
            [[0, 0], [1, 1]],
            [[0, 0], [1, 0], [float("nan"), 1]],
            [[0, 0], [1.01, 0], [0, 1]],
            [[0, 0], [0.5, 0.5], [1, 1]],
            [[0, 0], [1, 1], [0, 1], [1, 0.2]],
        ]
        for points in invalid:
            with self.subTest(points=points), self.assertRaises(ValueError):
                validate_polygon(points)

    def test_changed_original_is_rejected(self):
        self.image.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.save()

    def test_edited_manual_and_discarded_stats(self):
        moved = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]]
        payload = {**self.payload, "polygons": [
            {"class_id": 0, "points": moved, "source_candidate_id": "c000"},
            {"class_id": 2, "points": self.points, "source_candidate_id": None}]}
        record = self.save(payload)
        self.assertEqual(record["stats"]["retained_boundary_edited"], 1)
        self.assertEqual(record["stats"]["manual_polygons"], 1)
        self.assertEqual(group_summary(self.raw, self.manifest)["manual_polygon_fraction"], 0.5)

    def test_duplicate_candidate_and_invalid_class_are_rejected(self):
        with self.assertRaises(ValueError):
            self.save({**self.payload, "polygons": self.payload["polygons"] * 2})
        with self.assertRaises(ValueError):
            self.save({**self.payload, "polygons": [{"class_id": 41, "points": self.points}]})

    def test_http_review_requires_same_origin_and_known_image(self):
        report = self.raw / "test_report.json"
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.raw, self.manifest, report))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = "http://127.0.0.1:" + str(server.server_port)
        try:
            body = json.dumps(self.payload).encode()
            for origin, expected in (("https://example.com", 403), (base, 200)):
                request = Request(base + "/api/review/test_001", data=body,
                                  headers={"Origin": origin, "Content-Type": "application/json"})
                try:
                    with urlopen(request, timeout=3) as response:
                        self.assertEqual(response.status, expected)
                except HTTPError as error:
                    self.assertEqual(error.code, expected)
            self.assertEqual(read_json(report)["reviewed_target_images"], 1)
            with self.assertRaises(HTTPError):
                urlopen(base + "/api/image/not_a_sample", timeout=3)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
