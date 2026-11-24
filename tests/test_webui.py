"""API-level tests for the arrangement audition interface.

The browser page is driven through the same HTTP API the page uses: a
session is created, performance input is actually played into the kernel,
following is toggled, and the kernel variants (before/after the candidate
update speedup) are compared on the same trajectory. Kernel failures must
surface as JSON errors the page can show.
"""
import json
import threading
import time
import unittest
import urllib.request
import urllib.error

from score_follow.trajectory import GenConfig, generate
from score_follow.score import demo_score
from score_follow.webui import AuditionServer


def _post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _get(port, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


class WebUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = AuditionServer(0)  # ephemeral port
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        status, _ = _post(self.port, "/api/session",
                          {"score": "demo", "kernel": "optimized", "follow": True})
        self.assertEqual(status, 200)

    # ------------------------------------------------------------------ #
    def test_page_is_served_with_kernel_driven_controls(self):
        status, html = _get(self.port, "/")
        self.assertEqual(status, 200)
        for hook in ('id="follow"', 'id="staff"', 'id="play"', 'id="kernel"',
                     'id="traj"', "/api/tick", "/api/event"):
            self.assertIn(hook, html)

    def test_trajectories_are_listed_and_loadable(self):
        status, body = _get(self.port, "/api/trajectories")
        self.assertEqual(status, 200)
        items = json.loads(body)["items"]
        self.assertTrue(any("benchmark" in n for n in items))
        status, data = _get(self.port, "/api/trajectory?name=" + items[0])
        self.assertEqual(status, 200)
        self.assertIn("events", json.loads(data))
        # path traversal is refused: only listed names load
        status, _ = _get(self.port, "/api/trajectory?name=../pytest.ini")
        self.assertEqual(status, 404)

    def test_performance_input_drives_cursor_and_accompaniment(self):
        score = demo_score()
        traj = generate(score, GenConfig(seed=3, timing_jitter=0.005))
        onsets = [e for e in traj["events"] if e["kind"] == "on"][:8]
        beats = []
        played = []
        for ev in onsets:
            status, payload = _post(self.port, "/api/event",
                                    {"kind": "on", "pitch": ev["pitch"]})
            self.assertEqual(status, 200)
            beats.append(payload["cursor"]["best_beat"])
            played += [a for a in payload["actions"] if a["action"] == "play"]
            _post(self.port, "/api/event", {"kind": "off", "pitch": ev["pitch"]})
            status, tick = _post(self.port, "/api/tick", {})
            self.assertEqual(status, 200)
            played += [a for a in tick["actions"] if a["action"] == "play"]
        # the kernel's cursor advanced with the played input
        self.assertGreater(beats[-1], beats[0])
        # ...and the kernel sounded accompaniment notes through the API
        self.assertTrue(played)

    def test_follow_toggle_switches_playhead_mode(self):
        status, off = _post(self.port, "/api/follow", {"enabled": False})
        self.assertEqual(status, 200)
        self.assertFalse(off["follow"])
        # feed one event so the tracker has a reading, then compare modes
        _post(self.port, "/api/event", {"kind": "on", "pitch": 60})
        status, tick = _post(self.port, "/api/tick", {})
        self.assertEqual(status, 200)
        # fixed mode: the playhead marches with the wall clock, not the cursor
        self.assertNotEqual(tick["playhead_beat"], tick["cursor"]["best_beat"])
        status, on = _post(self.port, "/api/follow", {"enabled": True})
        self.assertTrue(on["follow"])
        status, tick = _post(self.port, "/api/tick", {})
        self.assertEqual(tick["playhead_beat"], tick["cursor"]["best_beat"])

    def test_kernel_variants_compared_on_same_input(self):
        # the page's kernel selector A/Bs the accompaniment change; here the
        # same performance is played through both variants via the API
        score = demo_score()
        traj = generate(score, GenConfig(seed=5, timing_jitter=0.005))
        onsets = [e for e in traj["events"] if e["kind"] == "on"][:10]
        cursors = {}
        for kernel in ("baseline", "optimized"):
            status, _ = _post(self.port, "/api/session",
                              {"score": "demo", "kernel": kernel, "follow": True})
            self.assertEqual(status, 200)
            series = []
            for ev in onsets:
                _post(self.port, "/api/event", {"kind": "on", "pitch": ev["pitch"]})
                status, payload = _post(self.port, "/api/tick", {})
                self.assertEqual(status, 200)
                series.append(payload["cursor"]["best_beat"])
                _post(self.port, "/api/event", {"kind": "off", "pitch": ev["pitch"]})
            cursors[kernel] = series
        # before/after the speedup, the cursor tracks the same positions
        for a, b in zip(cursors["baseline"], cursors["optimized"]):
            self.assertAlmostEqual(a, b, delta=0.5)

    def test_bad_input_and_kernel_failure_report_back(self):
        status, body = _post(self.port, "/api/event", {"kind": "bang", "pitch": 60})
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        status, body = _post(self.port, "/api/event", {"kind": "on", "pitch": 999})
        self.assertEqual(status, 400)
        status, body = _post(self.port, "/api/nope", {})
        self.assertEqual(status, 404)
        # a kernel exception must come back as a JSON error for the page
        kernel = self.server.kernel
        boom = lambda ev: (_ for _ in ()).throw(RuntimeError("boom"))
        kernel.estimator.handle_event, orig = boom, kernel.estimator.handle_event
        try:
            status, body = _post(self.port, "/api/event", {"kind": "on", "pitch": 60})
            self.assertEqual(status, 500)
            self.assertIn("boom", body["error"])
        finally:
            kernel.estimator.handle_event = orig
        # after the failure the kernel still answers (page can reset/resume)
        status, _ = _post(self.port, "/api/reset", {})
        self.assertEqual(status, 200)
        status, _ = _post(self.port, "/api/tick", {})
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
