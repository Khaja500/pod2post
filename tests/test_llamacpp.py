"""LlamaCppGenerator against a fake llama-server: start-up, adapter switching per request, shutdown."""
import json
import socket
import sys
import textwrap

from src.infer.generator import LlamaCppGenerator

FAKE_SERVER = textwrap.dedent('''\
    import http.server, json, sys
    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    loras = [args[i + 1] for i, a in enumerate(args) if a == "--lora"]

    class H(http.server.BaseHTTPRequestHandler):
        def _send(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send({{"status": "ok"}})
            elif self.path == "/lora-adapters":
                self._send([{{"id": i, "path": p, "scale": 0.0}} for i, p in enumerate(loras)])

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                self._send({{"tokens": req["content"].split()}})
            else:
                on = [l["id"] for l in req["lora"] if l["scale"] > 0]
                self._send({{"choices": [{{"message": {{"content": json.dumps({{"lora_on": on, "temp": req["temperature"],
                             "user": req["messages"][-1]["content"]}})}}}}], "usage": {{"completion_tokens": 3}}}})

        def log_message(self, *a):
            pass

    http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
    ''')


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_llamacpp_generator_switches_adapters_per_request(tmp_path):
    (tmp_path / "server.py").write_text(FAKE_SERVER.format())
    binary = tmp_path / "llama-server"  # a shell wrapper: the venv path may contain spaces, which a shebang cannot
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{tmp_path / "server.py"}" "$@"\n')
    binary.chmod(0o755)
    sft, dpo = tmp_path / "sft-lora.gguf", tmp_path / "dpo-lora.gguf"
    gen = LlamaCppGenerator("base.gguf", [str(sft), str(dpo)], port=free_port(), binary=str(binary), temperature=0.3)
    try:
        assert gen.count("three short words") == 3
        conv = [[{"role": "user", "content": "hi"}]]
        assert json.loads(gen.chat(conv, 10)[0]) == {"lora_on": [], "temp": 0.3, "user": "hi"}
        gen.use_adapter("site", str(dpo))
        assert json.loads(gen.chat(conv, 10, temperature=0.0)[0])["lora_on"] == [1]
        gen.use_adapter(None)
        assert json.loads(gen.chat(conv * 2, 10)[1])["lora_on"] == []
        assert gen.generated_tokens == 12  # 4 requests x 3
    finally:
        gen.close()
    assert gen.proc.poll() is not None
