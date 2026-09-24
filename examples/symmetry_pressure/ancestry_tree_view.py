"""View step for the interactive ancestry-tree viewer.

Turns a bundle produced by `ancestry_tree_prepare.py` (manifest.json +
images/) into an interactive page.

Default mode: a single self-contained .html file with every image inlined
as a base64 data: URI and the CSS/JS inlined too -- reopenable via plain
`file://` with no server, so it can be regenerated, reopened, or shared
freely without recomputing anything from the prepare step.

    python ancestry_tree_view.py <bundle_dir>

`--serve` mode: starts a local HTTP server rooted at the bundle directory
instead (keeps images as separate files on disk) -- useful for iterating on
ancestry_tree_static/*.js/*.css themselves without re-embedding each time.

    python ancestry_tree_view.py <bundle_dir> --serve
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import shutil
import threading
import webbrowser
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "ancestry_tree_static"
DATA_PLACEHOLDER = "<!--ANCESTRY_DATA-->"


def _mime_for(path: Path) -> str:
    return {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        path.suffix.lstrip(".").lower(), "application/octet-stream"
    )


def _inline_images(manifest: dict, bundle_dir: Path) -> dict:
    """Return a deep-ish copy of manifest with genome_image/phenotype_image
    paths replaced by base64 data: URIs."""
    cache: dict[str, str] = {}

    def to_data_uri(rel_path: str) -> str:
        if rel_path in cache:
            return cache[rel_path]
        full = bundle_dir / rel_path
        data = base64.b64encode(full.read_bytes()).decode("ascii")
        uri = f"data:{_mime_for(full)};base64,{data}"
        cache[rel_path] = uri
        return uri

    new_manifest = json.loads(json.dumps(manifest))  # cheap deep copy
    for node in new_manifest["nodes"]:
        if node.get("genome_image"):
            node["genome_image"] = to_data_uri(node["genome_image"])
        if node.get("phenotype_image"):
            node["phenotype_image"] = to_data_uri(node["phenotype_image"])
    return new_manifest


def _embed_script_json(obj: dict) -> str:
    """Safely embed a JSON object as a <script> literal (escape `</script>`)."""
    text = json.dumps(obj)
    return text.replace("</script", "<\\/script")


def build_embedded_html(bundle_dir: Path, manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text())
    inlined = _inline_images(manifest, bundle_dir)

    html = (STATIC_DIR / "ancestry_tree_viewer.html").read_text()
    css = (STATIC_DIR / "ancestry_tree_viewer.css").read_text()
    js = (STATIC_DIR / "ancestry_tree_viewer.js").read_text()

    html = html.replace(
        '<link rel="stylesheet" href="ancestry_tree_viewer.css">',
        f"<style>\n{css}\n</style>",
    )
    html = html.replace(
        '<script src="ancestry_tree_viewer.js"></script>',
        f"<script>\n{js}\n</script>",
    )
    data_script = f"<script>window.ANCESTRY_MANIFEST = {_embed_script_json(inlined)};</script>"
    html = html.replace(DATA_PLACEHOLDER, data_script)
    return html


def run_serve_mode(bundle_dir: Path, port: int, open_browser: bool) -> None:
    for name in ("ancestry_tree_viewer.html", "ancestry_tree_viewer.css", "ancestry_tree_viewer.js"):
        dest = bundle_dir / name
        if not dest.exists():
            shutil.copy(STATIC_DIR / name, dest)

    handler = http.server.SimpleHTTPRequestHandler
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), lambda *a, **kw: handler(*a, directory=str(bundle_dir), **kw))
    url = f"http://127.0.0.1:{httpd.server_address[1]}/ancestry_tree_viewer.html"
    print(f"Serving {bundle_dir} at {url} (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Default: <bundle_dir>/ancestry_tree.html")
    parser.add_argument("--open", dest="open_browser", action="store_true", default=True)
    parser.add_argument("--no-open", dest="open_browser", action="store_false")
    parser.add_argument("--serve", nargs="?", const=0, type=int, default=None,
                        help="Serve via local HTTP server instead of embedding (optional port, default random)")
    args = parser.parse_args()

    bundle_dir = args.bundle_dir.resolve()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found at {manifest_path} -- run ancestry_tree_prepare.py first")

    if args.serve is not None:
        run_serve_mode(bundle_dir, args.serve, args.open_browser)
        return

    out_path = args.out or (bundle_dir / "ancestry_tree.html")
    html = build_embedded_html(bundle_dir, manifest_path)
    out_path.write_text(html)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.1f} MB)")

    if args.open_browser:
        webbrowser.open(f"file://{out_path}")


if __name__ == "__main__":
    main()
