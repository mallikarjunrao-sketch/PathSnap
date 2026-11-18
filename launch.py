from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser

if __name__ == "__main__":
    folder = Path(__file__).resolve().parent / "web"
    handler = partial(SimpleHTTPRequestHandler, directory=str(folder))
    with ThreadingHTTPServer(("127.0.0.1", 8501), handler) as server:
        url = "http://127.0.0.1:8501"
        print("Open", url, "and enter the API address printed on Laptop 1.", flush=True)
        webbrowser.open(url)
        server.serve_forever()
