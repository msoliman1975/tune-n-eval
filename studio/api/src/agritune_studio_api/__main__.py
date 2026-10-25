"""Gateway: proxies service APIs and serves the web app."""

from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

SERVICES = {"data": "http://127.0.0.1:8101", "train": "http://127.0.0.1:8102", "eval": "http://127.0.0.1:8103"}
WEB = Path(__file__).resolve().parents[3] / "web" / "dist"

app = FastAPI(title="agritune-studio", version="0.1.0")
client = httpx.AsyncClient(timeout=httpx.Timeout(30, read=None))


@app.get("/api/health")
async def health() -> dict:
    out = {}
    for name, url in SERVICES.items():
        try:
            out[name] = (await client.get(f"{url}/openapi.json", timeout=2)).status_code == 200
        except httpx.HTTPError:
            out[name] = False
    return out


@app.api_route("/api/{service}/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy(service: str, path: str, request: Request):
    if service not in SERVICES:
        return JSONResponse({"detail": f"unknown service {service}"}, status_code=404)
    req = client.build_request(request.method, f"{SERVICES[service]}/{path}", params=request.query_params,
                               content=await request.body(), headers={"content-type": request.headers.get("content-type", "")})
    try:
        resp = await client.send(req, stream=True)
    except httpx.ConnectError:
        return JSONResponse({"detail": f"{service} service is not running"}, status_code=503)
    if resp.headers.get("content-type", "").startswith("text/event-stream"):
        return StreamingResponse(resp.aiter_raw(), media_type="text/event-stream",
                                 background=BackgroundTask(resp.aclose), headers={"cache-control": "no-cache"})
    body = await resp.aread()
    await resp.aclose()
    return Response(body, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


if WEB.exists():
    app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

    @app.get("/{path:path}")
    async def spa(path: str):
        return FileResponse(WEB / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8100)
