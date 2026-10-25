# agritune-studio-api

Routes `/api/data/*`, `/api/train/*` and `/api/eval/*` to the services on ports 8101-8103
(server-sent event streams pass straight through) and serves the built web app from
`studio/web/dist`. Holds no ML logic. Users and workspaces get added here later.

```bash
~/.local/bin/uv run python -m agritune_studio_api      # http://127.0.0.1:8100
```
