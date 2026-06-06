# Run @askaimbot 24/7 (free options)

Your bot uses **polling** (always-on process). You need a host that stays running — not serverless that sleeps after each request.

Set these environment variables on any host:

- `TELEGRAM_TOKEN`
- `GEMINI_API_KEY`

---

## Best free 24/7: Oracle Cloud Always Free

- **Cost:** $0/month (always-free VM)
- **Pros:** Real VPS, does not sleep, good for polling bots
- **Cons:** Signup + setup ~30–60 min (card for verification, often not charged)

Rough steps:

1. Create account at https://www.oracle.com/cloud/free/
2. Create an **Ampere** VM (Ubuntu 22.04), open port 22
3. SSH in, install Python 3.11+, clone your repo
4. `pip install -r requirements.txt`
5. Set env vars in `~/.bashrc` or a systemd service
6. Run with **systemd** so it restarts on crash:

```ini
# /etc/systemd/system/askaimbot.service
[Unit]
Description=askaimbot Telegram
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/AIMBOT
Environment=TELEGRAM_TOKEN=your_token
Environment=GEMINI_API_KEY=your_key
ExecStart=/usr/bin/python3 aimbot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable askaimbot
sudo systemctl start askaimbot
```

**Stop the bot on your laptop** when the server runs — only one poller per bot token.

---

## Fastest tonight: Render (free tier)

- **Cost:** $0 starter tier
- **Pros:** Git push deploy, easy UI
- **Cons:** Free web services **sleep** when idle; use a **Background Worker** (not a web service) for polling

1. Push `AIMBOT` to a **private** GitHub repo (no keys in code)
2. https://render.com → New → **Background Worker**
3. Build: `pip install -r requirements.txt`
4. Start: `python aimbot.py`
5. Add env vars in Render dashboard

---

## Other options

| Host | Free? | Good for polling? |
|------|-------|-------------------|
| **Fly.io** | Small free allowance | Yes, with `fly.toml` + always-on machine |
| **Railway** | Limited trial credit | Yes, while credit lasts |
| **Google Cloud e2-micro** | Free tier 1 VM | Yes, similar to Oracle |
| **Your PC** | Free | Yes, but stops when PC sleeps |

---

## Checklist before deploy

- [ ] `requirements.txt` installed on server
- [ ] Env vars set on host (never commit keys)
- [ ] Bot stopped on laptop (avoid double polling)
- [ ] Test `/start` in Telegram after deploy
