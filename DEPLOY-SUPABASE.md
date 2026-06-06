# Deploy @askaimbot on Supabase (24/7, no Oracle)

## Important: how Supabase hosting works

Supabase does **not** run your Python file (`aimbot.py`) like a VPS.

| What Supabase gives you | What your bot needs |
|-------------------------|---------------------|
| **Edge Functions** (serverless) | A URL Telegram can POST updates to |
| **Database** (optional, later) | Citizen memory, knowledge files |

We use **webhooks** instead of **polling**:

- **Polling** (your laptop / `aimbot.py`): your script constantly asks Telegram “any messages?”
- **Webhook** (Supabase): Telegram pushes each message to your function URL → **stays “live” 24/7 without a running PC**

The code for Supabase lives in: `supabase/functions/askaimbot/index.ts`  
(Same AIM personality + Gemini — TypeScript on Deno.)

Later you can add **Supabase Database** for memory and your Nigerian knowledge files.

---

## What you need before starting

- [ ] GitHub account (free)
- [ ] [Supabase](https://supabase.com) account (free)
- [ ] Telegram bot token from [@BotFather](https://t.me/BotFather)
- [ ] Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
- [ ] **Stop** `python aimbot.py` on your laptop when Supabase is live (only one mode at a time)

---

## Step 1 — Create a Supabase project

1. Go to https://supabase.com/dashboard  
2. **New project**  
3. Name: `askaimbot` (or anything)  
4. Set a **database password** (save it — you need it for dashboard; the bot does not use DB yet)  
5. Pick a **region** close to you (e.g. EU or US)  
6. Wait until the project status is **Active**

Copy from **Project Settings → General**:

- **Project URL** → `https://YOUR_PROJECT_REF.supabase.co`  
- **Project ref** → `YOUR_PROJECT_REF` (short id in the URL)

---

## Step 2 — Install Supabase CLI (Windows)

In **PowerShell** (run as normal user):

```powershell
# Option A — Scoop (if you use Scoop)
scoop bucket add supabase https://github.com/supabase/scoop-bucket.git
scoop install supabase

# Option B — npm (if you have Node.js)
npm install -g supabase
```

Check:

```powershell
supabase --version
```

Login:

```powershell
supabase login
```

(Browser opens — approve access.)

---

## Step 3 — Link this folder to your project

```powershell
cd C:\Users\Admin\Desktop\AIMBOT
supabase link --project-ref YOUR_PROJECT_REF
```

Enter your database password when asked.

---

## Step 4 — Set secrets (API keys)

Pick a random **webhook secret** (e.g. a long password you make up). Telegram will call your function with `?secret=...`.

```powershell
supabase secrets set TELEGRAM_BOT_TOKEN="paste_bot_token_from_botfather"
supabase secrets set GEMINI_API_KEY="paste_gemini_key"
supabase secrets set FUNCTION_SECRET="paste_a_long_random_secret"
```

**Never commit these to GitHub.**

---

## Step 5 — Deploy the Edge Function

```powershell
cd C:\Users\Admin\Desktop\AIMBOT
supabase functions deploy askaimbot --no-verify-jwt
```

`--no-verify-jwt` is required so Telegram can POST without a Supabase login token.

Your function URL will be:

```text
https://YOUR_PROJECT_REF.supabase.co/functions/v1/askaimbot
```

Test in browser (GET):

```text
https://YOUR_PROJECT_REF.supabase.co/functions/v1/askaimbot?secret=YOUR_FUNCTION_SECRET
```

You should see: `askaimbot edge function is up`

---

## Step 6 — Tell Telegram to use the webhook

**Stop `python aimbot.py` first (Ctrl+C)** — or you will get `409 Conflict` in the terminal.

If PowerShell fails with *"underlying connection was closed"*, run this once, then retry:

```powershell
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
```

Replace placeholders and run in PowerShell **once**:

```powershell
$token = "YOUR_TELEGRAM_BOT_TOKEN"
$ref = "YOUR_PROJECT_REF"
$secret = "YOUR_FUNCTION_SECRET"

$webhookUrl = "https://$ref.supabase.co/functions/v1/askaimbot?secret=$secret"

# Remove old polling mode
Invoke-RestMethod "https://api.telegram.org/bot$token/deleteWebhook?drop_pending_updates=true"

# Enable webhook (GET style — reliable on Windows)
$encoded = [uri]::EscapeDataString($webhookUrl)
Invoke-RestMethod "https://api.telegram.org/bot$token/setWebhook?url=$encoded"
```

Check webhook status:

```powershell
Invoke-RestMethod "https://api.telegram.org/bot$token/getWebhookInfo"
```

`url` should match your Supabase function URL and `last_error_message` should be empty.

---

## Step 7 — Test on Telegram

1. Open @askaimbot  
2. Send `/start`  
3. Send a normal question  

If it fails, check logs:

```powershell
supabase functions logs askaimbot
```

Or: Supabase Dashboard → **Edge Functions** → **askaimbot** → **Logs**

---

## Step 8 — Push code to GitHub (backup, not required for run)

```powershell
cd C:\Users\Admin\Desktop\AIMBOT
git init
git add aimbot.py supabase requirements.txt todo.txt DEPLOY-SUPABASE.md
git commit -m "Add Supabase edge deploy for askaimbot"
```

Create a **private** repo on GitHub and push. Do **not** commit `.env` or tokens.

---

## Switching back to local Python (optional)

```powershell
# Turn off Supabase webhook
Invoke-RestMethod "https://api.telegram.org/bot$token/deleteWebhook"

# Run locally again
$env:TELEGRAM_TOKEN = "..."
$env:GEMINI_API_KEY = "..."
python aimbot.py
```

---

## Free tier limits (know this)

- Edge Functions: generous free tier; each message = one invocation  
- Max run time per call: ~150s on free (plenty for one reply)  
- Cold start: first message after idle may be 1–3s slower  
- Gemini: still ~30 RPM / daily cap on your Google key  

---

## Later: use Supabase Database (Phase 2+)

When you add memory / Empire ID:

1. Dashboard → **Table Editor** → create `citizens`, `messages`  
2. Edge function reads/writes with `SUPABASE_SERVICE_ROLE_KEY`  
3. Your Nigerian `knowledge/` files can be stored in **Storage** or Postgres  

`aimbot.py` on your PC can stay for local testing; production = Edge Function.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `not allowed` | Wrong `?secret=` in webhook URL vs `FUNCTION_SECRET` |
| `401` on function | Redeploy with `--no-verify-jwt` |
| Bot silent | `getWebhookInfo` → read `last_error_message` |
| Double replies | Stop Python bot on laptop while webhook is on |
| `409 Conflict` / `can't use getUpdates while webhook is active` | **Good sign** — webhook is on. Press **Ctrl+C** to stop `python aimbot.py` |
| Gemini 429 | Wait 1 min; free tier limit |

---

## Quick reference

```text
Function URL:
  https://YOUR_PROJECT_REF.supabase.co/functions/v1/askaimbot?secret=YOUR_FUNCTION_SECRET

Redeploy after code changes:
  supabase functions deploy askaimbot --no-verify-jwt

Logs:
  supabase functions logs askaimbot
```
