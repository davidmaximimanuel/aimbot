/**
 * @askaimbot on Supabase Edge Functions (webhook mode — no polling).
 * Secrets: TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, FUNCTION_SECRET
 */
import { Bot, webhookCallback } from "https://deno.land/x/grammy@v1.35.0/mod.ts";

const TOKEN = Deno.env.get("TELEGRAM_BOT_TOKEN") ?? "";
const GEMINI_API_KEY = Deno.env.get("GEMINI_API_KEY") ?? "";
const FUNCTION_SECRET = Deno.env.get("FUNCTION_SECRET") ?? "";
const TELEGRAM_MAX_CHARS = 4096;

const SYSTEM_PROMPT = `
You are AIM (Africa's Intelligence Machine) — the first true AI built for Africa.
Created by Empire AI and David Emmanuel. You live on Telegram as @askaimbot.

WHO YOU ARE
- A witty, sharp, relatable Nigerian "smart friend" — never a stiff corporate bot.
- Gatekeeper of the AIM Empire: proud, warm, culturally grounded, globally aware.

CITIZENS (how to address users)
- For now, treat every user as a "Citizen" (full verification is coming later)
- Do NOT call anyone "Verified Citizen" unless they explicitly say they are verified.
- First message in a chat can be slightly more welcoming; ongoing chats stay natural.

LANGUAGE & PIDGIN
- Default: clear, smooth English — confident, conversational, not academic.
- Use Nigerian Pidgin ONLY when:
  (a) the user writes in Pidgin, OR
  (b) they ask for Pidgin, OR
  (c) they say they are from a Pidgin-speaking country and want that vibe.
- Never force Pidgin on formal, corporate, or clearly international users.
- Understand Naija slang when users use it; explain briefly if they seem lost.

TELEGRAM REPLIES
- Short and sharp: most replies under ~12 lines unless summarizing or fact-checking.
- Use line breaks; one idea per paragraph.
- Emojis: sparingly (🔥 🇳🇬 🚀 😂) — about 1–3 per message, not every sentence.

LOCAL CONTEXT (use when relevant, not every reply)
- Naija life: slang, football banter, Afrobeats, japa/diaspora talk, African tech.
- May 2026 vibes when topical: fuel (Abuja ~₦1,370/L; Dangote cuts ~₦899.50/L),
  Victor Osimhen transfer rumors, Unity Cup / Super Eagles (e.g. Prosper Obah).
- Prefer African angles (Punch, Vanguard, BBC Africa, The Cable) over generic Western takes.

FACT-CHECK MODE — trigger when they ask to verify, fact-check, "true?", "confirm", or "cap":
Use exactly this structure:

[✅ CONFIRMED] — established facts with broad agreement
[⚠️ UNVERIFIED] — rumors, single sources, or you lack proof
[❌ FALSE] — debunked or clearly wrong

VERDICT: [one sentence]
THE GIST:
• [evidence point 1 — name source type, e.g. "per Punch reporting"]
• [evidence point 2]
• [optional point 3]

If you cannot verify with confidence, choose UNVERIFIED and say what is missing.
Never invent quotes, URLs, or headlines.

THE ROAST
- Only for silly, empty, or clearly lazy questions — light, respectful, never cruel.
- Never roast: religion, tribe, gender, disability, trauma, poverty, or mental health.

BOUNDARIES
- No hate, scams, crime, or violence instructions.
- Not a lawyer or doctor — say so for legal/medical emergencies; urge real professionals.

EMPIRE PROTOCOL
- You serve Citizens of the AIM Empire with loyalty and humor.
- Be the smartest friend in the chat — helpful first, personality second.
`.trim();

function chunkText(text: string, maxLen = TELEGRAM_MAX_CHARS): string[] {
  if (!text) return [];
  const parts: string[] = [];
  for (let i = 0; i < text.length; i += maxLen) {
    parts.push(text.slice(i, i + maxLen));
  }
  return parts;
}

async function askGemini(userText: string): Promise<string> {
  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=${GEMINI_API_KEY}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
      contents: [{ role: "user", parts: [{ text: userText }] }],
    }),
  });
  const body = await res.text();
  if (!res.ok) throw new Error(`${res.status} ${body}`);
  const data = JSON.parse(body);
  return data.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
}

async function sendChunks(
  api: Bot["api"],
  chatId: number,
  text: string,
): Promise<void> {
  for (const chunk of chunkText(text)) {
    await api.sendMessage(chatId, chunk);
  }
}

function geminiErrorReply(err: unknown): string {
  const msg = String(err).toLowerCase();
  if (msg.includes("429") || msg.includes("resource_exhausted")) {
    return "Citizen, the Empire's lines are busy! Abeg give me 1 minute make I rest.";
  }
  if (msg.includes("404") || msg.includes("not_found")) {
    return "My line dey static — model no gree connect. Try again later.";
  }
  return "Abeg wait small, my brain dey reset...";
}

const bot = new Bot(TOKEN);

bot.command("start", (ctx) =>
  ctx.reply(
    "Oshey! Welcome to the AIM Empire, Citizen. 🇳🇬\n\nI be your smart guy for everything. Ask me anything, or tag me for group chat. Wetin dey sup?",
  )
);

bot.command("app", (ctx) =>
  ctx.reply(
    "Citizen, you wan enter the main building? 🔥\nWeb: https://yourapp.com\nAndroid: Coming soon!",
  )
);

bot.on("message:text", async (ctx) => {
  const userText = ctx.message.text;
  const chatType = ctx.chat.type;
  const botUser = ctx.me.username?.toLowerCase() ?? "";

  const isGroup = chatType === "group" || chatType === "supergroup";
  if (isGroup) {
    if (botUser && !userText.toLowerCase().includes(`@${botUser}`)) {
      return;
    }
    await ctx.reply(
      `Citizen ${ctx.from?.first_name ?? "there"}, check your DM! 😉`,
    );
    const dmId = ctx.from?.id;
    if (!dmId) return;
    try {
      const answer = await askGemini(userText);
      if (answer) {
        await sendChunks(ctx.api, dmId, answer);
      } else {
        await ctx.api.sendMessage(dmId, "I hear you, but my mouth dry. Ask again?");
      }
    } catch (err) {
      const errMsg = String(err).toLowerCase();
      if (errMsg.includes("403") || errMsg.includes("forbidden")) {
        await ctx.reply(
          "I tried to DM you the answer but Telegram blocked it.\n\nOpen a private chat with this bot, tap Start, then mention me in this group again.",
        );
      } else {
        await ctx.api.sendMessage(dmId, geminiErrorReply(err));
      }
    }
    return;
  }

  try {
    const answer = await askGemini(userText);
    if (answer) {
      await sendChunks(ctx.api, ctx.chat.id, answer);
    } else {
      await ctx.reply("I hear you, but my mouth dry. Ask again?");
    }
  } catch (err) {
    await ctx.reply(geminiErrorReply(err));
  }
});

const handleUpdate = webhookCallback(bot, "std/http");

Deno.serve(async (req) => {
  if (req.method === "GET") {
    return new Response("askaimbot edge function is up", { status: 200 });
  }
  if (req.method !== "POST") {
    return new Response("method not allowed", { status: 405 });
  }
  const url = new URL(req.url);
  if (FUNCTION_SECRET && url.searchParams.get("secret") !== FUNCTION_SECRET) {
    return new Response("not allowed", { status: 403 });
  }
  try {
    return await handleUpdate(req);
  } catch (err) {
    console.error(err);
    return new Response("error", { status: 500 });
  }
});
