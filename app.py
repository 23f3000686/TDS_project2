# app_with_faster_whisper.py
"""
Complete app.py with faster-whisper local transcription integrated.
- Requires: fastapi, uvicorn, httpx, playwright, beautifulsoup4, pdfplumber (optional), faster-whisper
- Also requires ffmpeg on PATH.

Environment variables (set in same shell before starting):
  SECRET_VALUE   - secret from Google Form
  AIPIPE_URL     - e.g. https://aipipe.org/openrouter/v1/chat/completions
  AIPIPE_TOKEN   - AIPipe bearer token (optional)
  OPENAI_API_KEY - optional (legacy OpenAI fallback)
  PROCESS_TIMEOUT - optional (default 170)
  FW_MODEL       - faster-whisper model name (tiny, small, medium). Default: small

Run:
  python -m uvicorn app_with_faster_whisper:app --host 0.0.0.0 --port 8000

This file prefers AIPipe, falls back to OpenAI if configured, and uses faster-whisper for audio transcription.
"""

import os
import re
import json
import base64
import logging
import asyncio
import tempfile
from io import BytesIO
from urllib.parse import urljoin, urlparse
from html import unescape
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi import BackgroundTasks

import httpx
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup

# Optional PDF parsing
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

# faster-whisper
try:
    from faster_whisper import WhisperModel
    HAS_FW = True
except Exception:
    HAS_FW = False

# legacy OpenAI client (optional)
try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# CONFIG
AIPIPE_URL = os.getenv("AIPIPE_URL", "").strip() or None
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "").strip() or None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or None
SECRET_VALUE = os.getenv("SECRET_VALUE", "").strip() or None
print(SECRET_VALUE)
PROCESS_TIMEOUT = int(os.getenv("PROCESS_TIMEOUT", "170"))  # must be < 180
MAX_SUBMIT_BYTES = 1024 * 1024  # 1 MB
FW_MODEL = os.getenv("FW_MODEL", "small")
FW_BEAM = int(os.getenv("FW_BEAM", "5"))

# normalize AIPipe common mistakes
if AIPIPE_URL:
    if "aipipe.org/playground" in AIPIPE_URL or AIPIPE_URL.rstrip("/").endswith("playground"):
        AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
    if AIPIPE_URL.rstrip("/") in ("https://aipipe.org", "https://aipipe.org/"):
        AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"

# OpenAI legacy setup (if provided)
if OPENAI_API_KEY and HAS_OPENAI:
    openai.api_key = OPENAI_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("llm-quiz-fw")


if not SECRET_VALUE:
    logger.warning("SECRET_VALUE not set - server will reject requests (403).")

app = FastAPI(title="LLM Analysis Quiz Endpoint (faster-whisper)")

# ThreadPool for blocking transcription calls
_TP = ThreadPoolExecutor(max_workers=1)

# ---------------------------
# Helpers
# ---------------------------
def validate_payload_json(data):
    if not isinstance(data, dict):
        raise ValueError("JSON payload must be an object")
    for k in ("email", "secret", "url"):
        if k not in data:
            raise ValueError(f"Missing required field: {k}")


def sanitize_answer_for_submit(ans):
    if ans is None:
        return None
    if isinstance(ans, (str, int, float, bool)):
        return ans
    if isinstance(ans, dict) and ans.get("type") == "file":
        content = ans.get("content_base64") or ans.get("content") or ans.get("data")
        if isinstance(content, str):
            b = content.encode("utf-8")
            if len(b) > 900 * 1024:
                return content[:900 * 1024]
            return content
        b = ans.get("bytes")
        if isinstance(b, (bytes, bytearray)):
            return "data:application/octet-stream;base64," + base64.b64encode(b).decode("ascii")
    try:
        j = json.dumps(ans, ensure_ascii=False)
        if len(j.encode("utf-8")) > 900 * 1024:
            j = j[:900 * 1024]
        return j
    except Exception:
        txt = str(ans)
        if len(txt.encode("utf-8")) > 900 * 1024:
            txt = txt[:900 * 1024]
        return txt

# ---------------------------
# LLM callers (async)
# ---------------------------
async def call_aipipe(payload: dict) -> dict:
    """
    Call AIPipe / OpenRouter-style endpoint. AIPIPE_TOKEN is optional; some setups don't require it.
    Returns dict with top-level key 'answer' (or raw assistant text if parsing fails).
    """
    if not AIPIPE_URL:
        raise RuntimeError("AIPipe not configured")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    # include auth only if token provided
    if AIPIPE_TOKEN:
        headers["Authorization"] = f"Bearer {AIPIPE_TOKEN}"

    body = {
        "model": os.getenv("AIPIPE_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": "Return valid JSON with top-level key 'answer'."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ],
        "max_tokens": 800,
        "temperature": 0.0
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(AIPIPE_URL, headers=headers, json=body)
        r.raise_for_status()
        j = r.json()
    assistant_text = None
    if isinstance(j, dict):
        assistant_text = j.get("choices", [{}])[0].get("message", {}).get("content") or j.get("text") or j.get("result")
    if not assistant_text:
        return {"answer": None}
    m = re.search(r'(\{[\s\S]*\})', assistant_text)
    candidate = m.group(1) if m else assistant_text
    try:
        parsed = json.loads(candidate)
        if "answer" in parsed:
            return parsed
        return {"answer": parsed}
    except Exception:
        return {"answer": assistant_text.strip()}


async def call_openai_legacy(payload: dict) -> dict:
    # Only attempt OpenAI if a key is provided and client imported successfully
    if not OPENAI_API_KEY or not HAS_OPENAI:
        raise RuntimeError("OpenAI legacy not configured")

    prompt_system = "You are an assistant that MUST return valid JSON object with top-level key 'answer'."
    user_msg = "Input JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\nReturn only JSON with key 'answer'."

    resp = openai.ChatCompletion.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[{"role": "system", "content": prompt_system}, {"role": "user", "content": user_msg}],
        temperature=0.0, max_tokens=800
    )
    content = None

    try:
        content = resp["choices"][0]["message"]["content"]
    except Exception:
        content = resp.get("choices", [{}])[0].get("text")
    if not content:
        return {"answer": None}
    m = re.search(r'(\{[\s\S]*\})', content)
    candidate = m.group(1) if m else content
    try:
        parsed = json.loads(candidate)
        if "answer" in parsed:
            return parsed
        return {"answer": parsed}
    except Exception:
        return {"answer": content.strip()}


async def call_llm_with_preference(payload: dict) -> dict:
    """
    Prefer AIPipe (if AIPIPE_URL set). Fallback to OpenAI legacy only if configured. Returns dict with 'answer'.
    """
    if AIPIPE_URL:
        try:
            return await call_aipipe(payload)
        except Exception as e:
            logger.warning("AIPipe call failed: %s", e)
    if OPENAI_API_KEY and HAS_OPENAI:
        try:
            return await call_openai_legacy(payload)
        except Exception as e:
            logger.warning("OpenAI legacy call failed: %s", e)
    logger.error("No LLM available; returning placeholder.")
    return {"answer": "PLACEHOLDER_ANSWER"}

# ---------------------------
# Page parsing helpers
# ---------------------------
async def find_submit_url(page):
    html = await page.content()
    try:
        visible = await page.inner_text("body")
    except Exception:
        visible = ""
    m = re.search(r'POST\s+.*\s+(https?://\S+submit\S*)', visible, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip('.,;')
    try:
        form = await page.query_selector("form[action]")
        if form:
            action = await form.get_attribute("action")
            if action:
                return urljoin(page.url, action)
    except Exception:
        pass
    try:
        anchors = await page.query_selector_all("a")
        for a in anchors:
            href = await a.get_attribute("href")
            txt = (await a.inner_text() or "").lower()
            if href and ("submit" in href.lower() or "submit" in txt):
                return urljoin(page.url, href)
    except Exception:
        pass
    m = re.search(r'(https?://[^"\'>\s]*submit[^\s"\'<>]*)', html, re.IGNORECASE)
    if m:
        return m.group(1)
    for m in re.finditer(r'<script[^>]*type=["\']application/json["\'][^>]*>([\s\S]*?)</script>', html, re.IGNORECASE):
        try:
            j = json.loads(m.group(1))
            def walk(obj):
                if isinstance(obj, str) and obj.startswith("http") and "submit" in obj:
                    return obj
                if isinstance(obj, dict):
                    for v in obj.values():
                        r = walk(v)
                        if r:
                            return r
                if isinstance(obj, list):
                    for el in obj:
                        r = walk(el)
                        if r:
                            return r
                return None
            cand = walk(j)
            if cand:
                return cand
        except Exception:
            pass
    for match in re.finditer(r'atob\(\s*(["\'`])([A-Za-z0-9+/=\s]+)\1\s*\)', html):
        b64 = match.group(2).replace("\n", "").replace(" ", "")
        try:
            decoded = base64.b64decode(b64 + "=" * (-len(b64) % 4)).decode("utf-8", errors="ignore")
            m2 = re.search(r'(https?://[^\s"\'<>]*submit[^\s"\'<>]*)', decoded, re.IGNORECASE)
            if m2:
                return m2.group(1)
        except Exception:
            pass
    m = re.search(r'["\'](/[^"\']*submit[^"\']*)["\']', html, re.IGNORECASE)
    if m:
        return urljoin(page.url, m.group(1))
    return None

async def handle_pdf_text(pdf_url):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(pdf_url)
            if r.status_code == 200 and HAS_PDFPLUMBER:
                with pdfplumber.open(BytesIO(r.content)) as pdf:
                    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                    return text
    except Exception:
        logger.exception("PDF parse failed")
    return None

async def handle_table_sum(page, page_text):
    try:
        tables = await page.query_selector_all("table")
        if tables:
            for table in tables:
                html_table = await table.inner_html()
                soup = BeautifulSoup(html_table, "html.parser")
                headers = [th.get_text(strip=True).lower() for th in soup.find_all("th")]
                if headers and "value" in headers:
                    idx = headers.index("value")
                    total = 0.0
                    found = False
                    for row in soup.find_all("tr"):
                        cells = row.find_all(["td","th"])
                        if len(cells) > idx:
                            txt = cells[idx].get_text().strip().replace(",", "")
                            m = re.search(r'-?\d+(?:\.\d+)?', txt)
                            if m:
                                total += float(m.group(0))
                                found = True
                    if found:
                        return total
        nums = [float(x.replace(",","")) for x in re.findall(r'-?\d+(?:\.\d+)?', page_text)]
        if nums:
            return sum(nums)
    except Exception:
        logger.exception("Table sum handler failed")
    return None

# faster-whisper helper (blocking) to run in thread
def _transcribe_sync_with_faster_whisper(model_name: str, audio_path: str, language: str | None = None, beam_size:int=5):
    model = WhisperModel(model_name, device="cpu")
    segments, info = model.transcribe(audio_path, language=language, beam_size=beam_size)
    parts = []
    for seg in segments:
        parts.append(seg.text.strip())
    return " ".join(p for p in parts if p)

async def handle_audio_and_transcribe(page, page_text, page_url: str):
    audio_url = None
    try:
        audio_el = await page.query_selector("audio")
        if audio_el:
            audio_url = await audio_el.get_attribute("src")
            if not audio_url:
                source = await page.query_selector("audio source")
                if source:
                    audio_url = await source.get_attribute("src")
        if not audio_url:
            m = re.search(r'(https?://\S+\.(?:mp3|wav|m4a|ogg|opus))', page_text, re.IGNORECASE)
            if m:
                audio_url = m.group(1)
    except Exception:
        pass

    if not audio_url:
        return None

    audio_url = urljoin(page_url, audio_url)
    logger.info("Found audio: %s", audio_url)

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(audio_url)
            if r.status_code != 200:
                logger.warning("Audio download failed: %s", r.status_code)
                return None
            audio_bytes = r.content
    except Exception as e:
        logger.exception("Audio download exception: %s", e)
        return None

    # save to temp
    suffix = Path(audio_url).suffix or ".mp3"
    try:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tf.name
        tf.write(audio_bytes)
        tf.flush()
        tf.close()
    except Exception as e:
        logger.exception("Failed to write temp audio file: %s", e)
        return None

    if not HAS_FW:
        logger.warning("faster-whisper not installed; falling back to LLM transcription for small audio")
        if len(audio_bytes) < 500 * 1024:
            encoded = "data:audio/mpeg;base64," + base64.b64encode(audio_bytes).decode("ascii")
            payload = {"email": "", "url": page_url, "question_text": page_text, "extra": {"audio_data": encoded}, "instructions": "Transcribe the audio and return {\"answer\": \"...\"}"}
            res = await call_llm_with_preference(payload)
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
            return res.get("answer")
        else:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
            return None

    # Transcribe locally via faster-whisper in thread
    loop = asyncio.get_event_loop()
    try:
        transcript = await loop.run_in_executor(_TP, _transcribe_sync_with_faster_whisper, FW_MODEL, tmp_path, None, FW_BEAM)
    except Exception as e:
        logger.exception("Local transcription failed: %s", e)
        transcript = None
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    if not transcript:
        return None

    # If question likely asks for a sum, compute numeric sum from transcript
    if re.search(r'\bsum\b', page_text.lower()) or re.search(r'\bsum\b', page_url.lower()):
        nums = re.findall(r'-?\d+(?:\.\d+)?', transcript.replace(",", ""))
        if nums:
            try:
                total = sum(float(n) for n in nums)
                if all('.' not in n for n in nums):
                    return int(total)
                return total
            except Exception:
                return transcript

    return transcript

# ---------------------------
# Main processing
# ---------------------------
async def process_request_safe(data):
    try:
        await asyncio.wait_for(process_request(data), timeout=PROCESS_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error("Processing timed out for email=%s", data.get("email"))
    except Exception:
        logger.exception("Unhandled error while processing request for %s", data.get("email"))

async def process_request(data):
    email = data.get("email")
    secret = data.get("secret")
    url = data.get("url")

    logger.info("Incoming request: email=%s url=%s", email, url)

    start_time = asyncio.get_event_loop().time()
    elapsed = lambda: asyncio.get_event_loop().time() - start_time

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context()
        page = await context.new_page()

        while url and elapsed() < (PROCESS_TIMEOUT - 2):
            logger.info("Visiting %s (email=%s)", url, email)
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
            except PlaywrightTimeoutError:
                logger.warning("Page load timeout for %s", url)

            page_text = await page.content()

            submit_url = await find_submit_url(page)
            if not submit_url:
                logger.error("No submit URL found on page %s", url)
                try:
                    with open("last_page_dump.html", "w", encoding="utf-8") as f:
                        f.write(await page.content())
                    logger.error("Dumped HTML to last_page_dump.html")
                except Exception:
                    logger.exception("Failed to write HTML dump")
                await browser.close()
                return

            # sanitize submit_url
            try:
                submit_url = re.sub(r"<[^>]+>", "", submit_url)
                submit_url = unescape(submit_url)
                submit_url = submit_url.strip()
                submit_url = "".join(ch for ch in submit_url if ord(ch) >= 32)
                submit_url = urljoin(page.url, submit_url)
            except Exception:
                pass

            parsed = urlparse(submit_url)
            if not parsed.scheme or not parsed.netloc:
                logger.error("Submit URL not valid after sanitization: %r (from page %s)", submit_url, url)
                await browser.close()
                return

            logger.info("Found submit URL: %s", submit_url)

            # collect visible text for LLM / parsing
            try:
                texts = []
                for sel in ["#result", ".quiz", "body"]:
                    el = await page.query_selector(sel)
                    if el:
                        t = await el.inner_text()
                        if t and len(t.strip()) > 10:
                            texts.append(t.strip())
                if not texts:
                    texts = [await page.inner_text("body")]
                question_text = "\n\n".join(texts)[:4000]
            except Exception:
                question_text = (await page.content())[:4000]

            # follow demo-scrape-data link if present
            try:
                anchors = await page.query_selector_all("a[href]")
                for a in anchors:
                    href = await a.get_attribute("href") or ""
                    if "demo-scrape-data" in href or "scrape-data" in href:
                        follow_url = urljoin(page.url, href)
                        try:
                            await page.goto(follow_url, wait_until="networkidle", timeout=20000)
                            page_text = await page.content()
                            try:
                                elbody = await page.inner_text("body")
                                question_text = elbody[:4000]
                            except Exception:
                                question_text = page_text[:4000]
                        except Exception:
                            pass
                        break
            except Exception:
                pass

            # try to extract a page secret like "Secret code is 20706"
            page_secret = None
            m = re.search(r"Secret code\s*(?:is)?\s*[:\-]?\s*(\d+)", question_text, re.IGNORECASE)
            if not m:
                m = re.search(r"Secret code\s*(?:is)?\s*[:\-]?\s*(\d+)", page_text, re.IGNORECASE)
            if m:
                page_secret = m.group(1)
                submission = {
                    "email": email,
                    "secret": secret,
                    "url": url,
                    "answer": int(page_secret)
                }
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(submit_url, json=submission)
                    try:
                        resp_json = r.json()
                    except Exception:
                        resp_json = {"raw": r.text}
                logger.info("Auto-submitted page secret -> %s", resp_json)
                next_url = resp_json.get("url")
                if next_url:
                    url = next_url
                    continue
                else:
                    await browser.close()
                    return

            # look for attached PDF link(s)
            pdf_url = None
            try:
                anchors = await page.query_selector_all("a")
                for a in anchors:
                    href = await a.get_attribute("href")
                    if href and href.lower().endswith(".pdf"):
                        pdf_url = urljoin(page.url, href)
                        break
            except Exception:
                pass

            extra_data = {}
            if pdf_url:
                pdf_text = await handle_pdf_text(pdf_url)
                if pdf_text:
                    extra_data["pdf_text"] = pdf_text

            # auto-handle HTML table sums
            if re.search(r'\bsum\b', question_text.lower()) or re.search(r'\bsum\b', page_text.lower()) or 'sum' in url.lower():
                total = await handle_table_sum(page, page_text)
                if total is not None:
                    submission = {"email": email, "secret": secret, "url": url, "answer": sanitize_answer_for_submit(total)}
                    try:
                        b = json.dumps(submission).encode("utf-8")
                        if len(b) > MAX_SUBMIT_BYTES:
                            logger.error("Submission too large for %s", url)
                            await browser.close()
                            return
                    except Exception:
                        pass
                    async with httpx.AsyncClient(timeout=30) as client:
                        r = await client.post(submit_url, json=submission)
                        try:
                            resp_json = r.json()
                        except Exception:
                            resp_json = {"raw": r.text}
                    logger.info("Auto-submitted numeric answer -> %s", resp_json)
                    next_url = resp_json.get("url")
                    if next_url:
                        url = next_url
                        continue
                    else:
                        await browser.close()
                        return

            # audio handling (transcription & acting on instructions)
            audio_text = await handle_audio_and_transcribe(page, page_text, page.url)
            if audio_text:
                audio_lower = audio_text.lower()

                # If audio instructs downloading CSV or computing sum, attempt to do so
                if ("download" in audio_lower and "csv" in audio_lower) or ("sum" in audio_lower and "csv" in audio_lower) or ("sum" in audio_lower and "download" in audio_lower):
                    csv_url = None
                    try:
                        anchors = await page.query_selector_all("a")
                        for a in anchors:
                            href = await a.get_attribute("href")
                            if not href:
                                continue
                            if href.lower().endswith(".csv") or ("csv" in href.lower() and "download" in href.lower()):
                                csv_url = urljoin(page.url, href)
                                break
                        if not csv_url:
                            body_html = await page.content()
                            m_csv = re.search(r'(https?://[^\s"\']+\.csv)', body_html, re.IGNORECASE)
                            if m_csv:
                                csv_url = m_csv.group(1)
                    except Exception:
                        logger.exception("Error while scanning for CSV link")

                    if csv_url:
                        try:
                            async with httpx.AsyncClient(timeout=60) as client:
                                r = await client.get(csv_url)
                                r.raise_for_status()
                                csv_bytes = r.content

                            # detect cutoff from audio/question/page
                            def find_cutoff_from_text(txt):
                                if not txt:
                                    return None
                                m_cut = re.search(r'cutoff\s*[:=]\s*([+-]?\d+(?:\.\d+)?)', txt, re.IGNORECASE)
                                if m_cut:
                                    return float(m_cut.group(1))
                                m = re.search(r'cutoff(?: value)?\s*(?:is|=|:)?\s*([+-]?\d+(?:\.\d+)?)', txt, re.IGNORECASE)
                                if m:
                                    return float(m.group(1))
                                m2 = re.search(r'greater than or equal to\s*([+-]?\d+(?:\.\d+)?)', txt, re.IGNORECASE)
                                if m2:
                                    return float(m2.group(1))
                                nums = re.findall(r'([+-]?\d+(?:\.\d+)?)', txt)
                                if nums and re.search(r'cutoff', txt, re.IGNORECASE):
                                    return float(nums[-1])
                                return None

                            cutoff = find_cutoff_from_text(audio_text) or find_cutoff_from_text(question_text) or find_cutoff_from_text(page_text)

                            import io, csv
                            s = io.StringIO(csv_bytes.decode("utf-8", errors="ignore"))
                            reader = csv.reader(s)
                            headers = next(reader, None)
                            has_header = False
                            col_index = 0
                            if headers:
                                if any(re.search(r'[A-Za-z]', (h or "")) for h in headers):
                                    has_header = True
                                    low = [(h or "").strip().lower() for h in headers]
                                    if "value" in low:
                                        col_index = low.index("value")
                                    else:
                                        col_index = 0
                                else:
                                    s.seek(0)
                                    reader = csv.reader(s)
                                    has_header = False

                            nums = []
                            total_rows = 0
                            for row in reader:
                                if not row:
                                    continue
                                total_rows += 1
                                candidate_value = None
                                if has_header and len(row) > col_index:
                                    cell = row[col_index]
                                    mm = re.search(r'-?\d+(?:\.\d+)?', str(cell).replace(",", ""))
                                    if mm:
                                        candidate_value = float(mm.group(0))
                                else:
                                    for cell in row:
                                        mc = re.search(r'-?\d+(?:\.\d+)?', str(cell).replace(",", ""))
                                        if mc:
                                            candidate_value = float(mc.group(0))
                                            break
                                if candidate_value is None:
                                    continue
                                if cutoff is None or candidate_value >= float(cutoff):
                                    nums.append(candidate_value)

                            if nums:
                                total = sum(nums)
                                if all(float(x).is_integer() for x in nums):
                                    total_out = int(total)
                                else:
                                    total_out = round(total, 6)

                                submission = {"email": email, "secret": secret, "url": url, "answer": sanitize_answer_for_submit(total_out)}
                                async with httpx.AsyncClient(timeout=30) as client:
                                    r = await client.post(submit_url, json=submission)
                                    try:
                                        resp_json = r.json()
                                    except Exception:
                                        resp_json = {"raw": r.text}
                                logger.info("Submitted CSV-sum -> %s", resp_json)
                                next_url = resp_json.get("url")
                                if next_url:
                                    url = next_url
                                    continue
                                else:
                                    await browser.close()
                                    return
                            else:
                                logger.warning("No numeric data matched cutoff (or no numeric cells found) in CSV at %s", csv_url)
                        except Exception:
                            logger.exception("Failed to download/parse CSV")
                    else:
                        logger.warning("Audio requested CSV download but no csv link was found on page.")

                # fallback: submit transcription text
                submission = {"email": email, "secret": secret, "url": url, "answer": sanitize_answer_for_submit(audio_text)}
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(submit_url, json=submission)
                    try:
                        resp_json = r.json()
                    except Exception:
                        resp_json = {"raw": r.text}
                logger.info("Submitted transcription -> %s", resp_json)
                next_url = resp_json.get("url")
                if next_url:
                    url = next_url
                    continue
                else:
                    await browser.close()
                    return

            # -------------------
            # LLM fallback path
            # -------------------
            llm_payload = {"email": email, "url": url, "question_text": question_text, "extra": extra_data, "instructions": "Return JSON with top-level key 'answer'."}
            llm_answer = await call_llm_with_preference(llm_payload)

            if not isinstance(llm_answer, dict):
                logger.error("LLM returned unexpected non-dict response.")
                raw_answer = "LLM error"
            else:
                raw_answer = llm_answer.get("answer")

            if raw_answer is None:
                logger.warning("LLM returned no answer (placeholder).")
                raw_answer = "LLM unavailable"

            if isinstance(raw_answer, str) and raw_answer.strip().startswith("{"):
                try:
                    parsed = json.loads(raw_answer)
                    if isinstance(parsed, dict) and "answer" in parsed:
                        raw_answer = parsed["answer"]
                except Exception:
                    pass

            logger.info("LLM responded.")
            sanitized = sanitize_answer_for_submit(raw_answer)

            submission = {"email": email, "secret": secret, "url": url, "answer": sanitized}
            logger.info("Submitting payload summary: email=%s url=%s answer_type=%s answer_len=%d",
                        email, url, type(sanitized).__name__, (len(str(sanitized)) if sanitized is not None else 0))

            try:
                sb = json.dumps(submission).encode("utf-8")
                if len(sb) > MAX_SUBMIT_BYTES:
                    logger.error("Submission too large; aborting for %s", url)
                    await browser.close()
                    return
            except Exception:
                pass

            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(submit_url, json=submission)
                    try:
                        resp_json = r.json()
                    except Exception:
                        resp_json = {"raw": r.text}
                logger.info("Submit response: %s", str(resp_json)[:1000])
                next_url = resp_json.get("url")
                if next_url:
                    url = next_url
                    continue
                else:
                    logger.info("No further URL returned. Finished for %s", email)
                    await browser.close()
                    return
            except Exception:
                logger.exception("Failed to POST answer to submit url")
                await browser.close()
                return

        logger.warning("Loop ended (no url or time exceeded).")
        await browser.close()

def find_cutoff_from_text(txt):
    if not txt:
        return None

    # 1. Exact demo format: "Cutoff: 20706"
    m = re.search(r'cutoff\s*[:=]\s*([+-]?\d+(?:\.\d+)?)', txt, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # 2. Variations: "cutoff value is 20706"
    m = re.search(r'cutoff(?: value)?\s*(?:is|=|:)?\s*([+-]?\d+(?:\.\d+)?)', txt, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # 3. "greater than or equal to 20706"
    m = re.search(r'greater than or equal to\s*([+-]?\d+(?:\.\d+)?)', txt, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # 4. If appears in audio: last number
    nums = re.findall(r'([+-]?\d+(?:\.\d+)?)', txt)
    if nums and re.search(r'cutoff', txt, re.IGNORECASE):
        return float(nums[-1])

    return None

# ---------------------------
# FastAPI endpoint
# ---------------------------
@app.post("/receive_request")
async def receive_request(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"message": "Invalid JSON"})

    try:
        validate_payload_json(data)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"message": str(e)})

    if data.get("secret") != SECRET_VALUE:
        return JSONResponse(status_code=403, content={"message": "Forbidden"})

    background_tasks.add_task(process_request_safe, data)
    return JSONResponse(status_code=200, content={"message": "Request accepted"})

# ---------------------------
# run locally
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_with_faster_whisper:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
