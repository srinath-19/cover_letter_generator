import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# Optional: better main-content extraction
try:
    import trafilatura  # type: ignore
except Exception:
    trafilatura = None


@dataclass
class FetchResult:
    url: str
    html: str
    fetched_via: str  # "requests" or "playwright"


def slugify_url(url: str, max_len: int = 80) -> str:
    p = urlparse(url)
    base = (p.netloc + p.path).strip("/").replace("/", "_")
    base = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return (base or "job_post")[:max_len]


def fetch_html_requests(url: str, timeout: int = 25, retries: int = 2) -> FetchResult:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return FetchResult(url=url, html=resp.text, fetched_via="requests")
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch via requests: {last_err}")


def fetch_html_playwright(url: str, timeout_ms: int = 30_000) -> FetchResult:
    # Import only when needed
    from playwright.sync_api import sync_playwright  # type: ignore

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=timeout_ms, wait_until="networkidle")
        html = page.content()
        browser.close()

    return FetchResult(url=url, html=html, fetched_via="playwright")


def strip_noise(soup: BeautifulSoup) -> None:
    for sel in ["script", "style", "noscript", "svg", "canvas", "iframe", "form"]:
        for tag in soup.select(sel):
            tag.decompose()

    for sel in ["nav", "footer", "header", "aside"]:
        for tag in soup.select(sel):
            tag.decompose()

    # best-effort cookie/modal removal (conservative)
    for tag in soup.find_all(True):
        classes = " ".join(tag.get("class", [])) if tag.get("class") else ""
        tid = tag.get("id", "") or ""
        blob = f"{classes} {tid}".lower()
        if any(k in blob for k in ["cookie", "consent", "modal", "subscribe", "newsletter"]):
            if tag.name in {"div", "section"} and len(tag.get_text(" ", strip=True)) < 800:
                tag.decompose()


def pick_main_node(soup: BeautifulSoup):
    for sel in ["main", "article", "[role='main']"]:
        node = soup.select_one(sel)
        if node and len(node.get_text(" ", strip=True)) > 400:
            return node

    for sel in [
        ".job-description", "#job-description",
        ".description", "#content", ".content",
        ".posting", ".jobPosting",
        ".posting-requirements", ".posting-categories",
    ]:
        node = soup.select_one(sel)
        if node and len(node.get_text(" ", strip=True)) > 400:
            return node

    best = soup.body or soup
    best_len = len(best.get_text(" ", strip=True))
    for node in soup.select("div, section"):
        tlen = len(node.get_text(" ", strip=True))
        if tlen > best_len:
            best = node
            best_len = tlen
    return best


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def html_to_clean_text(html: str) -> str:
    # Try trafilatura first if available
    if trafilatura is not None:
        extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
        if extracted and len(extracted.strip()) > 300:
            return normalize_text(extracted)

    soup = BeautifulSoup(html, "lxml")
    strip_noise(soup)
    main = pick_main_node(soup)
    text = main.get_text("\n", strip=True)
    return normalize_text(text)


def looks_blocked_or_js_gate(html: str, clean_text: str) -> bool:
    blob = (html + "\n" + clean_text).lower()
    signals = [
        "access denied",
        "request blocked",
        "verify you are human",
        "captcha",
        "cloudflare",
        "unusual traffic",
        "enable javascript",
        "please enable javascript",
        "datadome",
        "akamai",
        "incident id",
        "bot detection",
        "are you a robot",
    ]
    return any(s in blob for s in signals)


def content_quality_ok(clean_text: str) -> bool:
    # Job posts are typically long. These thresholds are conservative.
    chars = len(clean_text)
    words = len(clean_text.split())
    if chars < 1200 or words < 180:
        return False

    # Bonus: if it contains typical job-post sections, it's probably correct.
    low = clean_text.lower()
    section_hits = sum(
        1 for k in ["responsibilities", "requirements", "qualifications", "what you'll do", "about the role", "preferred"]
        if k in low
    )
    return section_hits >= 1


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def ingest_url(
    url: str,
    out_dir: Path,
    save_raw_html: bool,
    force_playwright: bool,
    allow_fallback: bool,
) -> Tuple[Path, Optional[Path], str]:
    raw_dir = out_dir / "raw"
    clean_dir = out_dir / "clean"
    ensure_dirs(raw_dir, clean_dir)

    slug = slugify_url(url)

    def save_outputs(fr: FetchResult) -> Tuple[Path, Optional[Path], str, str]:
        raw_path = None
        if save_raw_html:
            raw_path = raw_dir / f"{slug}.{fr.fetched_via}.html"
            write_file(raw_path, fr.html)

        clean_text = html_to_clean_text(fr.html)
        clean_path = clean_dir / f"{slug}.txt"
        write_file(clean_path, clean_text)
        return clean_path, raw_path, fr.fetched_via, clean_text

    # 1) If user forces playwright, do it directly
    if force_playwright:
        fr = fetch_html_playwright(url)
        clean_path, raw_path, via, _ = save_outputs(fr)
        return clean_path, raw_path, via

    # 2) Otherwise: requests first
    fr = fetch_html_requests(url)
    clean_path, raw_path, via, clean_text = save_outputs(fr)

    # 3) Heuristic: fallback to playwright if content looks bad/blocked
    if allow_fallback:
        if looks_blocked_or_js_gate(fr.html, clean_text) or not content_quality_ok(clean_text):
            print("[WARN] Content looks blocked/JS-rendered or too short. Trying Playwright fallback...")
            try:
                fr2 = fetch_html_playwright(url)
                clean_path2, raw_path2, via2, clean_text2 = save_outputs(fr2)

                # Keep the better result
                if content_quality_ok(clean_text2) and len(clean_text2) >= len(clean_text):
                    return clean_path2, raw_path2, via2

                print("[WARN] Playwright did not improve quality; keeping requests result.")
            except Exception as e:
                print(f"[WARN] Playwright fallback failed ({e}).")
                print("       If you want fallback support: `uv add playwright` then `uv run playwright install`")

    return clean_path, raw_path, via


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Ingest job post URL and output cleaned text.")
    parser.add_argument("url", help="Job posting URL")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument("--save-html", action="store_true", help="Save raw HTML (requests/playwright) to outputs/raw/")
    parser.add_argument("--playwright", action="store_true", help="Force Playwright (skip requests)")
    parser.add_argument("--no-fallback", action="store_true", help="Disable auto Playwright fallback")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    clean_path, raw_path, fetched_via = ingest_url(
        url=args.url,
        out_dir=out_dir,
        save_raw_html=args.save_html,
        force_playwright=args.playwright,
        allow_fallback=(not args.no_fallback),
    )

    print(f"[OK] fetched_via={fetched_via}")
    if raw_path:
        print(f"[OK] raw_html:  {raw_path}")
    print(f"[OK] clean_txt: {clean_path}")
    print("\n--- Preview (first 30 lines) ---")
    preview = clean_path.read_text(encoding="utf-8").splitlines()[:30]
    print("\n".join(preview))


if __name__ == "__main__":
    main()
