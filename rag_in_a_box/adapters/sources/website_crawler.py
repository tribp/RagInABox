from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import httpx
from bs4 import BeautifulSoup

from rag_in_a_box.core.models import SourceItem


def _normalize_url(base: str, href: str) -> Optional[str]:
    """
    - Resolve relative URLs
    - Drop fragments (#section)
    - Skip mailto:, javascript:, tel:
    """
    href = (href or "").strip()
    if not href:
        return None

    lowered = href.lower()
    if lowered.startswith(("mailto:", "javascript:", "tel:")):
        return None

    absolute = urljoin(base, href)
    absolute, _frag = urldefrag(absolute)
    return absolute


def _domain(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    # Remove www. prefix to normalize domains
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _looks_like_pdf(url: str) -> bool:
    return url.lower().split("?")[0].endswith(".pdf")


@dataclass
class WebsiteCrawlerSource:
    start_urls: list[str]
    allowed_domains: list[str]
    exclude_prefixes: list[str] = field(default_factory=list)
    max_pages: int = 200
    max_depth: int = 3
    timeout_seconds: int = 20
    user_agent: str = "RagInABoxBot/0.1"
    include_pdfs: bool = True

    def _is_excluded(self, url: str) -> bool:
        u = url.rstrip("/")
        for p in self.exclude_prefixes:
            if u.startswith(p.rstrip("/")):
                return True
        return False

    def iter_items(self) -> Iterable[SourceItem]:
        if not self.start_urls:
            raise ValueError("WebsiteCrawlerSource.start_urls is empty")

        # If allowed_domains not provided, default to domains of start URLs
        allowed = [d.lower() for d in self.allowed_domains if d.strip()]
        if not allowed:
            allowed = sorted({_domain(u) for u in self.start_urls})

        client = httpx.Client(
            timeout=httpx.Timeout(self.timeout_seconds),
            follow_redirects=True,
            headers={"User-Agent": self.user_agent},
        )

        visited: set[str] = set()
        # queue entries: (url, depth, referrer_url)
        q = deque([(u, 0, None) for u in self.start_urls])

        pages_fetched = 0

        try:
            while q and pages_fetched < self.max_pages:
                url, depth, referrer_url = q.popleft()

                if self._is_excluded(url):
                    continue

                if url in visited:
                    continue
                visited.add(url)

                # Domain gate
                d = _domain(url)
                if d not in allowed:
                    continue

                # Optional PDF short-circuit
                if self.include_pdfs and _looks_like_pdf(url):
                    item = self._fetch_as_source_item(client, url, referrer_url=referrer_url)
                    if item is not None:
                        pages_fetched += 1
                        yield item
                    continue

                # Fetch
                item = self._fetch_as_source_item(client, url, referrer_url=referrer_url)
                if item is None:
                    continue

                pages_fetched += 1
                yield item

                # Only parse links from HTML
                if depth >= self.max_depth:
                    continue
                if item.mime_type not in ("text/html", "application/xhtml+xml"):
                    continue

                print(f"Crawled: {url} (depth={depth}, pages_fetched={pages_fetched})")

                html = item.data.decode("utf-8", errors="ignore") if isinstance(item.data, bytes) else item.data
                for link in self._extract_links(html, base_url=url):
                    if not link:
                        continue
                    if self._is_excluded(link):
                        continue
                    if link in visited:
                        continue
                    # Domain gate early
                    if _domain(link) not in allowed:
                        continue
                    q.append((link, depth + 1, url))

        finally:
            client.close()

    def _fetch_as_source_item(self, client: httpx.Client, url: str, referrer_url: Optional[str]) -> Optional[SourceItem]:
        try:
            resp = client.get(url)
        except Exception:
            return None

        if resp.status_code >= 400:
            return None

        content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()

        # Decide mime type
        if content_type in ("text/html", "application/xhtml+xml"):
            mime_type = content_type
            source_type = "web"
        elif content_type == "application/pdf":
            mime_type = "application/pdf"
            source_type = "web_pdf"
        else:
            # Heuristic: treat unknown text-like as text/plain; otherwise skip
            if content_type.startswith("text/"):
                mime_type = content_type
                source_type = "web"
            elif self.include_pdfs and _looks_like_pdf(url):
                mime_type = "application/pdf"
                source_type = "web_pdf"
            else:
                # Skip images, zips, etc. for MVP
                return None

        try:
            data = resp.content
        except Exception:
            return None

        return SourceItem(
            uri=url,
            data=data,
            mime_type=mime_type,
            metadata={
                "source": "website",
                "source_type": source_type,      # <--- NEW
                "mime_type": mime_type,          # <--- NEW (also in Document, later)
                "url": url,
                "referrer_url": referrer_url,    # <--- NEW
                "domain": _domain(url),          # <--- helpful
                "fetched_content_type": content_type,
                "status_code": resp.status_code,
            },
        )

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []

        for a in soup.find_all("a"):
            href = a.get("href")
            u = _normalize_url(base_url, href)
            if not u:
                continue
            links.append(u)

        return links

