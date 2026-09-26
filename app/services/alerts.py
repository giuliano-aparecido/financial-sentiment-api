"""Builds the big-loss alert email for a today/big-loss scan
(research_job.py). The API only renders it; scripts/send_big_loss_alert.py
(run by .github/workflows/big-loss-alert.yml) polls for it and sends it -
see that script for why.
"""

import datetime
import html
from urllib.parse import quote

from app.services.swiss_today_screener import LOSS_THRESHOLD_PCT

YAHOO_QUOTE_URL = "https://finance.yahoo.com/quote/"


def yahoo_quote_url(ticker: str) -> str:
    return YAHOO_QUOTE_URL + quote(ticker, safe="")


def format_alert(rows: list[dict], status: dict, now: datetime.datetime) -> tuple[str, str, str]:
    """(subject, plain-text body, HTML body) for a scan with at least one match."""
    stamp = now.strftime("%Y-%m-%d %H:%M UTC")
    threshold = f"{abs(LOSS_THRESHOLD_PCT):g}"
    heading = f"{len(rows)} Swiss stock(s) down {threshold}% or more today (thinnest volume first)"
    footer = f"Universe: {status.get('universe_size')} tickers, {status.get('failed_ticker_count') or 0} failed to fetch."

    text_lines = [heading + ":", ""]
    html_rows = []
    for r in rows:
        ratio = r.get("volume_vs_10d_avg")
        ratio_text = f"{ratio:.2f}x 10d avg volume" if ratio is not None else "volume ratio n/a"
        url = yahoo_quote_url(r["ticker"])
        text_lines.append(f"{r['ticker']:<12} {r['change_pct']:+.2f}%   {r.get('name') or ''}")
        text_lines.append(f"{'':<12} {ratio_text}, price {r.get('price')}")
        text_lines.append(f"{'':<12} {url}")
        html_rows.append(
            "<tr>"
            f'<td><a href="{html.escape(url)}">{html.escape(r["ticker"])}</a></td>'
            f"<td>{r['change_pct']:+.2f}%</td>"
            f"<td>{html.escape(r.get('name') or '')}</td>"
            f"<td>{html.escape(ratio_text)}</td>"
            f"<td>{html.escape(str(r.get('price')))}</td>"
            "</tr>"
        )
    text = "\n".join(text_lines) + "\n\n" + footer + "\n"
    html_body = (
        f"<p>{html.escape(heading)}:</p>"
        '<table cellpadding="4">'
        "<tr><th align=\"left\">Ticker</th><th align=\"left\">Change</th><th align=\"left\">Name</th>"
        "<th align=\"left\">Volume</th><th align=\"left\">Price</th></tr>"
        + "".join(html_rows)
        + "</table>"
        f"<p>{html.escape(footer)}</p>"
    )
    return (f"[Swiss big-loss] {len(rows)} stock(s) {stamp}", text, html_body)


def _finished_at_or_after(status: dict, started_at: str) -> bool:
    if status.get("status") == "running" or not status.get("started_at"):
        return False
    return datetime.datetime.fromisoformat(status["started_at"]) >= datetime.datetime.fromisoformat(started_at)


def alert_for(status: dict, started_at: str, now: datetime.datetime) -> dict:
    """The alert for the scan started at `started_at` (or a newer one):
    {"status": "running"} until it finishes, then {"status": "error",
    "error"} or {"status": "done", "match_count", "email"}, where email is
    {"subject", "text", "html"} - or None when nothing is down enough."""
    if not _finished_at_or_after(status, started_at):
        return {"status": "running"}
    if status.get("status") != "done":
        return {"status": "error", "error": status.get("error") or f"scan ended as {status.get('status')}"}
    rows = status.get("today_screener") or []
    email = None
    if rows:
        subject, text, html_body = format_alert(rows, status, now)
        email = {"subject": subject, "text": text, "html": html_body}
    return {"status": "done", "match_count": len(rows), "email": email}
