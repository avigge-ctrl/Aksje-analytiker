"""Daglig skanner: støtte/motstand + Williams %R (-5 / -95) for Oslo Børs-aksjer.
Kjøres etter børsslutt. Sender én samlet e-post via Gmail SMTP når noe er nytt.

Miljøvariabler (kun for e-post): GMAIL_USER, GMAIL_APP_PASSWORD, MAIL_TO
Bruk:  python scan.py            (sender e-post hvis det finnes nye varsler)
       python scan.py --dry      (skriver bare ut, sender ikke, endrer ikke state)
"""
import csv, html as _h, json, os, smtplib, sys
from datetime import datetime, date
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---- Innstillinger ----
WR_PERIOD   = 14      # Williams %R-periode
OVERBOUGHT  = -5      # >= -5  -> nær 0 (overkjøpt)
OVERSOLD    = -95     # <= -95 -> nær -100 (oversolgt)
NEAR_OB     = -10     # tidligere varsel: >= -10 "nærmer seg" overkjøpt
NEAR_OS     = -90     # tidligere varsel: <= -90 "nærmer seg" oversolgt
COMBO_LOW   = -80     # signal: nær støtte OG Williams %R <= -80
COMBO_HIGH  = -20     # signal: nær motstand OG Williams %R >= -20
ATR_PERIOD  = 14
SWING_K     = 5       # bars på hver side for å regne en swing high/low
CLUSTER_ATR = 0.5     # soner slås sammen innenfor 0,5 x ATR
NEAR_ATR    = 0.75    # "nær" sone = innenfor 0,75 x ATR
MIN_TOUCHES = 2       # minst så mange swing-punkter i sonen
LOOKBACK    = "1y"

HERE = Path(__file__).parent
STATE_FILE = HERE / "state.json"
WATCHLIST = HERE / "watchlist.csv"


def williams_r(df, n=WR_PERIOD):
    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    return (hh - df["close"]) / (hh - ll) * -100


def atr(df, n=ATR_PERIOD):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def swing_levels(df, k=SWING_K):
    levels = []
    for i in range(k, len(df) - k):
        w = df.iloc[i - k:i + k + 1]
        if df["high"].iloc[i] == w["high"].max():
            levels.append(float(df["high"].iloc[i]))
        if df["low"].iloc[i] == w["low"].min():
            levels.append(float(df["low"].iloc[i]))
    return sorted(levels)


def cluster(levels, tol):
    """Returnerer liste av (sonenivå, antall berøringer)."""
    if not levels:
        return []
    zones, cur = [], [levels[0]]
    for lv in levels[1:]:
        if lv - np.mean(cur) <= tol:
            cur.append(lv)
        else:
            zones.append((float(np.mean(cur)), len(cur)))
            cur = [lv]
    zones.append((float(np.mean(cur)), len(cur)))
    return zones


def analyze(df):
    """Returnerer (pris, wr, atr, {nøkkel: tekst})."""
    price = float(df["close"].iloc[-1])
    a = float(atr(df).iloc[-1])
    wr = float(williams_r(df).iloc[-1])
    alerts = {}
    if wr >= OVERBOUGHT:
        alerts["WR_OB"] = f"Williams %R {wr:.1f}: overkjøpt (≥ {OVERBOUGHT})"
    elif wr >= NEAR_OB:
        alerts["WR_OB_NEAR"] = f"Williams %R {wr:.1f}: nærmer seg overkjøpt (≥ {NEAR_OB})"
    if wr <= OVERSOLD:
        alerts["WR_OS"] = f"Williams %R {wr:.1f}: oversolgt (≤ {OVERSOLD})"
    elif wr <= NEAR_OS:
        alerts["WR_OS_NEAR"] = f"Williams %R {wr:.1f}: nærmer seg oversolgt (≤ {NEAR_OS})"
    zones = [z for z in cluster(swing_levels(df), CLUSTER_ATR * a) if z[1] >= MIN_TOUCHES]
    near = [z for z in zones if abs(price - z[0]) <= NEAR_ATR * a]
    for lvl, touches in near:
        kind = "RES" if lvl > price else "SUP"
        label = "motstand" if kind == "RES" else "støtte"
        # behold nærmeste sone per type
        prev = alerts.get(kind)
        if prev is None or abs(price - lvl) < prev[0]:
            alerts[kind] = (abs(price - lvl),
                            f"Nær {label} ved {lvl:.2f} ({touches} berøringer, "
                            f"{abs(price - lvl) / price * 100:.1f} % unna)")
    alerts = {k: (v[1] if isinstance(v, tuple) else v) for k, v in alerts.items()}
    if "SUP" in alerts and wr <= COMBO_LOW:
        alerts["BUY"] = f"Nær støtte og Williams %R {wr:.1f} (≤ {COMBO_LOW})"
    if "RES" in alerts and wr >= COMBO_HIGH:
        alerts["SELL"] = f"Nær motstand og Williams %R {wr:.1f} (≥ {COMBO_HIGH})"
    below = [z for z in zones if z[0] <= price]
    above = [z for z in zones if z[0] > price]
    levels = {"sup": max(below, key=lambda z: z[0]) if below else None,
              "res": min(above, key=lambda z: z[0]) if above else None}
    return price, wr, a, alerts, levels


def load_watchlist():
    with open(WATCHLIST, newline="", encoding="utf-8") as f:
        return [(r["ticker"].strip(), r["name"].strip()) for r in csv.DictReader(f)]


def fetch(ticker):
    df = yf.Ticker(ticker).history(period=LOOKBACK, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    return df if len(df) > 60 else None


COUNTRIES = {".OL": "Norge", ".ST": "Sverige", ".CO": "Danmark", ".HE": "Finland",
             ".VI": "Østerrike", ".MI": "Italia", ".PA": "Frankrike", ".DE": "Tyskland",
             ".AS": "Nederland", ".MC": "Spania", ".BR": "Belgia", ".LS": "Portugal",
             ".IR": "Irland"}
COUNTRY_ORDER = ["Norge", "Sverige", "Danmark", "Finland"]


def country_of(ticker):
    return COUNTRIES.get(ticker[ticker.rfind("."):].upper(), "Annet")


def report_days(ticker):
    """(dager siden siste rapport, dager til neste rapport). None hvis ukjent."""
    today = date.today()
    past, future = [], []
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=12)
        if ed is not None and len(ed):
            for ts in ed.index:
                d = ts.date()
                (past if d <= today else future).append(d)
    except Exception:
        pass
    if not future:  # reserve: kalenderen
        try:
            cal = yf.Ticker(ticker).calendar or {}
            for d in cal.get("Earnings Date", []) or []:
                if d > today:
                    future.append(d)
        except Exception:
            pass
    last = (today - max(past)).days if past else None
    nxt = (min(future) - today).days if future else None
    return last, nxt


def rpt_text(it):
    a = "–" if it["last_d"] is None else f'{it["last_d"]} d siden'
    b = "–" if it["next_d"] is None else ("i dag" if it["next_d"] == 0 else f'om {it["next_d"]} d')
    return a, b


BADGES = {  # nøkkel: (tekst, tekstfarge, bakgrunn)
    "BUY":        (f"Nær støtte + %R ≤ {COMBO_LOW}",  "#ffffff", "#12b76a"),
    "SELL":       (f"Nær motstand + %R ≥ {COMBO_HIGH}", "#ffffff", "#d92d20"),
    "WR_OB":      ("Overkjøpt",              "#b42318", "#fee4e2"),
    "WR_OB_NEAR": ("Nærmer seg overkjøpt",   "#b54708", "#fef0c7"),
    "WR_OS":      ("Oversolgt",              "#067647", "#d1fadf"),
    "WR_OS_NEAR": ("Nærmer seg oversolgt",   "#3b7c0f", "#ecfdf3"),
    "SUP":        ("Nær støtte",             "#067647", "#d1fadf"),
    "RES":        ("Nær motstand",           "#b42318", "#fee4e2"),
}


def fmt(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def gauge(wr, width=110):
    """Liten måler: -100 (venstre, oversolgt) til 0 (høyre, overkjøpt)."""
    pos = int(max(1, min(99, round(wr + 100))))
    color = "#d92d20" if wr >= NEAR_OB else "#12b76a" if wr <= NEAR_OS else "#98a2b3"
    return (f'<table role="presentation" cellpadding="0" cellspacing="0" width="{width}" '
            f'style="border-collapse:collapse;background:#eaecf0;border-radius:3px">'
            f'<tr><td width="{pos}%" style="background:{color};height:6px;line-height:6px;'
            f'font-size:0;border-radius:3px">&nbsp;</td>'
            f'<td width="{100 - pos}%" style="height:6px;line-height:6px;font-size:0">&nbsp;</td>'
            f'</tr></table>')


def pct_str(lv, price):
    p = (lv[0] - price) / price * 100
    return f"{p:+.1f} %".replace(".", ",").replace("-", "−")


def build_html(items, problems):
    esc = _h.escape
    today = datetime.now().strftime("%d.%m.%Y")

    def dist(it, key):
        lv = it["levels"][key]
        return abs(lv[0] - it["price"]) / it["price"] if lv else 9
    buy = sorted([i for i in items if "BUY" in i["alerts"]], key=lambda i: dist(i, "sup"))
    sell = sorted([i for i in items if "SELL" in i["alerts"]], key=lambda i: dist(i, "res"))
    others = [i for i in items if i["fresh"] and "BUY" not in i["alerts"] and "SELL" not in i["alerts"]]
    n_new = sum(1 for i in items if i["fresh"])

    def lvl_block(label, lv, price, hot, color):
        if not lv:
            return (f'<div style="font-size:11px;color:#667085">{label}</div>'
                    '<div style="font-size:13px;color:#98a2b3">–</div>')
        w = "700" if hot else "600"
        return (f'<div style="font-size:11px;color:#667085">{label}</div>'
                f'<div style="font-size:14px;font-weight:{w};color:#101828">{fmt(lv[0])} '
                f'<span style="font-size:12px;font-weight:400;color:{color if hot else "#667085"}">'
                f'{pct_str(lv, price)}</span></div>')

    def card(it, keys, accent="#e4e7ec", is_new=False):
        tag = (' <span style="font-size:10px;font-weight:700;color:#175cd3;background:#d1e9ff;'
               'padding:1px 6px;border-radius:8px;vertical-align:middle">NY</span>') if is_new else ""
        badges = ""
        for k in sorted(keys, key=lambda x: list(BADGES).index(x)):
            detail = ""
            if k in ("SUP", "RES"):
                detail = f' <span style="font-size:13px;color:#475467">{esc(it["alerts"][k])}</span>'
            badges += (f'<div style="margin:7px 0 0 0"><span style="display:inline-block;padding:2px 9px;'
                       f'border-radius:10px;font-size:12px;font-weight:600;color:{BADGES[k][1]};'
                       f'background:{BADGES[k][2]}">{BADGES[k][0]}</span>{detail}</div>')
        lv = it["levels"]
        return (
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
            f'style="border:1px solid #e4e7ec;border-left:5px solid {accent};border-radius:10px;'
            f'border-collapse:separate;margin:0 0 12px 0">'
            '<tr><td style="padding:14px 16px">'
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>'
            f'<td style="font-size:16px;font-weight:700;color:#101828">{esc(it["name"])} '
            f'<span style="font-weight:400;font-size:13px;color:#667085">{esc(it["ticker"])}</span>{tag}</td>'
            f'<td align="right" style="font-size:16px;font-weight:700;color:#101828">{fmt(it["price"])}</td>'
            '</tr></table>'
            f'{badges}'
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin-top:12px"><tr>'
            f'<td width="50%" valign="top">{lvl_block("Støtte", lv["sup"], it["price"], "SUP" in it["alerts"], "#067647")}</td>'
            f'<td width="50%" valign="top">{lvl_block("Motstand", lv["res"], it["price"], "RES" in it["alerts"], "#b42318")}</td>'
            '</tr></table>'
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin-top:12px"><tr>'
            f'<td style="font-size:12px;color:#667085">Williams %R <b style="color:#101828">{fmt(it["wr"], 1)}</b>'
            f'<div style="margin-top:3px">Rapport: siste {rpt_text(it)[0]} · neste {rpt_text(it)[1]}</div></td>'
            f'<td align="right">{gauge(it["wr"])}</td></tr></table>'
            '</td></tr></table>')

    def section(title, sub, color, body, empty):
        inner = body or f'<p style="font-size:13px;color:#667085;margin:0 0 12px 0">{empty}</p>'
        return (f'<div style="margin:18px 0 10px 0;padding-left:10px;border-left:4px solid {color}">'
                f'<div style="font-size:15px;font-weight:700;color:#101828">{title}</div>'
                f'<div style="font-size:12px;color:#667085">{sub}</div></div>' + inner)

    buy_html = "".join(card(i, list(i["alerts"]), "#12b76a", bool(i["fresh"])) for i in buy)
    sell_html = "".join(card(i, list(i["alerts"]), "#d92d20", bool(i["fresh"])) for i in sell)
    def compact_row(it):
        bd = "".join(
            f'<span style="display:inline-block;margin:0 4px 2px 0;font-size:10px;font-weight:600;'
            f'color:{BADGES[k][1]};background:{BADGES[k][2]};padding:1px 6px;border-radius:8px;white-space:nowrap">'
            f'{BADGES[k][0]}</span>' for k in sorted(it["fresh"], key=lambda x: list(BADGES).index(x)))
        return (f'<tr><td style="padding:7px 10px;font-size:13px;color:#101828;border-bottom:1px solid #f2f4f7">'
                f'{esc(it["name"])} <span style="color:#667085;font-size:11px">{esc(it["ticker"])}</span>'
                f'<div style="margin-top:3px">{bd}</div>'
                f'<div style="font-size:11px;color:#667085;margin-top:2px">Rapport: siste {rpt_text(it)[0]} · neste {rpt_text(it)[1]}</div></td>'
                f'<td align="right" valign="top" style="padding:7px 10px;font-size:13px;color:#101828;'
                f'border-bottom:1px solid #f2f4f7">{fmt(it["price"])}'
                f'<div style="font-size:11px;color:#667085">%R {fmt(it["wr"], 1)}</div></td></tr>')
    oth_html = ('<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
                'style="border:1px solid #e4e7ec;border-radius:8px;border-collapse:separate">'
                + "".join(compact_row(i) for i in others) + '</table>') if others else ""

    body = section(f"Nær støtte og Williams %R ≤ {COMBO_LOW}",
                   "Sortert etter avstand til støtte", "#12b76a", buy_html, "Ingen aksjer akkurat nå.")
    body += section(f"Nær motstand og Williams %R ≥ {COMBO_HIGH}",
                    "Sortert etter avstand til motstand", "#d92d20", sell_html, "Ingen aksjer akkurat nå.")
    if others:
        body += section("Andre nye varsler", "", "#98a2b3", oth_html, "")

    def small_lvl(lv, price, hot, color):
        if not lv:
            return '<span style="color:#98a2b3">–</span>'
        c = color if hot else "#667085"
        w = "700" if hot else "400"
        return (f'<div style="font-size:13px;color:#101828;font-weight:600">{fmt(lv[0])}</div>'
                f'<div style="font-size:11px;font-weight:{w};color:{c}">{pct_str(lv, price)}</div>')

    th = 'style="padding:7px 8px;font-size:11px;font-weight:600;color:#667085;text-align:{};border-bottom:1px solid #e4e7ec"'
    order = COUNTRY_ORDER + sorted({i["country"] for i in items} - set(COUNTRY_ORDER))
    parts = []
    for c in order:
        grp = sorted([i for i in items if i["country"] == c],
                     key=lambda i: (i["next_d"] is None, i["next_d"] if i["next_d"] is not None else 0, i["name"]))
        if not grp:
            continue
        rows = []
        for idx, it in enumerate(grp):
            bg = "#ecfdf3" if "BUY" in it["alerts"] else "#fef3f2" if "SELL" in it["alerts"] \
                else "#f9fafb" if idx % 2 else "#ffffff"
            ks = sorted(it["alerts"], key=lambda x: list(BADGES).index(x))
            status = "".join(
                f'<span style="display:inline-block;margin:3px 3px 0 0;font-size:10px;font-weight:600;color:{BADGES[k][1]};'
                f'background:{BADGES[k][2]};padding:1px 6px;border-radius:8px;white-space:nowrap">{BADGES[k][0]}</span>'
                for k in ks)
            lv = it["levels"]
            lvl = ""
            if it["alerts"]:
                lvl = ('<div style="font-size:11px;color:#667085;margin-top:3px">'
                       f'Støtte {fmt(lv["sup"][0]) + " (" + pct_str(lv["sup"], it["price"]) + ")" if lv["sup"] else "–"}'
                       f' · Motstand {fmt(lv["res"][0]) + " (" + pct_str(lv["res"], it["price"]) + ")" if lv["res"] else "–"}</div>')
            last, nxt = rpt_text(it)
            soon = it["next_d"] is not None and it["next_d"] <= 7
            nstyle = "font-weight:700;color:#b54708" if soon else "color:#101828"
            rows.append(
                f'<tr style="background:{bg}">'
                f'<td valign="top" style="padding:7px 8px;font-size:13px;color:#101828">{esc(it["name"])}'
                f' <span style="color:#98a2b3;font-size:10px">{esc(it["ticker"])}</span>{("<div>" + status + "</div>") if status else ""}{lvl}</td>'
                f'<td align="right" valign="top" style="padding:7px 8px;font-size:13px;color:#101828">{fmt(it["price"])}</td>'
                f'<td align="right" valign="top" style="padding:7px 8px;font-size:12px;color:#101828">{fmt(it["wr"], 0)}</td>'
                f'<td align="right" valign="top" style="padding:7px 8px;font-size:12px;color:#475467;white-space:nowrap">{last}</td>'
                f'<td align="right" valign="top" style="padding:7px 8px;font-size:12px;white-space:nowrap;{nstyle}">{nxt}</td></tr>')
        parts.append(
            f'<tr><td colspan="5" style="padding:9px 8px;font-size:13px;font-weight:700;color:#ffffff;background:#344054">'
            f'{esc(c)} <span style="font-weight:400;color:#d0d5dd">· {len(grp)} aksjer</span></td></tr>' + "".join(rows))
    overview = (
        '<h3 style="font-size:14px;color:#101828;margin:26px 0 4px 0">Alle aksjer per land</h3>'
        '<p style="font-size:12px;color:#667085;margin:0 0 8px 0">Sortert etter nærmeste neste rapport. '
        'Neste rapportdato er hentet fra Yahoo og kan være et estimat. Oransje = rapport innen 7 dager. '
        'Williams %R: −100 = oversolgt, 0 = overkjøpt.</p>'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="border:1px solid #e4e7ec;border-radius:8px;border-collapse:separate;overflow:hidden">'
        f'<tr><td {th.format("left")}>Selskap</td><td {th.format("right")}>Kurs</td>'
        f'<td {th.format("right")}>%R</td><td {th.format("right")}>Siste rapport</td>'
        f'<td {th.format("right")}>Neste</td></tr>' + "".join(parts) + '</table>')

    prob_html = ""
    if problems:
        prob_html = (
            '<div style="margin-top:16px;padding:10px 12px;background:#fffaeb;border:1px solid #fedf89;'
            'border-radius:8px;font-size:12px;color:#93370d"><b>Problemer med data</b><br>'
            + "<br>".join(esc(p) for p in problems) + '</div>')

    pre = f"{len(buy)} nær støtte · {len(sell)} nær motstand · {n_new} nye varsler"
    return (
        '<!DOCTYPE html><html lang="no"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"></head>'
        '<body style="margin:0;padding:0;background:#f2f4f7;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">'
        f'<div style="display:none;max-height:0;overflow:hidden;opacity:0">{pre}</div>'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f2f4f7">'
        '<tr><td align="center" style="padding:20px 10px">'
        '<table role="presentation" width="600" cellpadding="0" cellspacing="0" '
        'style="max-width:600px;width:100%;background:#ffffff;border-radius:12px;border-collapse:separate;overflow:hidden">'
        '<tr><td style="background:#0b2545;padding:18px 22px">'
        '<div style="font-size:20px;font-weight:700;color:#ffffff">Aksjevarsel</div>'
        f'<div style="font-size:13px;color:#b9c7dc;margin-top:2px">{today} · {pre}</div></td></tr>'
        '<tr><td style="padding:6px 22px 20px 22px">'
        f'{body}{overview}{prob_html}'
        '<p style="font-size:11px;color:#98a2b3;margin:22px 0 0 0">Tekniske signaler basert på daglige '
        'kurser. Ikke investeringsråd.</p>'
        '</td></tr></table></td></tr></table></body></html>')


def lvl_text(lv, price):
    return f"{lv[0]:.2f} ({(lv[0] - price) / price * 100:+.1f} %)" if lv else "-"


def build_text(items, problems):
    def line(it):
        return (f"  {it['name']} ({it['ticker']})  kurs {it['price']:.2f}  %R {it['wr']:.1f}  "
                f"støtte {lvl_text(it['levels']['sup'], it['price'])}  "
                f"motstand {lvl_text(it['levels']['res'], it['price'])}"
                f"{'  [NY]' if it['fresh'] else ''}")
    lines = [f"NÆR STØTTE OG WILLIAMS %R <= {COMBO_LOW}:"]
    b = [i for i in items if "BUY" in i["alerts"]]
    lines += [line(i) for i in b] or ["  Ingen akkurat nå."]
    lines += ["", f"NÆR MOTSTAND OG WILLIAMS %R >= {COMBO_HIGH}:"]
    sl = [i for i in items if "SELL" in i["alerts"]]
    lines += [line(i) for i in sl] or ["  Ingen akkurat nå."]
    oth = [i for i in items if i["fresh"] and "BUY" not in i["alerts"] and "SELL" not in i["alerts"]]
    if oth:
        lines += ["", "ANDRE NYE VARSLER:"]
        for it in oth:
            lines.append(line(it))
            lines += [f"    - {it['alerts'][k]}" for k in it["fresh"]]
    lines += ["", "ALLE AKSJER PER LAND (sortert etter neste rapport; Williams %R: -100 = oversolgt, 0 = overkjøpt):"]
    order = COUNTRY_ORDER + sorted({i["country"] for i in items} - set(COUNTRY_ORDER))
    for c in order:
        grp = sorted([i for i in items if i["country"] == c],
                     key=lambda i: (i["next_d"] is None, i["next_d"] if i["next_d"] is not None else 0, i["name"]))
        if not grp:
            continue
        lines += ["", f"{c.upper()} ({len(grp)})"]
        for it in grp:
            last, nxt = rpt_text(it)
            flag = "  *" + ",".join(sorted(it["alerts"])) if it["alerts"] else ""
            lines.append(f"  {it['name']:<24} kurs {it['price']:>9.2f}  %R {it['wr']:>6.1f}  "
                         f"siste rapport {last:<11} neste {nxt}{flag}")
    if problems:
        lines += ["", "Problemer med data:"] + [f"  - {p}" for p in problems]
    lines += ["", "Tekniske signaler, ikke investeringsråd."]
    return "\n".join(lines)


def send_mail(subject, text, html):
    user = os.environ["GMAIL_USER"]
    pw = os.environ["GMAIL_APP_PASSWORD"]
    to = os.environ.get("MAIL_TO", user)
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(user, pw)
        s.send_message(msg)


def main():
    dry = "--dry" in sys.argv
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    new_state, items, problems = {}, [], []

    for ticker, name in load_watchlist():
        try:
            df = fetch(ticker)
        except Exception as e:
            df = None
            print(f"{ticker}: feil {e}")
        if df is None:
            problems.append(f"{ticker} ({name}): ingen/for lite data")
            new_state[ticker] = state.get(ticker, [])  # behold forrige status
            continue
        price, wr, a, alerts, levels = analyze(df)
        new_state[ticker] = sorted(alerts)
        # i --dry vises alle aktive varsler, ellers bare nye siden forrige kjøring
        fresh = [k for k in alerts if dry or k not in state.get(ticker, [])]
        fresh.sort(key=lambda k: list(BADGES).index(k))
        print(f"{ticker:9} kurs {price:8.2f}  %R {wr:7.1f}  aktive: {sorted(alerts) or '-'}")
        last_d, next_d = report_days(ticker)
        items.append(dict(name=name, ticker=ticker, price=price, wr=wr, alerts=alerts, fresh=fresh,
                          levels=levels, country=country_of(ticker), last_d=last_d, next_d=next_d))

    hits = [i for i in items if i["fresh"]]
    if hits or problems:
        names = [i["name"] for i in hits]
        subject = "Aksjevarsel: " + ", ".join(names[:3]) + (f" +{len(names) - 3}" if len(names) > 3 else "")
        nb = [i["name"] for i in hits if "BUY" in i["fresh"]]
        ns = [i["name"] for i in hits if "SELL" in i["fresh"]]
        parts = ([f"støtte+%R: {', '.join(nb[:3])}"] if nb else []) + \
                ([f"motstand+%R: {', '.join(ns[:3])}"] if ns else [])
        if parts:
            subject = "Aksjevarsel: " + " · ".join(parts)
        if not hits:
            subject = "Aksjeskanner: problemer med data"
        text, html = build_text(items, problems), build_html(items, problems)
        if dry:
            (HERE / "preview.html").write_text(html, encoding="utf-8")
            print("\n" + subject + "\n" + text + "\n\n(HTML-forhåndsvisning lagret i preview.html)")
        elif os.environ.get("GMAIL_USER"):
            send_mail(subject, text, html)
    elif dry:
        (HERE / "preview.html").write_text(build_html(items, problems), encoding="utf-8")
        print("\nIngen varsler nå. Oversikts-e-post lagret i preview.html")
    if not dry:
        STATE_FILE.write_text(json.dumps(new_state, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
