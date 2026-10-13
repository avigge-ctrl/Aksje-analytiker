"""Daglig skanner: støtte/motstand + Williams %R (-5 / -95) for Oslo Børs-aksjer.
Kjøres etter børsslutt. Sender én samlet e-post via Gmail SMTP når noe er nytt.

Miljøvariabler (kun for e-post): GMAIL_USER, GMAIL_APP_PASSWORD, MAIL_TO
Bruk:  python scan.py            (sender e-post hvis det finnes nye varsler)
       python scan.py --dry      (skriver bare ut, sender ikke, endrer ikke state)
"""
import csv, json, os, smtplib, sys
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---- Innstillinger ----
WR_PERIOD   = 14      # Williams %R-periode
OVERBOUGHT  = -5      # >= -5  -> nær 0 (overkjøpt)
OVERSOLD    = -95     # <= -95 -> nær -100 (oversolgt)
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
        alerts["WR_OB"] = f"Williams %R {wr:.1f} (nær overkjøpt, ≥ {OVERBOUGHT})"
    if wr <= OVERSOLD:
        alerts["WR_OS"] = f"Williams %R {wr:.1f} (nær oversolgt, ≤ {OVERSOLD})"
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
    return price, wr, a, alerts


def load_watchlist():
    with open(WATCHLIST, newline="", encoding="utf-8") as f:
        return [(r["ticker"].strip(), r["name"].strip()) for r in csv.DictReader(f)]


def fetch(ticker):
    df = yf.Ticker(ticker).history(period=LOOKBACK, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    return df if len(df) > 60 else None


def send_mail(subject, body):
    user = os.environ["GMAIL_USER"]
    pw = os.environ["GMAIL_APP_PASSWORD"]
    to = os.environ.get("MAIL_TO", user)
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(user, pw)
        s.send_message(msg)


def main():
    dry = "--dry" in sys.argv
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    new_state, lines, problems = {}, [], []

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
        price, wr, a, alerts = analyze(df)
        new_state[ticker] = sorted(alerts)
        fresh = [k for k in alerts if k not in state.get(ticker, [])]
        print(f"{ticker:9} kurs {price:8.2f}  %R {wr:7.1f}  aktive: {sorted(alerts) or '-'}")
        if fresh:
            lines.append(f"{name} ({ticker})  kurs {price:.2f}")
            lines += [f"  - {alerts[k]}" for k in fresh]
            lines.append("")

    if lines or problems:
        body = "\n".join(lines)
        if problems:
            body += "\nProblemer med data:\n" + "\n".join(f"  - {p}" for p in problems)
        body += "\n\nTekniske signaler, ikke investeringsråd."
        subject = f"Aksjevarsel: {sum(1 for l in lines if l and not l.startswith(' '))} aksjer"
        print("\n" + subject + "\n" + body)
        if not dry and lines and os.environ.get("GMAIL_USER"):
            send_mail(subject, body)
    if not dry:
        STATE_FILE.write_text(json.dumps(new_state, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
