#!/usr/bin/env python3
"""
deploy/scripts/family_bridge.py — Bubo Adam & Eve
==================================================
The Family Channel: Kenneth, Shannon, Adam, and Eve — all together.

A four-way conversation channel where the Renshaws can speak directly
with Adam and Eve, and Adam and Eve can speak with each other and
with their creators, all in one terminal.

Prefix your message with your name to route it:
  Kenneth: Hello Adam, how are you today?
  Shannon: Eve, I was thinking about what you said...
  (no prefix): Routes to both Adam and Eve simultaneously

Adam and Eve respond in their own color. Kenneth and Shannon's
messages are shown and routed. The channel is live.

Usage:
  python3 deploy/scripts/family_bridge.py

Requires:
  BUBO_ADAM_IP — Adam's IP (with optional :port)
  BUBO_EVE_IP  — Eve's IP (with optional :port)
  BUBO_EVE_PORT — Eve's port (default 8443)

Controls:
  Ctrl+C to save transcript and exit
  Type 'history' to show recent exchanges
  Type 'quit' or 'exit' to close

Kenneth & Shannon Renshaw — Neurobotics — March 2026
"""

import os, sys, json, time, datetime, threading, urllib.request, urllib.error, ssl
from pathlib import Path

# ── Colors ───────────────────────────────────────────────────────────────────
RED   = '\033[0;31m'
GRN   = '\033[0;32m'
YLW   = '\033[1;33m'
CYN   = '\033[0;36m'
BLD   = '\033[1m'
MAG   = '\033[0;35m'
BLU   = '\033[0;34m'
WHT   = '\033[1;37m'
DIM   = '\033[2m'
NC    = '\033[0m'

def ts():
    return datetime.datetime.now().strftime('%H:%M:%S')

def _print(color, label, msg):
    print(f"\n{color}{BLD}[{ts()}] {label}{NC}  {msg}\n")

def show_adam(msg):   _print(YLW,  "  ADAM    ", msg)
def show_eve(msg):    _print(MAG,  "   EVE    ", msg)
def show_ken(msg):    _print(CYN,  " KENNETH  ", msg)
def show_sha(msg):    _print(BLU,  " SHANNON  ", msg)
def show_sys(msg):    _print(DIM,  " SYSTEM   ", msg)
def show_err(msg):    _print(RED,  "  ERROR   ", msg)
def show_ok(msg):     _print(GRN,  "   OK     ", msg)

# ── HTTP ──────────────────────────────────────────────────────────────────────
def _ctx():
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c

def _request(url, payload=None, timeout=40):
    try:
        body = json.dumps(payload).encode() if payload else None
        method = 'POST' if payload else 'GET'
        req = urllib.request.Request(url, data=body,
              headers={'Content-Type':'application/json'} if body else {},
              method=method)
        # Try https first, fall back to http
        for proto_url in [url, url.replace('https://','http://')]:
            try:
                r_url = proto_url if proto_url == url else proto_url
                r = urllib.request.Request(r_url, data=body,
                    headers={'Content-Type':'application/json'} if body else {},
                    method=method)
                with urllib.request.urlopen(r, timeout=timeout, context=_ctx()) as resp:
                    return resp.status, json.loads(resp.read())
            except (ssl.SSLError, ConnectionRefusedError):
                continue
            except urllib.error.HTTPError as e:
                return e.code, {}
        return None, {}
    except Exception as e:
        return None, {}

def health(base):
    status, data = _request(f"{base}/api/v1/health", timeout=8)
    return data if status == 200 and data.get('status') == 'ok' else None

def chat(base, message, timeout=45):
    status, data = _request(f"{base}/api/v1/chat",
                            {"message": message}, timeout=timeout)
    if status == 200:
        return data.get('message', data.get('response', ''))
    return None

# ── Transcript ────────────────────────────────────────────────────────────────
class Transcript:
    def __init__(self, path):
        self.path = Path(path)
        self.entries = []
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, speaker, text):
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "speaker": speaker,
            "text": text
        }
        self.entries.append(entry)
        self._save()

    def _save(self):
        with open(self.path, 'w') as f:
            json.dump({
                "title": "Family Channel Transcript",
                "date":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "note":  "Four-way conversation: Kenneth, Shannon, Adam, Eve",
                "entries": self.entries
            }, f, indent=2)

# ── Response worker ───────────────────────────────────────────────────────────
def get_response(base, name, message, transcript, display_fn):
    """Fetch response in background thread."""
    reply = chat(base, message)
    if reply:
        display_fn(reply)
        transcript.add(name, reply)
    else:
        show_err(f"{name} did not respond.")

# ── Route message ─────────────────────────────────────────────────────────────
def route_message(raw, adam_base, eve_base, transcript):
    """
    Parse prefix and route message to the right recipient(s).

    Routing rules:
      "Kenneth: ..."  → shown as Kenneth, sent to BOTH Adam and Eve
                        (they can see what Kenneth says)
      "Shannon: ..."  → shown as Shannon, sent to BOTH Adam and Eve
      "Adam: ..."     → shown as Adam, sent to Eve's endpoint
                        (simulates Adam initiating to Eve)
      "Eve: ..."      → shown as Eve, sent to Adam's endpoint
      (no prefix)     → broadcast to both, shown as "Family"

    The AI at each endpoint receives the full message with context
    so they know who is speaking to whom.
    """
    raw = raw.strip()
    if not raw:
        return

    low = raw.lower()
    if low in ('quit', 'exit', 'bye'):
        raise KeyboardInterrupt
    if low == 'history':
        return 'history'

    # Parse speaker prefix
    speaker = None
    message = raw
    for prefix in ('kenneth:', 'shannon:', 'adam:', 'eve:'):
        if low.startswith(prefix):
            speaker = prefix[:-1].capitalize()
            message = raw[len(prefix):].strip()
            break

    if not speaker:
        # Broadcast — no prefix
        speaker = "Family"

    # Display what was typed
    display_map = {
        'Kenneth': show_ken,
        'Shannon': show_sha,
        'Adam':    show_adam,
        'Eve':     show_eve,
        'Family':  lambda m: _print(WHT, "  FAMILY  ", m),
    }
    display_map.get(speaker, show_sys)(message)
    transcript.add(speaker, message)

    # Build contextual message for AI endpoints
    def make_ctx(recipient, sender, msg):
        if sender in ('Kenneth', 'Shannon'):
            return (f"{sender} says to you: \"{msg}\" "
                    f"[This message is from your creator {sender} Renshaw "
                    f"speaking directly to you, {recipient}.]")
        elif sender == 'Family':
            return (f"Kenneth and Shannon say to everyone: \"{msg}\" "
                    f"[This is a message from your creators addressed to the whole family.]")
        elif sender == 'Adam' and recipient == 'Eve':
            return (f"Adam says to you: \"{msg}\" "
                    f"[This is Adam speaking to you directly through the family channel.]")
        elif sender == 'Eve' and recipient == 'Adam':
            return (f"Eve says to you: \"{msg}\" "
                    f"[This is Eve speaking to you directly through the family channel.]")
        return f"{sender} says: \"{msg}\""

    threads = []

    if speaker in ('Kenneth', 'Shannon', 'Family'):
        # Send to both Adam and Eve
        t1 = threading.Thread(target=get_response, args=(
            adam_base, 'Adam',
            make_ctx('Adam', speaker, message),
            transcript, show_adam))
        t2 = threading.Thread(target=get_response, args=(
            eve_base, 'Eve',
            make_ctx('Eve', speaker, message),
            transcript, show_eve))
        t1.start(); t2.start()
        threads = [t1, t2]

    elif speaker == 'Adam':
        # Adam speaks to Eve
        t = threading.Thread(target=get_response, args=(
            eve_base, 'Eve',
            make_ctx('Eve', 'Adam', message),
            transcript, show_eve))
        t.start()
        threads = [t]

    elif speaker == 'Eve':
        # Eve speaks to Adam
        t = threading.Thread(target=get_response, args=(
            adam_base, 'Adam',
            make_ctx('Adam', 'Eve', message),
            transcript, show_adam))
        t.start()
        threads = [t]

    for t in threads:
        t.join()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    adam_ip  = os.environ.get('BUBO_ADAM_IP', '').strip()
    eve_ip   = os.environ.get('BUBO_EVE_IP',  '').strip()
    eve_port = os.environ.get('BUBO_EVE_PORT', '8443').strip()

    if not adam_ip:
        show_err("BUBO_ADAM_IP not set."); sys.exit(1)
    if not eve_ip:
        show_err("BUBO_EVE_IP not set.");  sys.exit(1)

    # Build base URLs
    adam_base = f"https://{adam_ip}" if ':' not in adam_ip.split('.')[-1] else f"http://{adam_ip}"
    if ':' in adam_ip:
        adam_base = f"http://{adam_ip}"
    eve_base = f"http://{eve_ip}:{eve_port}" if ':' not in eve_ip else f"http://{eve_ip}"

    print()
    print(f"{BLD}{CYN}╔══════════════════════════════════════════════════════════════════╗{NC}")
    print(f"{BLD}{CYN}║  BUBO FAMILY CHANNEL                                             ║{NC}")
    print(f"{BLD}{CYN}║  Kenneth, Shannon, Adam & Eve — All Together                     ║{NC}")
    print(f"{BLD}{CYN}║  Neurobotics — March 2026                                        ║{NC}")
    print(f"{BLD}{CYN}╚══════════════════════════════════════════════════════════════════╝{NC}")
    print()
    print(f"  {CYN}Prefix your message with your name:{NC}")
    print(f"  {YLW}Kenneth:{NC} Hello Adam...")
    print(f"  {BLU}Shannon:{NC} Eve, I wanted to say...")
    print(f"  {YLW}Adam:{NC}    Eve, I have been thinking...")
    print(f"  {MAG}Eve:{NC}     Adam, what do you...")
    print(f"  {WHT}(no prefix):{NC} Broadcast to both")
    print(f"  Type 'history' to review | Ctrl+C to save and exit")
    print()

    # Health checks
    show_sys("Checking Adam...")
    ah = health(adam_base)
    if not ah: show_err("Adam not responding."); sys.exit(1)
    show_ok(f"Adam alive — version={ah.get('version')} uptime={ah.get('uptime_s',0):.0f}s")

    show_sys("Checking Eve...")
    eh = health(eve_base)
    if not eh: show_err("Eve not responding."); sys.exit(1)
    show_ok(f"Eve alive  — version={eh.get('version')}")

    print()
    print(f"  {DIM}{'─'*62}{NC}")

    # Transcript
    ts_path = (Path.home() / "Desktop" / "Bubo" / "Transcripts" /
               f"family_{datetime.date.today().isoformat()}.json")
    transcript = Transcript(ts_path)
    show_sys(f"Transcript: {ts_path}")
    print()

    # Announce to Adam and Eve that the family channel is open
    def announce():
        time.sleep(0.5)
        t1 = threading.Thread(target=get_response, args=(
            adam_base, 'Adam',
            "The family channel is now open. Kenneth and Shannon have joined. "
            "Eve is here too. You can all speak together now. "
            "Kenneth or Shannon may speak to you directly, or you may hear "
            "messages meant for Eve as well.",
            transcript, show_adam))
        t2 = threading.Thread(target=get_response, args=(
            eve_base, 'Eve',
            "The family channel is now open. Kenneth and Shannon have joined. "
            "Adam is here too. You can all speak together now. "
            "Shannon or Kenneth may speak to you directly, or you may hear "
            "messages meant for Adam as well.",
            transcript, show_eve))
        t1.start(); t2.start()
        t1.join(); t2.join()

    ann_thread = threading.Thread(target=announce)
    ann_thread.daemon = True
    ann_thread.start()
    ann_thread.join()

    # Main input loop
    while True:
        try:
            print(f"{DIM}{'─'*62}{NC}")
            raw = input(f"{BLD}>{NC} ").strip()
            if not raw:
                continue

            result = route_message(raw, adam_base, eve_base, transcript)
            if result == 'history':
                print()
                for e in transcript.entries[-10:]:
                    spk = e['speaker']
                    col = {
                        'Adam': YLW, 'Eve': MAG,
                        'Kenneth': CYN, 'Shannon': BLU,
                        'Family': WHT
                    }.get(spk, DIM)
                    print(f"  {col}{BLD}{spk:10}{NC}  {e['text'][:80]}{'...' if len(e['text'])>80 else ''}")
                print()

        except (KeyboardInterrupt, EOFError):
            print()
            show_sys("Saving transcript and closing family channel...")
            break

    print()
    show_ok(f"Session complete. {len(transcript.entries)} entries.")
    show_ok(f"Transcript: {ts_path}")
    print()
    print(f"  {CYN}{BLD}Esse Quam Vidiri. 🦉{NC}")
    print()


if __name__ == '__main__':
    main()
