# relay_server.py  —  Flappy 1v1 Relay
# Deploy this on Railway, Render, or any VPS.
# Both players connect here — no port forwarding needed.
#
# Deploy on Railway:
#   1. Push this file + requirements.txt to a GitHub repo
#   2. New project on railway.app → Deploy from GitHub
#   3. Set start command:  python relay_server.py
#   4. Railway gives you a public URL like:  flappy.up.railway.app
#   Paste that URL into RELAY_HOST in flappy_1v1.py
#
# Deploy on Render (also free):
#   1. New Web Service → connect GitHub repo
#   2. Start command:  python relay_server.py
#   3. Paste the .onrender.com URL into RELAY_HOST in flappy_1v1.py

import socket, threading, json, random, string, time, os

PORT = int(os.environ.get("PORT", 55201))

# rooms: { code: { "host": conn, "guest": conn, "created": time } }
rooms = {}
rooms_lock = threading.Lock()

def gen_code():
    while True:
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        if code not in rooms:
            return code

def send(conn, msg):
    try:
        conn.sendall((json.dumps(msg) + "\n").encode())
    except:
        pass

def recv_loop(conn):
    """Generator: yields parsed messages from conn."""
    buf = b""
    while True:
        try:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    yield json.loads(line.decode())
                except:
                    pass
        except:
            break

def handle_client(conn, addr):
    print(f"[+] {addr}")
    role = None
    code = None

    for msg in recv_loop(conn):
        t = msg.get("type")

        # ── HOST: create a room ───────────────────────────────────────────
        if t == "host":
            with rooms_lock:
                code = gen_code()
                rooms[code] = {"host": conn, "guest": None, "created": time.time()}
            role = "host"
            send(conn, {"type": "hosted", "code": code})
            print(f"[room {code}] hosted by {addr}")

        # ── GUEST: join a room ────────────────────────────────────────────
        elif t == "join":
            code = msg.get("code", "").upper().strip()
            with rooms_lock:
                room = rooms.get(code)
                if room is None:
                    send(conn, {"type": "error", "msg": "Room not found"})
                    continue
                if room["guest"] is not None:
                    send(conn, {"type": "error", "msg": "Room is full"})
                    continue
                room["guest"] = conn
            role = "guest"
            send(conn, {"type": "joined", "code": code})
            # tell host guest arrived + send seed
            seed = random.randint(0, 10**9)
            with rooms_lock:
                rooms[code]["seed"] = seed
            send(room["host"], {"type": "guest_joined"})
            send(room["host"], {"type": "start", "seed": seed, "role": "host"})
            send(conn,         {"type": "start", "seed": seed, "role": "guest"})
            print(f"[room {code}] guest joined from {addr} — starting")

        # ── RELAY: forward everything else to the opponent ────────────────
        elif role and code:
            with rooms_lock:
                room = rooms.get(code)
            if not room:
                break
            opponent = room["guest"] if role == "host" else room["host"]
            if opponent:
                send(opponent, msg)

    # ── cleanup ───────────────────────────────────────────────────────────
    print(f"[-] {addr} disconnected")
    if code:
        with rooms_lock:
            room = rooms.get(code)
            if room:
                opponent = room["guest"] if role == "host" else room["host"]
                if opponent:
                    send(opponent, {"type": "opponent_disconnected"})
                del rooms[code]
    try: conn.close()
    except: pass

def cleanup_old_rooms():
    """Remove rooms older than 10 minutes with no guest."""
    while True:
        time.sleep(60)
        now = time.time()
        with rooms_lock:
            stale = [c for c, r in rooms.items()
                     if r["guest"] is None and now - r["created"] > 600]
            for c in stale:
                try: rooms[c]["host"].close()
                except: pass
                del rooms[c]
                print(f"[room {c}] cleaned up (no guest)")

def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("", PORT))
    srv.listen(50)
    print(f"Flappy 1v1 relay running on port {PORT}")
    threading.Thread(target=cleanup_old_rooms, daemon=True).start()
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    main()