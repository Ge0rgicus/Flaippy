# flappy_1v1.py  –  Flappy Bird Online 1v1
# ─────────────────────────────────────────────────────────────────────────────
# Controls:
#   SPACE / W / UP      →  flap (human mode)
#   R                   →  rematch (game over screen)
#   ESC                 →  back to lobby
#
# Secret AI mode — activate in the LOBBY before the round starts:
#   Type "KONAMI"       →  AI flies your bird the whole round
#   Ctrl+Shift+A        →  same
#
# A tiny "[ AI ARMED ]" shows in the corner of YOUR screen only.
# Your opponent sees nothing. Toggle again to disarm before game starts.
# ─────────────────────────────────────────────────────────────────────────────

import sys, os, socket, threading, json, time, math, random
import pygame
import numpy as np
from Flappy_evo import decide, unflatten

# ── resolve paths for frozen .exe ────────────────────────────────────────────
def resource_path(rel):
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)

# ════════════════════════════════════════════════════════════════════════════
#  LOAD AI NETWORK
# ════════════════════════════════════════════════════════════════════════════
def load_best_net():
    try:
        data  = np.load(resource_path("evo_checkpoint.npz"))
        net   = unflatten(data["best_genome"])
        score = int(data["best_score"][0])
        gen   = int(data["generation"][0])
        print(f"[AI] Loaded — gen={gen}  best_score={score}")
        return net
    except Exception as e:
        print(f"[AI] Could not load checkpoint: {e}")
        return None

AI_NET = load_best_net()

def ai_decide(net, bird_y, bird_vel, pipes):
    """Return 1 = flap, 0 = do nothing. Same observation as training."""
    next_pipe = None
    for p in pipes:
        if p["x"] + PIPE_W > BIRD_X:
            next_pipe = p
            break
    if next_pipe is None:
        dx, gap_top, gap_bot = 1.0, 0.5, 0.5
    else:
        dx      = (next_pipe["x"] - BIRD_X) / GW
        gap_top = next_pipe["h"] / GH
        gap_bot = (next_pipe["h"] + GAP) / GH
    obs = np.array([bird_y / GH, bird_vel / MAX_FALL, dx, gap_top, gap_bot],
                   dtype=np.float32)
    return decide(net, obs)

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
GW, GH   = 340, 560
WIN_W    = GW * 2 + 60
WIN_H    = GH + 80

GROUND_H      = 60
PIPE_W        = 50
GAP           = 130
BIRD_R        = 13
BIRD_X        = 70
GRAVITY       = 0.44
FLAP_VEL      = -9.0
MAX_FALL      = 11.0
PIPE_SPEED    = 3.8
PIPE_INTERVAL = 90        # ticks between pipe spawns
FLAP_COOLDOWN = 10        # min frames between AI flaps

PORT    = 55201
NET_HZ  = 30              # state syncs per second

C_BG    = (  8,   8,  22)
C_GND   = ( 22,  18,  40)
C_GNDL  = ( 80,  60, 120)
C_PIPE  = ( 10, 180,  80)
C_PIPH  = ( 60, 255, 140)
C_YELLOW= (255, 230,   0)
C_PINK  = (255,  45, 120)
C_CYAN  = (  0, 255, 247)
C_GREEN = ( 57, 255,  20)
C_WHITE = (255, 255, 255)
C_DARK  = ( 30,  30,  60)

# ════════════════════════════════════════════════════════════════════════════
#  NETWORKING  (line-delimited JSON over TCP)
# ════════════════════════════════════════════════════════════════════════════
class NetPeer:
    def __init__(self):
        self.sock      = None
        self.connected = False
        self._buf      = b""
        self._lock     = threading.Lock()
        self._inbox    = []

    def host(self, status_cb):
        def _run():
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("", PORT))
            srv.listen(1)
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]; s.close()
            except Exception:
                ip = "your IP"
            status_cb(f"Your IP:  {ip}\nWaiting for opponent...")
            srv.settimeout(300)
            try:
                conn, addr = srv.accept()
                self.sock = conn
                self.connected = True
                status_cb(f"Opponent connected from {addr[0]}!")
                self._recv_loop()
            except socket.timeout:
                status_cb("Timed out. Restart to try again.")
            finally:
                try: srv.close()
                except: pass
        threading.Thread(target=_run, daemon=True).start()

    def join(self, ip, status_cb):
        def _run():
            for attempt in range(15):
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(5)
                    s.connect((ip.strip(), PORT))
                    self.sock = s
                    self.connected = True
                    status_cb("Connected!")
                    self._recv_loop()
                    return
                except Exception as e:
                    status_cb(f"Attempt {attempt+1}/15... ({e})")
                    time.sleep(1)
            status_cb("Could not connect. Check the IP and try again.")
        threading.Thread(target=_run, daemon=True).start()

    def _recv_loop(self):
        self.sock.settimeout(None)
        try:
            while True:
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                self._buf += chunk
                while b"\n" in self._buf:
                    line, self._buf = self._buf.split(b"\n", 1)
                    try:
                        with self._lock:
                            self._inbox.append(json.loads(line.decode()))
                    except Exception:
                        pass
        except Exception:
            pass
        self.connected = False

    def send(self, msg):
        if not self.connected:
            return
        try:
            self.sock.sendall((json.dumps(msg) + "\n").encode())
        except Exception:
            self.connected = False

    def poll(self):
        with self._lock:
            out, self._inbox = self._inbox, []
        return out

    def close(self):
        try: self.sock.close()
        except: pass

# ════════════════════════════════════════════════════════════════════════════
#  DETERMINISTIC PIPE GENERATOR
#  Stepped by a shared TICK COUNT → identical on both clients, never drifts.
# ════════════════════════════════════════════════════════════════════════════
class PipeGen:
    def __init__(self, seed):
        self._rng        = random.Random(seed)
        self.pipes       = []
        self._last_spawn = -(PIPE_INTERVAL // 2)   # first pipe spawns early

    def sync(self, tick):
        """Advance by one tick. Call exactly once per frame."""
        if tick - self._last_spawn >= PIPE_INTERVAL:
            self._last_spawn = tick
            min_h = 70
            max_h = GH - GROUND_H - GAP - 70
            h = self._rng.randint(min_h, max_h)
            self.pipes.append({"x": GW + PIPE_W + 10, "h": h})
        for p in self.pipes:
            p["x"] -= PIPE_SPEED
        self.pipes = [p for p in self.pipes if p["x"] + PIPE_W > -10]

    def next_ahead(self):
        for p in self.pipes:
            if p["x"] + PIPE_W > BIRD_X:
                return p
        return None

# ════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ════════════════════════════════════════════════════════════════════════════
def draw_bg(surf, scroll):
    surf.fill(C_BG)
    rng = random.Random(42)
    for li, (sp, alpha, height) in enumerate([(0.08,25,180),(0.18,18,280),(0.05,12,110)]):
        ms  = pygame.Surface((GW + 300, GH), pygame.SRCALPHA)
        off = int(scroll * sp) % (GW + 200)
        pts = [(-off, GH - GROUND_H)]
        for xi in range(-200, GW + 400, 36):
            pts.append((xi - off, GH - GROUND_H - rng.randint(height//3, height)))
        pts.append((GW + 200 - off, GH - GROUND_H))
        pygame.draw.polygon(ms, (20+li*8, 10+li*5, 50+li*10, alpha), pts)
        surf.blit(ms, (0, 0))

def draw_ground(surf):
    pygame.draw.rect(surf, C_GND,  (0, GH - GROUND_H, GW, GROUND_H))
    pygame.draw.line(surf, C_GNDL, (0, GH - GROUND_H), (GW, GH - GROUND_H), 3)

def draw_pipes(surf, pipes):
    for p in pipes:
        x, ph = p["x"], p["h"]
        pygame.draw.rect(surf, C_PIPE, (x, 0, PIPE_W, ph), border_radius=2)
        cap = pygame.Rect(x-5, ph-20, PIPE_W+10, 20)
        pygame.draw.rect(surf, C_PIPH, cap, border_radius=4)
        pygame.draw.rect(surf, C_PIPE, cap, 2, border_radius=4)
        bot_y = ph + GAP
        pygame.draw.rect(surf, C_PIPE, (x, bot_y, PIPE_W, GH), border_radius=2)
        cap2 = pygame.Rect(x-5, bot_y, PIPE_W+10, 20)
        pygame.draw.rect(surf, C_PIPH, cap2, border_radius=4)
        pygame.draw.rect(surf, C_PIPE, cap2, 2, border_radius=4)

def draw_bird(surf, x, y, vel, flap_tick, color):
    angle = max(-30, min(45, vel * 4.5))
    glow = pygame.Surface((70, 70), pygame.SRCALPHA)
    for gr in range(28, 0, -4):
        pygame.draw.circle(glow, (*color, int(18*(1-gr/28))), (35,35), gr)
    surf.blit(glow, (int(x)-35, int(y)-35))
    bs = pygame.Surface((BIRD_R*4, BIRD_R*4), pygame.SRCALPHA)
    cx = cy = BIRD_R * 2
    pygame.draw.circle(bs, color, (cx, cy), BIRD_R)
    pygame.draw.circle(bs, (255,255,255,80), (cx, cy), BIRD_R, 2)
    wy = cy + int(math.sin(flap_tick * 0.35) * 5)
    wc = (max(0,color[0]-50), max(0,color[1]-80), max(0,color[2]-80))
    pygame.draw.polygon(bs, wc, [(cx-3,cy+2),(cx-BIRD_R-6,wy+4),(cx-3,cy+8)])
    pygame.draw.circle(bs, (20,20,20), (cx+6,cy-4), 4)
    pygame.draw.circle(bs, (255,255,255), (cx+7,cy-5), 1)
    pygame.draw.polygon(bs, (255,140,0),
                        [(cx+BIRD_R-1,cy-2),(cx+BIRD_R+8,cy+1),(cx+BIRD_R-1,cy+4)])
    rot = pygame.transform.rotate(bs, -angle)
    surf.blit(rot, rot.get_rect(center=(int(x), int(y))))

# ════════════════════════════════════════════════════════════════════════════
#  LOBBY
# ════════════════════════════════════════════════════════════════════════════
def lobby_screen(screen, clock, fonts, ai_armed):
    """
    Returns (peer, seed, role, ai_armed) when both players are connected.
    ai_armed: persists between calls so rematch carries the same AI state.
    Secret toggle: type KONAMI or Ctrl+Shift+A → arms/disarms AI for next round.
    """
    big, med, small = fonts
    W, H = screen.get_size()

    status   = ""
    peer     = None
    mode     = None
    ip_input = ""
    ip_focus = False
    konami   = ""

    host_rect = join_rect = ip_box_rect = connect_rect = pygame.Rect(0,0,0,0)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); local_ip = s.getsockname()[0]; s.close()
    except Exception:
        local_ip = "unknown"

    def set_status(s): nonlocal status; status = s

    while True:
        clock.tick(60)
        screen.fill(C_BG)

        # title
        t = big.render("FLAPPY  1v1", True, C_YELLOW)
        screen.blit(t, t.get_rect(center=(W//2, 80)))
        sub = small.render("NEUROEVOLUTION EDITION", True, C_CYAN)
        screen.blit(sub, sub.get_rect(center=(W//2, 118)))

        # AI armed indicator — tiny, bottom-left corner, your eyes only
        if ai_armed:
            ab = small.render("[ AI ARMED ]", True, C_GREEN)
            screen.blit(ab, (12, H - 26))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                # secret toggle
                if event.unicode:
                    konami = (konami + event.unicode.upper())[-6:]
                    if konami == "KONAMI":
                        ai_armed = not ai_armed
                mods = pygame.key.get_mods()
                if event.key == pygame.K_a and (mods & pygame.KMOD_CTRL) and (mods & pygame.KMOD_SHIFT):
                    ai_armed = not ai_armed

                if event.key == pygame.K_ESCAPE:
                    mode = None; peer = None; ip_input = ""; status = ""; ip_focus = False

                if ip_focus:
                    if event.key == pygame.K_BACKSPACE:
                        ip_input = ip_input[:-1]
                    elif event.key == pygame.K_RETURN and ip_input:
                        peer.join(ip_input, set_status); ip_focus = False
                    elif event.unicode and event.unicode.isprintable() and len(ip_input) < 21:
                        ip_input += event.unicode

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if mode is None:
                    if host_rect.collidepoint(mx, my):
                        mode = "host"; peer = NetPeer(); peer.host(set_status)
                    if join_rect.collidepoint(mx, my):
                        mode = "join"; peer = NetPeer(); ip_focus = True
                elif mode == "join":
                    ip_focus = ip_box_rect.collidepoint(mx, my)
                    if connect_rect.collidepoint(mx, my) and ip_input:
                        peer.join(ip_input, set_status); ip_focus = False

        # panels
        if mode is None:
            host_rect = _btn(screen, med, "HOST GAME", W//2 - 195, 210, C_YELLOW)
            join_rect = _btn(screen, med, "JOIN GAME", W//2 + 15,  210, C_PINK)
            _hint(screen, small, "HOST: share your IP with your friend", W//2, 295)
            _hint(screen, small, "JOIN: enter the host's IP address",    W//2, 318)

        elif mode == "host":
            _hint(screen, small, "Share this IP with your friend:", W//2, 205)
            ip_surf = med.render(local_ip, True, C_GREEN)
            screen.blit(ip_surf, ip_surf.get_rect(center=(W//2, 240)))

        elif mode == "join":
            _hint(screen, small, "Enter host IP address:", W//2, 205)
            ip_box_rect = pygame.Rect(W//2 - 155, 228, 230, 38)
            pygame.draw.rect(screen, C_DARK, ip_box_rect, border_radius=4)
            pygame.draw.rect(screen, C_CYAN if ip_focus else (60,60,90),
                             ip_box_rect, 2, border_radius=4)
            it = med.render(ip_input + ("|" if ip_focus else ""), True, C_WHITE)
            screen.blit(it, it.get_rect(midleft=(ip_box_rect.x+8, ip_box_rect.centery)))
            connect_rect = _btn(screen, med, "CONNECT", W//2 + 85, 231, C_CYAN)

        for i, line in enumerate(status.split("\n")):
            s = small.render(line, True, (180,180,220))
            screen.blit(s, s.get_rect(center=(W//2, 300 + i*24)))

        # check for game start
        if peer and peer.connected:
            for msg in peer.poll():
                if msg.get("type") == "start":
                    return peer, msg["seed"], "guest", ai_armed
            if mode == "host":
                seed = random.randint(0, 10**9)
                peer.send({"type": "start", "seed": seed})
                return peer, seed, "host", ai_armed

        pygame.display.flip()

# ════════════════════════════════════════════════════════════════════════════
#  GAME LOOP
# ════════════════════════════════════════════════════════════════════════════
def game_loop(screen, clock, fonts, peer, seed, role, ai_mode):
    """
    ai_mode: if True the AI controls your bird this entire round.
    Pipes are driven by a shared tick counter — both clients stay in sync.
    """
    big, med, small = fonts
    W, H = screen.get_size()
    score_font = pygame.font.SysFont("couriernew", 32, bold=True)
    ai_font    = pygame.font.SysFont("couriernew", 13, bold=True)

    # Same seed + same tick → identical pipe sequence on both machines
    my_pipes  = PipeGen(seed)
    opp_pipes = PipeGen(seed)
    tick      = 0

    # local bird
    my_y     = GH / 2 - 40
    my_vel   = 0.0
    my_sc    = 0
    my_alive = True
    my_flap  = 0
    my_scored_xs = set()   # pipe x-buckets already scored

    # opponent bird (visual mirror, driven by network events)
    opp_y    = GH / 2 - 40
    opp_vel  = 0.0
    opp_sc   = 0
    opp_alive= True
    opp_flap = 0

    scroll      = 0.0
    ai_cooldown = 0
    game_over   = False
    result_text = ""
    last_net_t  = time.time()

    left_surf  = pygame.Surface((GW, GH))
    right_surf = pygame.Surface((GW, GH))

    def do_flap():
        nonlocal my_vel, my_flap
        my_vel  = FLAP_VEL
        my_flap = 0
        peer.send({"type": "flap"})

    def collides(y, pipes):
        by = round(y); br = BIRD_R - 2
        if by + br >= GH - GROUND_H: return True
        if by - br <= 0:             return True
        for p in pipes:
            if BIRD_X + br > p["x"] and BIRD_X - br < p["x"] + PIPE_W:
                if by - br < p["h"] or by + br > p["h"] + GAP:
                    return True
        return False

    while True:
        clock.tick(60)

        # ── events ─────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                peer.close(); pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_SPACE, pygame.K_w, pygame.K_UP):
                    if my_alive and not ai_mode:
                        do_flap()

        # ── network receive ─────────────────────────────────────────────────
        for msg in peer.poll():
            t = msg.get("type")
            if t == "flap":
                opp_vel  = FLAP_VEL
                opp_flap = 0
            elif t == "score":
                opp_sc = msg.get("v", opp_sc)
            elif t == "dead":
                opp_alive = False
                opp_sc    = msg.get("score", opp_sc)
            elif t in ("rematch", "start"):
                return "rematch"

        if not peer.connected and not game_over:
            result_text = "OPPONENT DISCONNECTED"
            game_over   = True

        # ── advance tick + pipes (both sides, every frame, unconditionally) ─
        tick   += 1
        scroll += PIPE_SPEED
        my_pipes.sync(tick)
        opp_pipes.sync(tick)

        # ── AI decision (with cooldown to prevent flap spam) ────────────────
        if ai_mode and my_alive and AI_NET is not None:
            ai_cooldown = max(0, ai_cooldown - 1)
            if ai_cooldown == 0:
                if ai_decide(AI_NET, my_y, my_vel, my_pipes.pipes) == 1:
                    do_flap()
                    ai_cooldown = FLAP_COOLDOWN

        # ── local bird physics ──────────────────────────────────────────────
        if my_alive:
            my_vel   = min(my_vel + GRAVITY, MAX_FALL)
            my_y    += my_vel
            my_flap += 1

            # score — bucket by spawn position to avoid double-counting
            for p in my_pipes.pipes:
                bucket = round(p["x"] / 5)
                if bucket not in my_scored_xs and p["x"] + PIPE_W < BIRD_X:
                    my_scored_xs.add(bucket)
                    my_sc += 1
                    peer.send({"type": "score", "v": my_sc})

            if collides(my_y, my_pipes.pipes):
                my_alive = False
                peer.send({"type": "dead", "score": my_sc})

        # ── opponent mirror physics (local gravity sim) ─────────────────────
        if opp_alive:
            opp_vel   = min(opp_vel + GRAVITY, MAX_FALL)
            opp_y    += opp_vel
            opp_flap += 1

        # ── periodic state broadcast ─────────────────────────────────────────
        now = time.time()
        if now - last_net_t >= 1 / NET_HZ and my_alive:
            peer.send({"type": "state", "y": round(my_y, 1), "vel": round(my_vel, 2)})
            last_net_t = now

        # ── game over check ──────────────────────────────────────────────────
        if not my_alive and not opp_alive and not game_over:
            game_over = True
            if   my_sc > opp_sc: result_text = f"YOU WIN!   {my_sc} - {opp_sc}"
            elif opp_sc > my_sc: result_text = f"YOU LOSE   {my_sc} - {opp_sc}"
            else:                result_text = f"DRAW!      {my_sc} - {opp_sc}"

        # ══════════════════════════════════════════════════════════════════
        #  DRAW
        # ══════════════════════════════════════════════════════════════════
        screen.fill((5, 5, 15))

        # left pane — my bird
        draw_bg(left_surf, scroll)
        draw_ground(left_surf)
        draw_pipes(left_surf, my_pipes.pipes)
        draw_bird(left_surf, BIRD_X, my_y, my_vel, my_flap,
                  C_YELLOW if my_alive else (70,70,70))
        sc_s = score_font.render(str(my_sc), True, C_YELLOW)
        left_surf.blit(sc_s, sc_s.get_rect(centerx=GW//2, y=10))
        if ai_mode:
            ab = ai_font.render("[ AI ]", True, C_GREEN)
            left_surf.blit(ab, ab.get_rect(centerx=GW//2, y=GH - GROUND_H - 22))
        screen.blit(left_surf, (20, 60))

        # right pane — opponent
        draw_bg(right_surf, scroll)
        draw_ground(right_surf)
        draw_pipes(right_surf, opp_pipes.pipes)
        draw_bird(right_surf, BIRD_X, opp_y, opp_vel, opp_flap,
                  C_PINK if opp_alive else (70,70,70))
        sc_s2 = score_font.render(str(opp_sc), True, C_PINK)
        right_surf.blit(sc_s2, sc_s2.get_rect(centerx=GW//2, y=10))
        screen.blit(right_surf, (GW + 40, 60))

        # header
        you_lbl = "YOU  [AI]" if ai_mode else "YOU"
        yl = med.render(you_lbl, True, C_YELLOW)
        screen.blit(yl, yl.get_rect(center=(20 + GW//2, 35)))
        ol = med.render("OPPONENT", True, C_PINK)
        screen.blit(ol, ol.get_rect(center=(GW + 40 + GW//2, 35)))

        # VS divider
        pygame.draw.line(screen, (25, 25, 60), (W//2, 0), (W//2, H), 3)
        vs = big.render("VS", True, C_CYAN)
        screen.blit(vs, vs.get_rect(center=(W//2, H//2)))

        # game over overlay
        if game_over and result_text:
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 155))
            screen.blit(ov, (0, 0))
            rt = big.render(result_text, True, C_YELLOW)
            screen.blit(rt, rt.get_rect(center=(W//2, H//2 - 50)))
            rb = med.render("R  →  Rematch        ESC  →  Lobby", True, C_CYAN)
            screen.blit(rb, rb.get_rect(center=(W//2, H//2 + 10)))
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                peer.send({"type": "rematch"})
                return "rematch"
            if keys[pygame.K_ESCAPE]:
                peer.close()
                return "lobby"

        pygame.display.flip()

# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _btn(screen, font, text, x, y, color):
    t = font.render(text, True, color)
    r = pygame.Rect(x, y, t.get_width() + 20, t.get_height() + 10)
    pygame.draw.rect(screen, color, r, 2, border_radius=4)
    if r.collidepoint(pygame.mouse.get_pos()):
        s = pygame.Surface(r.size, pygame.SRCALPHA)
        s.fill((*color, 40)); screen.blit(s, r.topleft)
    screen.blit(t, (r.x + 10, r.y + 5))
    return r

def _hint(screen, font, text, cx, y):
    s = font.render(text, True, (100, 100, 140))
    screen.blit(s, s.get_rect(center=(cx, y)))

# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Flappy 1v1  –  Neuroevolution Edition")
    clock = pygame.time.Clock()

    big   = pygame.font.SysFont("couriernew", 36, bold=True)
    med   = pygame.font.SysFont("couriernew", 20, bold=True)
    small = pygame.font.SysFont("couriernew", 15)
    fonts = (big, med, small)

    ai_armed = False

    while True:
        result = lobby_screen(screen, clock, fonts, ai_armed)
        if result is None:
            break
        peer, seed, role, ai_armed = result

        action = game_loop(screen, clock, fonts, peer, seed, role, ai_armed)

        while action == "rematch":
            # go back to lobby so AI can be re-armed or disarmed for next round
            result = lobby_screen(screen, clock, fonts, ai_armed)
            if result is None:
                peer.close(); break
            peer, seed, role, ai_armed = result
            action = game_loop(screen, clock, fonts, peer, seed, role, ai_armed)

        if action == "lobby":
            peer.close()

    pygame.quit()

if __name__ == "__main__":
    main()