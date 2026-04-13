import pygame
import random
import numpy as np
import math

W, H             = 420, 640
GROUND_H         = 70
PIPE_W           = 58
GAP              = 148
BIRD_R           = 15
GRAVITY          = 0.48
FLAP_VEL         = -9.2
MAX_FALL         = 11.0
PIPE_SPEED       = 4.2
BIRD_X           = 80
PIPE_SPAWN_TICKS = 96

FRAME_W    = 84
FRAME_H    = 84
STACK_SIZE = 4

NEON_CYAN   = (0,   255, 255)
NEON_PINK   = (255,  20, 147)
NEON_YELLOW = (255, 230,   0)
NEON_GREEN  = (57,  255,  20)
DARK_BG     = (8,    8,  22)
PIPE_BODY   = (10, 180,  80)
PIPE_SHINE  = (60, 255, 140)
GROUND_COL  = (22,  18,  40)
GROUND_LINE = (80,  60, 120)

pygame.init()

font_big   = pygame.font.SysFont("couriernew", 38, bold=True) if pygame.font.get_init() else None
font_small = pygame.font.SysFont("couriernew", 18) if pygame.font.get_init() else None

class FlappyWorld:
    def __init__(self):
        self.pipes = []
        self._pipe_timer = PIPE_SPAWN_TICKS // 2
        self._tick = 0

    def reset(self):
        self.pipes = []
        self._pipe_timer = PIPE_SPAWN_TICKS // 2
        self._tick = 0

    def step(self):
        self._tick += 1
        self._pipe_timer += 1
        if self._pipe_timer >= PIPE_SPAWN_TICKS:
            self._pipe_timer = 0
            self.pipes.extend(self._create_pipe())
        for p in self.pipes:
            p["x"] -= PIPE_SPEED
        self.pipes = [p for p in self.pipes if p["x"] + PIPE_W > -10]

    def _create_pipe(self):
        h = random.randint(110, H - GROUND_H - GAP - 110)
        return [
            {"x": W + 10, "h": h, "top": True},
            {"x": W + 10, "h": h, "top": False},
        ]

    def next_pipe_for_bird(self, bird_x=BIRD_X):
        for p in self.pipes:
            if p["top"] and p["x"] + PIPE_W > bird_x:
                return p
        return None

    def obs_for(self, bird_y, bird_vel, bird_x=BIRD_X):
        next_pipe = self.next_pipe_for_bird(bird_x)
        if next_pipe is None:
            dx = 1.0
            gap_top = 0.5
            gap_bot = 0.5
        else:
            dx = (next_pipe["x"] - bird_x) / W
            gap_top = next_pipe["h"] / H
            gap_bot = (next_pipe["h"] + GAP) / H
        return np.array([
            bird_y / H,
            bird_vel / MAX_FALL,
            dx,
            gap_top,
            gap_bot,
        ], dtype=np.float32)

    def collides(self, bird_rect):
        for p in self.pipes:
            if p["top"]:
                if bird_rect.colliderect(pygame.Rect(p["x"], 0, PIPE_W, p["h"])):
                    return True
            else:
                if bird_rect.colliderect(pygame.Rect(p["x"], p["h"] + GAP, PIPE_W, H)):
                    return True
        return False

    def draw_pipes(self, surf):
        for p in self.pipes:
            draw_pipe(surf, p["x"], p["h"])

class FlappyEnv:
    def __init__(self):
        self._surf   = pygame.Surface((W, H))
        self._frames = np.zeros((STACK_SIZE, FRAME_H, FRAME_W), dtype=np.float32)
        self.world = FlappyWorld()
        self.reset()

    def reset(self):
        self._bird_y     = H / 2 - 40
        self._bird_vel   = 0.0
        self._score      = 0
        self._alive      = True
        self._tick       = 0

        # use the world's timer and spawn one immediately so pipes are visible
        self.world.reset()

        # frame stack init
        self._frames[:] = 0
        frame = self._get_frame()
        for i in range(STACK_SIZE):
            self._frames[i] = frame

        return self._frames.copy(), {}

    def step(self, action):
        assert self._alive, "call reset() before stepping after a done"
        if action == 1:
            self._bird_vel = FLAP_VEL
        self._bird_vel = min(self._bird_vel + GRAVITY, MAX_FALL)
        self._bird_y  += self._bird_vel
        self._tick    += 1

        self.world.step()

        # scoring
        scored = False
        for p in self.world.pipes:
            if p["top"] and not p.get("scored", False) and p["x"] < BIRD_X:
                p["scored"] = True
                self._score += 1
                scored = True

        bird_rect = pygame.Rect(
            BIRD_X - BIRD_R + 3, int(self._bird_y) - BIRD_R + 3,
            (BIRD_R - 3) * 2,    (BIRD_R - 3) * 2
        )
        terminated = False
        if self.world.collides(bird_rect):
            terminated = True
        if self._bird_y + BIRD_R >= H - GROUND_H:
            terminated = True
        if self._bird_y - BIRD_R <= 0:
            terminated = True

        if terminated:
            reward = -5.0
            self._alive = False
        elif scored:
            reward = 10.0
        else:
            reward = 0.1

        frame = self._get_frame()
        self._frames = np.roll(self._frames, -1, axis=0)
        self._frames[-1] = frame

        return self._frames.copy(), reward, terminated, False, {"score": self._score}

    @property
    def score(self):
        return self._score

    def _get_frame(self):
        self._draw(self._surf)
        small = pygame.transform.scale(self._surf, (FRAME_W, FRAME_H))
        arr   = pygame.surfarray.array3d(small).transpose(1, 0, 2)
        gray  = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        return (gray / 255.0).astype(np.float32)

    def _draw(self, surf):
        draw_bg(surf, 0.0)
        draw_ground(surf, 0.0)
        for p in self.world.pipes:
            draw_pipe(surf, p["x"], p["h"])
        draw_bird(surf, BIRD_X, self._bird_y, self._bird_vel, 0)

# visual helpers (used by trainer/play)
def draw_bird(surf, x, y, vel, flap_anim):
    angle = max(-30, min(45, vel * 4.5))
    glow_surf = pygame.Surface((80, 80), pygame.SRCALPHA)
    for gr in range(30, 0, -5):
        pygame.draw.circle(glow_surf, (255, 230, 0, int(18*(1-gr/30))), (40,40), gr)
    surf.blit(glow_surf, (int(x-40), int(y-40)))
    body_surf = pygame.Surface((BIRD_R*4, BIRD_R*4), pygame.SRCALPHA)
    cx, cy = BIRD_R*2, BIRD_R*2
    pygame.draw.circle(body_surf, NEON_YELLOW, (cx,cy), BIRD_R)
    pygame.draw.circle(body_surf, (255,200,0),  (cx,cy), BIRD_R, 2)
    wing_y = cy + int(math.sin(flap_anim * 0.3) * 5)
    pygame.draw.polygon(body_surf, (220,150,0), [(cx-4,cy+2),(cx-BIRD_R-6,wing_y+4),(cx-4,cy+8)])
    pygame.draw.circle(body_surf, (20,20,20),    (cx+6,cy-4), 5)
    pygame.draw.circle(body_surf, (255,255,255), (cx+7,cy-5), 2)
    pygame.draw.polygon(body_surf, (255,140,0), [(cx+BIRD_R-2,cy-2),(cx+BIRD_R+8,cy+1),(cx+BIRD_R-2,cy+5)])
    rotated = pygame.transform.rotate(body_surf, angle)
    surf.blit(rotated, rotated.get_rect(center=(int(x),int(y))))

def draw_pipe(surf, x, h):
    top_rect = pygame.Rect(x, 0, PIPE_W, h)
    bot_rect = pygame.Rect(x, h + GAP, PIPE_W, H)
    for rect, is_top in [(top_rect, True), (bot_rect, False)]:
        pygame.draw.rect(surf, PIPE_BODY, rect, border_radius=3)
        cap_rect = pygame.Rect(rect.x-5, rect.bottom-22 if is_top else rect.top, PIPE_W+10, 22)
        pygame.draw.rect(surf, PIPE_SHINE, cap_rect, border_radius=4)
        pygame.draw.rect(surf, PIPE_BODY,  cap_rect, 2, border_radius=4)

def draw_ground(surf, scroll):
    pygame.draw.rect(surf, GROUND_COL, (0, H-GROUND_H, W, GROUND_H))
    pygame.draw.line(surf, GROUND_LINE, (0, H-GROUND_H), (W, H-GROUND_H), 3)

def draw_bg(surf, scroll):
    surf.fill(DARK_BG)
    for li, (speed, alpha, height) in enumerate([(0.1,25,200),(0.2,18,300),(0.05,12,120)]):
        ms = pygame.Surface((W+200, H), pygame.SRCALPHA)
        off = (scroll * speed) % (W+200)
        pts = [(-off, H-GROUND_H)]
        rs = random.Random(li * 999)
        for xi in range(-200, W+400, 40):
            pts.append((xi-off, H-GROUND_H-rs.randint(height//3, height)))
        pts.append((W+200-off, H-GROUND_H))
        pygame.draw.polygon(ms, (20+li*8, 10+li*5, 50+li*10, alpha), pts)
        surf.blit(ms, (0,0))
