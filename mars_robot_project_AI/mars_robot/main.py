"""Mars Robot Explorer - Giao diện chính pygame.

Tính năng:
- Cửa sổ resize tự do
- Animate theo visited thực tế của thuật toán
- Thanh tốc độ (slider)
- Địa hình ROUGH chi phí ×5
"""

import pygame
import sys
import time
import threading

from utils import generate_map, EMPTY, WALL, SAMPLE, BASE, ROUGH, GRID_SIZE, NUM_SAMPLES
from algorithms import (PATHFINDING_ALGOS, ORDER_ALGOS,
                        NONDETERMINISTIC_ALGOS, ADVERSARIAL_ALGOS, ALL_ALGOS)

# ─────────────────────────────────────────────
# Màu sắc
# ─────────────────────────────────────────────
C = {
    "bg":            (235, 238, 242),
    "panel_bg":      (255, 255, 255),
    "panel_border":  (210, 215, 220),
    "shadow":        (200, 205, 212),
    "text":          (25,  30,  40),
    "text_sub":      (110, 118, 130),
    "text_white":    (255, 255, 255),
    "accent":        (41,  128, 185),
    "accent_dark":   (26,  100, 150),
    "success":       (39,  174,  96),
    "danger":        (192,  57,  43),
    "warning":       (230, 126,  34),
    "btn_normal":    (248, 249, 251),
    "btn_hover":     (224, 235, 247),
    "btn_active":    (41,  128, 185),
    "btn_border":    (200, 208, 218),
    "divider":       (225, 229, 234),
    "scrollbar":     (200, 208, 218),
    "scrollbar_h":   (150, 170, 195),
    "slider_track":  (210, 218, 228),
    "slider_fill":   (41,  128, 185),
    "slider_thumb":  (41,  128, 185),
    # Bản đồ
    "cell_empty":    (248, 249, 251),
    "cell_wall":     (52,  58,  64),
    "cell_rough":    (210, 180, 140),
    "cell_rough_d":  (170, 130,  80),
    "cell_sample":   (230, 126,  34),
    "cell_sample_c": (189, 195, 199),
    "cell_base":     (41,  128, 185),
    "cell_robot":    (192,  57,  43),
    "cell_path":     (130, 210, 150),
    "cell_visited":  (253, 235, 150),
    "cell_border":   (215, 220, 226),
    # Non-deterministic / Adversarial
    "cell_plan_b":   (240, 180,  80),   # AND-OR: kế hoạch phục hồi (cam)
    "cell_belief":   (190, 155, 230),   # Sensorless: belief state (tím nhạt)
    "cell_robot_b":  (142,  68, 173),   # Adversarial: Robot B (tím đậm)
    "cell_rb_path":  (200, 170, 230),   # Adversarial: đường đi Robot B (tím nhạt)
    # Overlay
    "overlay_bg":    (15,  20,  30),
    "overlay_panel": (30,  38,  50),
    "overlay_border":(60,  80, 110),
    "help_accent":   (100, 180, 255),
}

FPS     = 60
MIN_W   = 900
MIN_H   = 600

# Delay animation: giá trị ms, 0 = nhanh nhất
DELAY_MIN = 0
DELAY_MAX = 300

COMBO_PATH  = "A*"
COMBO_ORDER = "Sim. Annealing"

# Nhóm thuật toán không xác định / đối kháng
NONDETERMINISTIC_NAMES = {"AND-OR", "Sensorless"}
ADVERSARIAL_NAMES      = {"Minimax", "Alpha-Beta"}


# ─────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────
class Layout:
    def __init__(self, sw, sh):
        self.SW = sw
        self.SH = sh
        self.PANEL_W  = max(195, int(sw * 0.18))
        self.MAP_AREA = sw - self.PANEL_W * 2
        self.CELL     = max(20, self.MAP_AREA // GRID_SIZE)
        self.MAP_X    = self.PANEL_W
        self.MAP_Y    = max(0, (sh - self.CELL * GRID_SIZE) // 2)

    def cell_rect(self, r, c):
        x = self.MAP_X + c * self.CELL
        y = self.MAP_Y + r * self.CELL
        return pygame.Rect(x + 1, y + 1, self.CELL - 2, self.CELL - 2)

    def map_rect(self):
        return pygame.Rect(self.MAP_X, self.MAP_Y,
                           self.CELL * GRID_SIZE, self.CELL * GRID_SIZE)


# ─────────────────────────────────────────────
# Trạng thái
# ─────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.grid, self.robot_start, self.samples, self.base = generate_map()

        self.selected_algo = "A*"
        self.combo_mode    = False

        self.result        = None
        self.robot_pos     = self.robot_start
        self.collected     = set()

        # Animation
        self.anim_phase    = 0
        self.anim_index    = 0
        self.animating     = False
        self.delay_ms      = 60
        self.last_step_t   = 0
        self.run_time_ms   = 0

        # Hiển thị cơ bản
        self.shown_visited = []
        self.shown_path    = []

        # Non-deterministic: AND-OR
        self.slip_prob     = 0.3          # Xác suất trượt (0.0 – 0.5)
        self.shown_plan_b  = []           # Ô kế hoạch phục hồi (vẽ cam)

        # Non-deterministic: Sensorless
        self.shown_belief  = set()        # Belief state hiện tại (vẽ tím)

        # Adversarial
        self.robot_b_pos       = None
        self.shown_robot_b_path = []

        self.log_lines     = []
        self.show_path_log = False
        self.show_help     = False
        self.help_scroll   = 0
        self.left_scroll   = 0
        self.calculating   = False

    def new_map(self, seed=None):
        self.grid, self.robot_start, self.samples, self.base = generate_map(seed)
        self.reset_run()

    def reset_run(self):
        self.result             = None
        self.robot_pos          = self.robot_start
        self.collected          = set()
        self.anim_phase         = 0
        self.anim_index         = 0
        self.animating          = False
        self.last_step_t        = 0
        self.run_time_ms        = 0
        self.shown_visited      = []
        self.shown_path         = []
        self.shown_plan_b       = []
        self.shown_belief       = set()
        self.robot_b_pos        = None
        self.shown_robot_b_path = []
        self.log_lines          = []
        self.show_path_log      = False
        self.calculating        = False   # dang tinh toan tren background thread

    def run_algorithm(self):
        if self.calculating:
            return  # dang tinh, bo qua click thu 2
        self.reset_run()
        self.calculating = True

        # Chay thuat toan tren background thread de pygame khong bi treo
        def _worker():
            t0 = time.time()
            try:
                if self.combo_mode:
                    self.log(f"▶ Kết hợp: {COMBO_PATH} + {COMBO_ORDER}")
                    order_fn = ORDER_ALGOS[COMBO_ORDER]
                    path_fn  = PATHFINDING_ALGOS[COMBO_PATH]
                    order_res = order_fn(self.grid, self.robot_start, self.samples, self.base)
                    if not order_res["path"]:
                        self.log("✗ Không tìm được đường!")
                        self.calculating = False
                        return
                    opt_order   = self._extract_order(order_res["path"])
                    self.result = path_fn(self.grid, self.robot_start, opt_order, self.base)
                else:
                    self.log(f"▶ Chạy: {self.selected_algo}")
                    fn = ALL_ALGOS.get(self.selected_algo)
                    if self.selected_algo == "AND-OR":
                        self.result = fn(self.grid, self.robot_start, self.samples, self.base,
                                         slip_prob=self.slip_prob)
                    else:
                        self.result = fn(self.grid, self.robot_start, self.samples, self.base)
            except Exception as e:
                self.log(f"✗ Lỗi: {e}")
                self.calculating = False
                return

            self.run_time_ms = (time.time() - t0) * 1000

            if self.result and self.result["path"]:
                s = self.result["stats"]
                self.log(f"✓ Bước: {s['steps']}  Chi phí: {s.get('cost','?')}")
                self.log(f"  Ô đã xét: {s['visited']}  ({self.run_time_ms:.1f} ms)")

                if self.selected_algo in ADVERSARIAL_NAMES:
                    self.robot_b_pos = self.result.get("robot_b_start")
                    res_str = self.result["stats"].get("result", "")
                    sa = self.result["stats"].get("score_a", 0)
                    sb = self.result["stats"].get("score_b", 0)
                    self.log(f"  Robot A: {sa} mẫu | Robot B: {sb} mẫu → {res_str}")

                if self.selected_algo == "AND-OR":
                    nb = self.result["stats"].get("plan_b_count", 0)
                    self.log(f"  Kế hoạch B: {nb} đoạn phục hồi (cam)")

                if self.selected_algo == "Sensorless":
                    bi = self.result["stats"].get("belief_init", 0)
                    bf = self.result["stats"].get("belief_final", 0)
                    self.log(f"  Belief: {bi} → {bf} ô")
                    self.shown_belief = set(self.result["visited"])

                self.anim_phase = 0
                self.anim_index = 0
                self.animating  = True
            else:
                self.log("✗ Không tìm được đường!")

            self.calculating = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _extract_order(self, path):
        order, seen = [], set(map(tuple, self.samples))
        for pos in path:
            t = tuple(pos)
            if t in seen and t not in order:
                order.append(t)
        for s in self.samples:
            if tuple(s) not in order:
                order.append(tuple(s))
        return order

    def update_animation(self):
        """Animate nhiều pha tuỳ thuật toán."""
        if not self.animating or not self.result:
            return

        algo = self.selected_algo
        is_sensorless  = (algo == "Sensorless")
        is_adversarial = (algo in ADVERSARIAL_NAMES)
        is_and_or      = (algo == "AND-OR")

        visited_list  = list(map(tuple, self.result.get("visited", [])))
        path_list     = list(map(tuple, self.result["path"]))
        rb_path_list  = list(map(tuple, self.result.get("robot_b_path", [])))
        belief_hist   = self.result.get("belief_history", [])
        plan_b_flat   = [tuple(p) for seg in self.result.get("plan_b_segments", [])
                         for p in seg]
        sample_set    = [tuple(s) for s in self.samples]

        # ── delay = 0: hiện hết ngay ──
        if self.delay_ms == 0:
            if is_sensorless:
                self.shown_belief  = set(belief_hist[-1]) if belief_hist else set()
                self.shown_visited = []
            elif is_and_or:
                self.shown_visited = visited_list
                self.shown_plan_b  = plan_b_flat
            elif is_adversarial:
                self.shown_robot_b_path = rb_path_list
                self.robot_b_pos = rb_path_list[-1] if rb_path_list else self.robot_b_pos
            else:
                self.shown_visited = visited_list

            self.shown_path = path_list
            self.robot_pos  = tuple(path_list[-1]) if path_list else self.robot_pos
            self.collected  = set(sample_set)
            self.animating  = False
            self.log("■ Hoàn thành!")
            return

        now = pygame.time.get_ticks()
        if now - self.last_step_t < self.delay_ms:
            return
        self.last_step_t = now

        # ── Pha 0 ──
        if self.anim_phase == 0:
            if is_sensorless:
                # Hiện belief state thu hẹp dần
                if self.anim_index < len(belief_hist):
                    self.shown_belief = set(belief_hist[self.anim_index])
                    self.anim_index += 1
                else:
                    self.shown_belief = set()
                    self.anim_phase = 1
                    self.anim_index = 0
            elif is_adversarial:
                # Bỏ qua pha visited, sang ngay pha 1
                self.anim_phase = 1
                self.anim_index = 0
            else:
                # Hiện từng ô visited (BFS/DFS/A*/AND-OR…)
                if self.anim_index < len(visited_list):
                    self.shown_visited.append(visited_list[self.anim_index])
                    self.anim_index += 1
                else:
                    self.anim_phase = 1
                    self.anim_index = 0
                    # AND-OR: hiện plan_b ngay khi xong visited
                    if is_and_or:
                        self.shown_plan_b = plan_b_flat

        # ── Pha 1 ──
        elif self.anim_phase == 1:
            if is_adversarial:
                # Cả 2 robot di chuyển đồng thời
                done_a = self.anim_index >= len(path_list)
                done_b = self.anim_index >= len(rb_path_list)
                if not done_a or not done_b:
                    if not done_a:
                        pos_a = path_list[self.anim_index]
                        self.shown_path.append(pos_a)
                        self.robot_pos = pos_a
                        if pos_a in sample_set:
                            self.collected.add(pos_a)
                    if not done_b:
                        pos_b = rb_path_list[self.anim_index]
                        self.shown_robot_b_path.append(pos_b)
                        self.robot_b_pos = pos_b
                    self.anim_index += 1
                else:
                    self.animating = False
                    self.log("■ Hoàn thành!")
            else:
                # Robot đi theo path (cho tất cả các thuật toán còn lại)
                if self.anim_index < len(path_list):
                    pos = path_list[self.anim_index]
                    self.shown_path.append(pos)
                    self.robot_pos = pos
                    if pos in sample_set:
                        self.collected.add(pos)
                    self.anim_index += 1
                else:
                    self.animating = False
                    self.log("■ Hoàn thành!")

    def log(self, msg):
        self.log_lines.append(msg)
        if len(self.log_lines) > 60:
            self.log_lines.pop(0)


# ─────────────────────────────────────────────
# Button
# ─────────────────────────────────────────────
class Button:
    def __init__(self, rect, label, bg=None, tc=None, radius=7):
        self.rect    = pygame.Rect(rect)
        self.label   = label
        self.bg      = bg
        self.tc      = tc
        self.radius  = radius
        self.hovered = False
        self.active  = False

    def draw(self, surf, font, offset_y=0):
        r = self.rect.move(0, offset_y)
        if self.bg and self.active:
            bg, tc = self.bg, self.tc or C["text_white"]
        elif self.active:
            bg, tc = C["btn_active"], C["text_white"]
        elif self.hovered:
            bg, tc = C["btn_hover"], C["text"]
        else:
            bg, tc = self.bg or C["btn_normal"], self.tc or C["text"]
        pygame.draw.rect(surf, bg, r, border_radius=self.radius)
        pygame.draw.rect(surf, C["btn_border"], r, 1, border_radius=self.radius)
        t = font.render(self.label, True, tc)
        surf.blit(t, t.get_rect(center=r.center))

    def hit(self, pos, offset_y=0):
        return self.rect.move(0, offset_y).collidepoint(pos)

    def handle(self, event, offset_y=0):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.hit(event.pos, offset_y)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hit(event.pos, offset_y):
                return True
        return False


# ─────────────────────────────────────────────
# Slider tốc độ
# ─────────────────────────────────────────────
class Slider:
    """Slider ngang: kéo để điều chỉnh một giá trị (delay, slip prob, …).

    val_max: giá trị tối đa (trái = val_max, phải = 0 với delay; trái=0, phải=val_max với slip).
    mode="delay": trái=chậm(val_max), phải=nhanh(0)
    mode="pct":   trái=0%, phải=val_max%
    """
    def __init__(self, x, y, w, h=16, val_max=DELAY_MAX, mode="delay"):
        self.rect    = pygame.Rect(x, y, w, h)
        self.dragging = False
        self.thumb_r  = h // 2 + 2
        self.val_max  = val_max
        self.mode     = mode

    def get_thumb_x(self, val):
        if self.mode == "delay":
            ratio = 1.0 - val / max(1, self.val_max)
        else:
            ratio = val / max(1, self.val_max)
        return self.rect.x + int(ratio * self.rect.w)

    def draw(self, surf, fonts, val):
        track_y = self.rect.centery
        pygame.draw.line(surf, C["slider_track"],
                         (self.rect.x, track_y), (self.rect.right, track_y), 4)
        tx = self.get_thumb_x(val)
        pygame.draw.line(surf, C["slider_fill"],
                         (self.rect.x, track_y), (tx, track_y), 4)
        pygame.draw.circle(surf, C["slider_thumb"], (tx, track_y), self.thumb_r)
        pygame.draw.circle(surf, C["accent_dark"],  (tx, track_y), self.thumb_r, 2)

        if self.mode == "delay":
            lbl_l = fonts["tiny"].render("Chậm",  True, C["text_sub"])
            lbl_r = fonts["tiny"].render("Nhanh", True, C["text_sub"])
            val_txt = "Nhanh nhất" if val == 0 else f"{val} ms/bước"
        else:
            lbl_l = fonts["tiny"].render("0%",              True, C["text_sub"])
            lbl_r = fonts["tiny"].render(f"{self.val_max}%", True, C["text_sub"])
            val_txt = f"{val}%"

        surf.blit(lbl_l, (self.rect.x, self.rect.y - 14))
        surf.blit(lbl_r, (self.rect.right - lbl_r.get_width(), self.rect.y - 14))
        vl = fonts["tiny"].render(val_txt, True, C["accent"])
        surf.blit(vl, vl.get_rect(centerx=self.rect.centerx, y=self.rect.y - 14))

    def handle_event(self, event, val):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            tx = self.get_thumb_x(val)
            tr = self.rect.centery
            if abs(event.pos[0] - tx) <= self.thumb_r + 4 and abs(event.pos[1] - tr) <= self.thumb_r + 4:
                self.dragging = True
            elif self.rect.collidepoint(event.pos):
                self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging:
            ratio = (event.pos[0] - self.rect.x) / max(1, self.rect.w)
            ratio = max(0.0, min(1.0, ratio))
            if self.mode == "delay":
                val = int((1.0 - ratio) * self.val_max)
                if val < 8:
                    val = 0
            else:
                val = int(ratio * self.val_max)
            return val
        return val


# ─────────────────────────────────────────────
# Vẽ bản đồ
# ─────────────────────────────────────────────
def draw_map(surf, state, lay, fonts):
    CELL = lay.CELL

    visited_set  = set(state.shown_visited)
    path_set     = set(state.shown_path)
    plan_b_set   = set(state.shown_plan_b)
    belief_set   = set(state.shown_belief)
    rb_path_set  = set(state.shown_robot_b_path)

    is_sensorless  = (state.selected_algo == "Sensorless")
    is_adversarial = (state.selected_algo in ADVERSARIAL_NAMES)
    is_and_or      = (state.selected_algo == "AND-OR")

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = lay.cell_rect(r, c)
            pos  = (r, c)
            val  = state.grid[r][c]

            # Màu nền — ưu tiên: path > plan_b > belief/visited > loại ô
            if val == WALL:
                color = C["cell_wall"]
            elif pos in path_set:
                color = C["cell_path"]
            elif pos in rb_path_set:
                color = C["cell_rb_path"]
            elif pos in plan_b_set:
                color = C["cell_plan_b"]
            elif pos in belief_set:
                color = C["cell_belief"]
            elif pos in visited_set:
                color = C["cell_visited"] if not is_sensorless else C["cell_belief"]
            elif val == ROUGH:
                color = C["cell_rough"]
            else:
                color = C["cell_empty"]

            pygame.draw.rect(surf, color, rect, border_radius=3)

            cx, cy = rect.centerx, rect.centery
            r2 = max(4, CELL // 3)

            # Nội dung ô
            if val == WALL:
                for dr2 in range(4, CELL - 3, 7):
                    for dc2 in range(4, CELL - 3, 7):
                        pygame.draw.circle(surf, (70, 76, 82),
                                           (rect.x + dc2, rect.y + dr2), 1)

            elif val == ROUGH and pos not in path_set and pos not in visited_set:
                for i in range(3):
                    rx2 = rect.x + 4 + i * max(4, CELL // 5)
                    ry2 = rect.y + CELL // 2 - 1 + (i % 2) * 3
                    pygame.draw.circle(surf, C["cell_rough_d"], (rx2, ry2), max(1, CELL // 10))
                if CELL >= 24:
                    lbl = fonts["tiny"].render("×5", True, (130, 80, 20))
                    surf.blit(lbl, (rect.right - lbl.get_width() - 2, rect.y + 2))

            elif val == SAMPLE:
                if pos in state.collected:
                    pygame.draw.circle(surf, C["cell_sample_c"], (cx, cy), r2)
                    pygame.draw.line(surf, (150, 158, 165),
                                     (cx - r2 + 2, cy), (cx + r2 - 2, cy), 2)
                else:
                    pygame.draw.circle(surf, C["cell_sample"], (cx, cy), r2)
                    pygame.draw.circle(surf, (200, 100, 20), (cx, cy), r2, 2)
                    pygame.draw.circle(surf, C["text_white"], (cx, cy), max(2, r2 // 3))

            elif val == BASE:
                s = r2
                pts = [(cx, cy - s), (cx + s, cy), (cx + s, cy + s),
                       (cx - s, cy + s), (cx - s, cy)]
                pygame.draw.polygon(surf, C["cell_base"], pts)
                pygame.draw.polygon(surf, C["accent_dark"], pts, 2)
                pygame.draw.circle(surf, C["text_white"], (cx, cy + s // 3), max(2, s // 3))

            # Ký hiệu ROUGH nhỏ đè lên visited/path để vẫn nhận biết được
            elif val == ROUGH and CELL >= 28:
                lbl = fonts["tiny"].render("×5", True, (160, 110, 30))
                surf.blit(lbl, (rect.right - lbl.get_width() - 2, rect.y + 2))

            # Robot A
            if state.robot_pos and pos == tuple(state.robot_pos):
                pygame.draw.circle(surf, C["cell_robot"], (cx, cy), r2 + 2)
                pygame.draw.circle(surf, (220, 80, 60), (cx, cy), r2 + 2, 2)
                er = max(2, r2 // 4)
                pygame.draw.circle(surf, C["text_white"], (cx - er * 2, cy - er), er)
                pygame.draw.circle(surf, C["text_white"], (cx + er * 2, cy - er), er)

            # Robot B (đối thủ — tím)
            if state.robot_b_pos and pos == tuple(state.robot_b_pos):
                pygame.draw.circle(surf, C["cell_robot_b"], (cx, cy), r2 + 2)
                pygame.draw.circle(surf, (180, 100, 210), (cx, cy), r2 + 2, 2)
                er = max(2, r2 // 4)
                pygame.draw.circle(surf, C["text_white"], (cx - er * 2, cy - er), er)
                pygame.draw.circle(surf, C["text_white"], (cx + er * 2, cy - er), er)

    pygame.draw.rect(surf, C["panel_border"], lay.map_rect(), 2, border_radius=4)

    # Thanh tiến trình animation (dưới bản đồ)
    if state.result and state.animating:
        mr = lay.map_rect()
        bar_y = mr.bottom + 6
        bar_w = mr.width
        total = len(state.result.get("visited", [])) + len(state.result["path"])
        done  = len(state.shown_visited) + len(state.shown_path)
        prog  = done / max(1, total)
        pygame.draw.rect(surf, C["slider_track"], (mr.x, bar_y, bar_w, 4), border_radius=2)
        pygame.draw.rect(surf, C["slider_fill"],  (mr.x, bar_y, int(bar_w * prog), 4), border_radius=2)


# ─────────────────────────────────────────────
# Panel trái
# ─────────────────────────────────────────────
ALGO_NAMES  = ["BFS", "DFS", "Greedy", "A*",
               "Hill Climbing", "Sim. Annealing", "Backtracking", "Forward Checking",
               "AND-OR", "Sensorless", "Minimax", "Alpha-Beta"]
ALGO_GROUPS = [
    ("UNINFORMED",   ["BFS", "DFS"]),
    ("INFORMED",     ["Greedy", "A*"]),
    ("LOCAL",        ["Hill Climbing", "Sim. Annealing"]),
    ("CSP",          ["Backtracking", "Forward Checking"]),
    ("NON-DETERM.",  ["AND-OR", "Sensorless"]),
    ("ADVERSARIAL",  ["Minimax", "Alpha-Beta"]),
]


def build_left_items(state, PW, slider, slip_slider=None):
    PAD = 12
    BW  = PW - PAD * 2 - 8
    BH  = 28
    x   = PAD
    items = []
    y = 10

    items.append({"type": "text_h", "x": x, "y": y, "text": "Mars Robot",
                  "color": C["accent"], "font": "heading"})
    y += 20
    items.append({"type": "text", "x": x, "y": y, "text": "Explorer v2.0",
                  "color": C["text_sub"], "font": "tiny"})
    y += 22

    if state.show_path_log:
        items.append({"type": "btn_ref", "key": "back_log",
                      "x": x, "y": y, "w": BW, "h": BH})
        y += BH + 8
        items.append({"type": "divider", "x": x, "y": y, "w": BW})
        y += 10

        algo = state.selected_algo

        if not state.result or not state.result["path"]:
            items.append({"type": "text", "x": x, "y": y, "text": "Chưa có kết quả.",
                          "color": C["text_sub"], "font": "small"})
            y += 20
            return items, y + 10

        path = state.result["path"]
        s    = state.result["stats"]

        # ── Adversarial: hiện bảng điểm + path 2 robot ──
        if algo in ADVERSARIAL_NAMES:
            sa  = s.get("score_a", 0)
            sb  = s.get("score_b", 0)
            res = s.get("result", "—")
            rc  = C["success"] if res == "THẮNG" else (C["danger"] if res == "THUA" else C["warning"])

            items.append({"type": "text", "x": x, "y": y,
                          "text": f"Kết quả: {res}", "color": rc, "font": "small"})
            y += 18
            items.append({"type": "text", "x": x, "y": y,
                          "text": f"Robot A (đỏ): {sa} mẫu",
                          "color": C["cell_robot"], "font": "small"})
            y += 16
            items.append({"type": "text", "x": x, "y": y,
                          "text": f"Robot B (tím): {sb} mẫu",
                          "color": C["cell_robot_b"], "font": "small"})
            y += 18
            items.append({"type": "divider", "x": x, "y": y, "w": BW})
            y += 8

            # Path Robot A
            items.append({"type": "text", "x": x, "y": y,
                          "text": f"Robot A — {len(path)-1} bước:",
                          "color": C["cell_robot"], "font": "small"})
            y += 16
            for p in path:
                pt = tuple(p)
                val = state.grid[pt[0]][pt[1]]
                note = " (×5)" if val == ROUGH else ""
                col  = C["warning"] if val == ROUGH else C["text_sub"]
                items.append({"type": "text", "x": x + 8, "y": y,
                              "text": f"({pt[0]},{pt[1]}){note}", "color": col, "font": "tiny"})
                y += 13
            y += 6

            rb_path = state.result.get("robot_b_path", [])
            items.append({"type": "text", "x": x, "y": y,
                          "text": f"Robot B — {len(rb_path)-1} bước:",
                          "color": C["cell_robot_b"], "font": "small"})
            y += 16
            for p in rb_path:
                pt = tuple(p)
                items.append({"type": "text", "x": x + 8, "y": y,
                              "text": f"({pt[0]},{pt[1]})", "color": C["text_sub"], "font": "tiny"})
                y += 13

        # ── AND-OR: path chính + tóm tắt kế hoạch B ──
        elif algo == "AND-OR":
            slip = state.result.get("slip_prob", state.slip_prob)
            nb   = s.get("plan_b_count", 0)
            items.append({"type": "text", "x": x, "y": y,
                          "text": f"Xác suất trượt: {slip*100:.0f}%",
                          "color": C["warning"], "font": "small"})
            y += 16
            items.append({"type": "text", "x": x, "y": y,
                          "text": f"Kế hoạch phục hồi: {nb} đoạn (cam)",
                          "color": C["cell_plan_b"], "font": "small"})
            y += 18
            items.append({"type": "divider", "x": x, "y": y, "w": BW})
            y += 8
            items.append({"type": "text", "x": x, "y": y,
                          "text": f"Plan A — {len(path)-1} bước:",
                          "color": C["cell_path"], "font": "small"})
            y += 16
            for i, p in enumerate(path):
                pt  = tuple(p)
                val = state.grid[pt[0]][pt[1]]
                note = " (×5)" if val == ROUGH else ""
                col  = C["warning"] if val == ROUGH else C["text_sub"]
                items.append({"type": "text", "x": x + 8, "y": y,
                              "text": f"({pt[0]},{pt[1]}){note}", "color": col, "font": "tiny"})
                y += 13
            y += 6
            if nb > 0:
                items.append({"type": "text", "x": x, "y": y,
                              "text": "Plan B (nếu trượt tại ROUGH):",
                              "color": C["cell_plan_b"], "font": "small"})
                y += 16
                for seg in state.result.get("plan_b_segments", []):
                    seg_txt = " → ".join(f"({p[0]},{p[1]})" for p in seg[:4])
                    if len(seg) > 4:
                        seg_txt += "…"
                    items.append({"type": "text", "x": x + 8, "y": y,
                                  "text": seg_txt, "color": C["text_sub"], "font": "tiny"})
                    y += 13

        # ── Sensorless: belief history + path ──
        elif algo == "Sensorless":
            bi = s.get("belief_init", 0)
            bf = s.get("belief_final", 0)
            items.append({"type": "text", "x": x, "y": y,
                          "text": f"Belief: {bi} → {bf} ô",
                          "color": C["cell_belief"], "font": "small"})
            y += 16
            bh = state.result.get("belief_history", [])
            items.append({"type": "text", "x": x, "y": y,
                          "text": f"Số bước thu hẹp: {len(bh)}",
                          "color": C["text_sub"], "font": "small"})
            y += 18
            items.append({"type": "divider", "x": x, "y": y, "w": BW})
            y += 8
            items.append({"type": "text", "x": x, "y": y,
                          "text": f"Path sau xác định — {len(path)-1} bước:",
                          "color": C["cell_path"], "font": "small"})
            y += 16
            for p in path:
                pt  = tuple(p)
                val = state.grid[pt[0]][pt[1]]
                note = " (×5)" if val == ROUGH else ""
                col  = C["warning"] if val == ROUGH else C["text_sub"]
                items.append({"type": "text", "x": x + 8, "y": y,
                              "text": f"({pt[0]},{pt[1]}){note}", "color": col, "font": "tiny"})
                y += 13

        # ── Các thuật toán thông thường ──
        else:
            items.append({"type": "text", "x": x, "y": y,
                          "text": f"Tổng {len(path)-1} bước:",
                          "color": C["text_sub"], "font": "small"})
            y += 18
            waypoints = [state.robot_start] + list(state.samples) + [state.base]
            pi, seg = 0, 1
            for wi in range(len(waypoints) - 1):
                frm, to = waypoints[wi], waypoints[wi + 1]
                items.append({"type": "text", "x": x, "y": y,
                              "text": f"Đoạn {seg}: {frm}→{to}",
                              "color": C["accent"], "font": "small"})
                y += 16
                seg += 1
                while pi < len(path):
                    p = tuple(path[pi])
                    val  = state.grid[p[0]][p[1]]
                    note = " (×5)" if val == ROUGH else ""
                    col  = C["warning"] if val == ROUGH else C["text_sub"]
                    items.append({"type": "text", "x": x + 8, "y": y,
                                  "text": f"({p[0]},{p[1]}){note}",
                                  "color": col, "font": "tiny"})
                    y += 13
                    pi += 1
                    if pi < len(path) and tuple(path[pi]) == tuple(to):
                        pi += 1
                        break
            y += 6

        return items, y + 10

    # ── Chọn thuật toán ──
    for g_name, algos in ALGO_GROUPS:
        items.append({"type": "text", "x": x, "y": y, "text": g_name,
                      "color": C["text_sub"], "font": "tiny"})
        y += 14
        for name in algos:
            items.append({"type": "btn_ref", "key": f"algo_{name}",
                          "x": x, "y": y, "w": BW, "h": BH})
            y += BH + 3
        y += 5

    items.append({"type": "divider", "x": x, "y": y, "w": BW})
    y += 10

    # Combo
    items.append({"type": "btn_ref", "key": "combo_toggle",
                  "x": x, "y": y, "w": BW, "h": BH})
    y += BH + 6

    if state.combo_mode:
        note_lines = [
            "Kết hợp tối ưu:",
            f"  · {COMBO_PATH}: tìm đường",
            "    tính chi phí ×5",
            f"  · {COMBO_ORDER}: tối ưu",
            "    thứ tự thu mẫu",
        ]
        for line in note_lines:
            items.append({"type": "text", "x": x + 4, "y": y, "text": line,
                          "color": C["text_sub"], "font": "tiny"})
            y += 13
        y += 4

    items.append({"type": "divider", "x": x, "y": y, "w": BW})
    y += 10

    # Slider tốc độ
    items.append({"type": "text", "x": x, "y": y, "text": "Tốc độ animation",
                  "color": C["text"], "font": "small"})
    y += 16
    slider.rect.x = x
    slider.rect.y = y + 14
    slider.rect.w = BW
    items.append({"type": "slider", "x": x, "y": y + 14, "w": BW})
    y += 42

    # Slider xác suất trượt (chỉ hiện khi AND-OR được chọn)
    if state.selected_algo == "AND-OR" and slip_slider is not None:
        items.append({"type": "divider", "x": x, "y": y, "w": BW})
        y += 10
        items.append({"type": "text", "x": x, "y": y,
                      "text": "Xác suất trượt (ROUGH)",
                      "color": C["warning"], "font": "small"})
        y += 16
        slip_slider.rect.x = x
        slip_slider.rect.y = y + 14
        slip_slider.rect.w = BW
        items.append({"type": "slip_slider", "x": x, "y": y + 14, "w": BW})
        y += 42

    items.append({"type": "divider", "x": x, "y": y, "w": BW})
    y += 10

    for key in ["new_map", "run", "reset"]:
        items.append({"type": "btn_ref", "key": key,
                      "x": x, "y": y, "w": BW, "h": 32})
        y += 37

    return items, y + 10


def draw_left_panel(surf, state, btns, fonts, lay, slider, slip_slider=None):
    PW = lay.PANEL_W
    SH = lay.SH
    PAD = 12

    pygame.draw.rect(surf, C["panel_bg"], (0, 0, PW, SH))
    pygame.draw.line(surf, C["panel_border"], (PW, 0), (PW, SH), 1)

    items, total_h = build_left_items(state, PW, slider, slip_slider)
    scroll = state.left_scroll

    clip = pygame.Rect(0, 0, PW - 8, SH)
    surf.set_clip(clip)

    for item in items:
        sy = item["y"] - scroll
        if sy > SH or sy < -60:
            continue
        t = item["type"]

        if t in ("text", "text_h"):
            f = fonts[item.get("font", "normal")]
            txt = f.render(item["text"], True, item["color"])
            surf.blit(txt, (item["x"], sy))

        elif t == "divider":
            pygame.draw.line(surf, C["divider"],
                             (item["x"], sy), (item["x"] + item["w"], sy))

        elif t == "btn_ref":
            key = item["key"]
            if key in btns:
                btn = btns[key]
                btn.rect.x = item["x"]
                btn.rect.y = item["y"]
                btn.rect.w = item["w"]
                btn.rect.h = item["h"]
                btn.draw(surf, fonts["small"], offset_y=-scroll)

        elif t == "slider":
            slider.rect.x = item["x"]
            slider.rect.y = item["y"] - scroll
            slider.rect.w = item["w"]
            slider.draw(surf, fonts, state.delay_ms)

        elif t == "slip_slider" and slip_slider is not None:
            slip_slider.rect.x = item["x"]
            slip_slider.rect.y = item["y"] - scroll
            slip_slider.rect.w = item["w"]
            slip_slider.draw(surf, fonts, int(state.slip_prob * 100))

    surf.set_clip(None)

    # Scrollbar
    if total_h > SH:
        sb_x = PW - 7
        sb_h = max(30, int(SH * SH / total_h))
        sb_y = int(scroll * SH / total_h)
        pygame.draw.rect(surf, C["scrollbar"],   (sb_x, 0,    5, SH),   border_radius=3)
        pygame.draw.rect(surf, C["scrollbar_h"], (sb_x, sb_y, 5, sb_h), border_radius=3)

    return total_h


# ─────────────────────────────────────────────
# Panel phải
# ─────────────────────────────────────────────
def draw_right_panel(surf, state, btns, fonts, lay):
    PW  = lay.PANEL_W
    SW, SH = lay.SW, lay.SH
    RX  = SW - PW
    PAD = 12
    x   = RX + PAD

    pygame.draw.rect(surf, C["panel_bg"], (RX, 0, PW, SH))
    pygame.draw.line(surf, C["panel_border"], (RX, 0), (RX, SH), 1)

    f_h = fonts["heading"]
    f_n = fonts["normal"]
    f_s = fonts["small"]
    f_t = fonts["tiny"]

    surf.blit(f_h.render("Thống kê", True, C["text"]), (x, 14))
    y = 36

    # Trạng thái animation
    if state.animating:
        if state.anim_phase == 0:
            phase_lbl = f"Đang duyệt... ({len(state.shown_visited)} ô)"
        else:
            phase_lbl = f"Robot di chuyển... ({len(state.shown_path)} bước)"
        surf.blit(f_t.render(phase_lbl, True, C["warning"]), (x, y))
        y += 16

    algo = state.selected_algo
    is_adversarial = algo in ADVERSARIAL_NAMES

    # Thống kê chung
    stats = [
        ("Số bước",  str(state.result["stats"]["steps"])         if state.result else "—", C["text"]),
        ("Chi phí",  str(state.result["stats"].get("cost", "—")) if state.result else "—", C["accent"]),
        ("Ô đã xét", str(state.result["stats"]["visited"])       if state.result else "—", C["text"]),
        ("Thời gian",f"{state.run_time_ms:.1f} ms"               if state.run_time_ms else "—", C["text"]),
    ]
    if not is_adversarial:
        stats.append(("Mẫu thu", f"{len(state.collected)} / {NUM_SAMPLES}",
                      C["success"] if len(state.collected) == NUM_SAMPLES else C["warning"]))
    for label, val, vc in stats:
        surf.blit(f_t.render(label, True, C["text_sub"]), (x, y))
        surf.blit(f_n.render(val,   True, vc),            (x, y + 13))
        y += 34

    # Thống kê đặc thù từng nhóm
    if state.result:
        s = state.result["stats"]

        if algo == "AND-OR":
            pygame.draw.line(surf, C["divider"], (x, y), (SW - PAD, y)); y += 8
            surf.blit(f_t.render("Xác suất trượt", True, C["text_sub"]), (x, y))
            surf.blit(f_n.render(f"{state.slip_prob*100:.0f}%", True, C["warning"]), (x, y + 13))
            y += 34
            surf.blit(f_t.render("Kế hoạch dự phòng", True, C["text_sub"]), (x, y))
            surf.blit(f_n.render(str(s.get("plan_b_count", 0)), True, C["warning"]), (x, y + 13))
            y += 34

        elif algo == "Sensorless":
            pygame.draw.line(surf, C["divider"], (x, y), (SW - PAD, y)); y += 8
            surf.blit(f_t.render("Belief ban đầu", True, C["text_sub"]), (x, y))
            surf.blit(f_n.render(str(s.get("belief_init", "—")), True, C["cell_belief"]), (x, y + 13))
            y += 34
            surf.blit(f_t.render("Belief cuối", True, C["text_sub"]), (x, y))
            bf = s.get("belief_final", 0)
            surf.blit(f_n.render(str(bf), True, C["success"] if bf <= 1 else C["warning"]), (x, y + 13))
            y += 34

        elif is_adversarial:
            pygame.draw.line(surf, C["divider"], (x, y), (SW - PAD, y)); y += 8
            # Bảng điểm
            surf.blit(f_s.render("Bảng điểm", True, C["text_sub"]), (x, y)); y += 14
            # Robot A
            sa = s.get("score_a", 0)
            pygame.draw.rect(surf, C["cell_robot"], (x, y, 10, 10), border_radius=2)
            surf.blit(f_s.render(f" Robot A: {sa} mẫu", True, C["cell_robot"]), (x + 12, y)); y += 16
            # Robot B
            sb = s.get("score_b", 0)
            pygame.draw.rect(surf, C["cell_robot_b"], (x, y, 10, 10), border_radius=2)
            surf.blit(f_s.render(f" Robot B: {sb} mẫu", True, C["cell_robot_b"]), (x + 12, y)); y += 18
            # Kết quả
            res = s.get("result", "—")
            rc = C["success"] if res == "THẮNG" else (C["danger"] if res == "THUA" else C["warning"])
            surf.blit(f_t.render("Kết quả", True, C["text_sub"]), (x, y))
            surf.blit(f_n.render(res, True, rc), (x, y + 13)); y += 34
            # Node đã xét
            surf.blit(f_t.render("Node Minimax", True, C["text_sub"]), (x, y))
            surf.blit(f_n.render(str(s.get("nodes_explored", "—")), True, C["text"]), (x, y + 13))
            y += 34

    pygame.draw.line(surf, C["divider"], (x, y), (SW - PAD, y))
    y += 8

    vp = btns["view_path"]
    vp.rect = pygame.Rect(x, y, PW - PAD * 2, 28)
    vp.draw(surf, f_s)
    y += 36

    pygame.draw.line(surf, C["divider"], (x, y), (SW - PAD, y))
    y += 8

    surf.blit(f_h.render("Chú thích", True, C["text"]), (x, y))
    y += 20
    legend = [
        (C["cell_robot"],   "Robot A"),
        (C["cell_base"],    "Base (đích)"),
        (C["cell_sample"],  "Mẫu vật"),
        (C["cell_rough"],   "Địa hình ×5"),
        (C["cell_path"],    "Đường đi"),
        (C["cell_visited"], "Đã xét"),
        (C["cell_wall"],    "Tường"),
        (C["cell_plan_b"],  "Plan B (AND-OR)"),
        (C["cell_belief"],  "Belief state"),
        (C["cell_robot_b"], "Robot B (địch)"),
    ]
    for color, name in legend:
        pygame.draw.rect(surf, color, (x, y + 2, 13, 13), border_radius=3)
        pygame.draw.rect(surf, C["panel_border"], (x, y + 2, 13, 13), 1, border_radius=3)
        surf.blit(f_s.render(name, True, C["text"]), (x + 18, y))
        y += 19

    pygame.draw.line(surf, C["divider"], (x, y + 4), (SW - PAD, y + 4))
    y += 14

    surf.blit(f_h.render("Nhật ký", True, C["text"]), (x, y))
    y += 18
    max_lines = max(1, (SH - y - 10) // 15)
    for line in state.log_lines[-max_lines:]:
        col = C["success"] if "✓" in line else (C["danger"] if "✗" in line else C["text_sub"])
        surf.blit(f_t.render(line, True, col), (x, y))
        y += 15


# ─────────────────────────────────────────────
# Help overlay
# ─────────────────────────────────────────────
HELP_CONTENT = [
    ("🗺  Các loại ô trên bản đồ", "heading"),
    ("Ô trống (trắng) — Chi phí đi vào: 1", "normal"),
    ("Địa hình gồ ghề (nâu) — Chi phí: 5. A* tránh nếu có đường rẻ hơn.", "normal"),
    ("Tường (đen) — Không thể đi qua.", "normal"),
    ("Mẫu vật (cam) — Robot cần thu đủ 6 mẫu.", "normal"),
    ("Base (xanh dương) — Điểm xuất phát và đích đến cuối.", "normal"),
    ("", "spacer"),
    ("🎬  Animation theo từng nhóm thuật toán", "heading"),
    ("Nhóm 1–3, 6 (thông thường):", "normal"),
    ("  Pha vàng: các ô thuật toán đã xét (visited).", "normal"),
    ("  Pha xanh lá: robot di chuyển theo đường đi.", "normal"),
    ("Nhóm 4 — AND-OR:", "normal"),
    ("  Pha vàng: visited → pha cam: kế hoạch phục hồi (Plan B) → pha xanh: robot đi.", "normal"),
    ("  Slider 'Xác suất trượt' điều chỉnh khả năng robot trượt trên ROUGH (0–50%).", "normal"),
    ("Nhóm 4 — Sensorless:", "normal"),
    ("  Pha tím: belief state thu hẹp dần (robot xác định vị trí).", "normal"),
    ("  Pha xanh lá: robot di chuyển sau khi đã biết vị trí.", "normal"),
    ("Nhóm 5 — Minimax / Alpha-Beta:", "normal"),
    ("  Hai robot (đỏ = ta, tím = địch) di chuyển đồng thời, đấu tranh thu mẫu.", "normal"),
    ("  Panel phải hiển thị bảng điểm và kết quả THẮNG / THUA / HÒA.", "normal"),
    ("", "spacer"),
    ("🤖  Ràng buộc khi duyệt node", "heading"),
    ("• Robot chỉ đi 4 hướng: lên, xuống, trái, phải.", "normal"),
    ("• Không ra ngoài biên hoặc vào ô tường.", "normal"),
    ("• Mỗi ô chỉ xét 1 lần để tránh vòng lặp (trừ Sensorless dùng belief state).", "normal"),
    ("", "spacer"),
    ("🧠  Nhóm 1 — Tìm kiếm mù thông tin", "heading"),
    ("BFS — Duyệt theo từng lớp (FIFO). Ít bước nhất, không tính chi phí ROUGH.", "algo"),
    ("DFS — Duyệt sâu (stack). Không tối ưu, tiêu thụ ít bộ nhớ hơn BFS.", "algo"),
    ("", "spacer"),
    ("🧠  Nhóm 2 — Tìm kiếm có thông tin", "heading"),
    ("Greedy — Chọn ô có h(n)=Manhattan nhỏ nhất. Nhanh nhưng không tối ưu.", "algo"),
    ("A* — f=g+h, g tính chi phí thực (ROUGH×5). Tối ưu chi phí.", "algo"),
    ("", "spacer"),
    ("🧠  Nhóm 3 — Tìm kiếm cục bộ (tối ưu thứ tự thu mẫu)", "heading"),
    ("Hill Climbing — Hoán đổi 2 mẫu, giữ nếu tổng đường đi ngắn hơn.", "algo"),
    ("Sim. Annealing — Như HC nhưng chấp nhận nghiệm xấu hơn theo e^(-ΔE/T).", "algo"),
    ("", "spacer"),
    ("🧠  Nhóm 4 — Không xác định (Non-deterministic)", "heading"),
    ("AND-OR — Xây kế hoạch có điều kiện: Plan A (bình thường) + Plan B (nếu trượt ROUGH).", "algo"),
    ("Sensorless — Robot không biết vị trí ban đầu; thu hẹp belief state bằng greedy.", "algo"),
    ("", "spacer"),
    ("🧠  Nhóm 5 — Đối kháng (Adversarial)", "heading"),
    ("Minimax — Robot A (MAX) và B (MIN) đi xen kẽ, depth=3, không cắt nhánh.", "algo"),
    ("Alpha-Beta — Minimax + cắt nhánh α/β, depth=4. Nhanh hơn ~2× Minimax.", "algo"),
    ("", "spacer"),
    ("🧠  Nhóm 6 — Ràng buộc CSP (tối ưu thứ tự thu mẫu)", "heading"),
    ("Backtracking — Thử mọi hoán vị, cắt nhánh khi vượt chi phí tốt nhất.", "algo"),
    ("Forward Checking — BT + thu hẹp domain sau mỗi lần gán.", "algo"),
    ("", "spacer"),
    ("⚡  Kết hợp tối ưu: A* + Sim. Annealing", "heading"),
    ("SA tối ưu thứ tự thu mẫu → A* đi theo thứ tự đó với chi phí thực nhỏ nhất.", "normal"),
    ("", "spacer"),
    ("⌨️  Phím tắt", "heading"),
    ("Space — Chạy  |  N — Bản đồ mới  |  R — Reset  |  Esc — Đóng/Thoát", "normal"),
]


def draw_help_overlay(surf, state, fonts, close_btn, lay):
    SW, SH = lay.SW, lay.SH
    ov = pygame.Surface((SW, SH), pygame.SRCALPHA)
    ov.fill((*C["overlay_bg"], 210))
    surf.blit(ov, (0, 0))

    PW2 = min(680, SW - 80)
    PH2 = SH - 100
    px  = (SW - PW2) // 2
    py  = 50

    pygame.draw.rect(surf, C["overlay_panel"],  (px, py, PW2, PH2), border_radius=14)
    pygame.draw.rect(surf, C["overlay_border"], (px, py, PW2, PH2), 2, border_radius=14)

    title = fonts["heading_lg"].render("Hướng dẫn sử dụng", True, C["help_accent"])
    surf.blit(title, title.get_rect(centerx=SW // 2, y=py + 14))

    cy0  = py + 48
    clip = pygame.Rect(px + 16, cy0, PW2 - 32, PH2 - 64)
    surf.set_clip(clip)

    y = cy0 - state.help_scroll
    for text, style in HELP_CONTENT:
        if style == "spacer":
            y += 8; continue
        if style == "heading":
            if cy0 - 30 <= y < cy0 + PH2:
                surf.blit(fonts["heading"].render(text, True, C["help_accent"]), (px + 20, y))
            y += 24
        elif style == "algo":
            if cy0 - 30 <= y < cy0 + PH2:
                parts = text.split(" — ", 1)
                xp = px + 24
                if len(parts) == 2:
                    n = fonts["small_bold"].render(parts[0] + " — ", True, (100, 200, 255))
                    surf.blit(n, (xp, y))
                    surf.blit(fonts["small"].render(parts[1], True, (190, 200, 215)),
                              (xp + n.get_width(), y))
                else:
                    surf.blit(fonts["small"].render(text, True, (190, 200, 215)), (xp, y))
            y += 19
        else:
            if cy0 - 30 <= y < cy0 + PH2:
                surf.blit(fonts["small"].render(text, True, (190, 200, 215)), (px + 24, y))
            y += 18

    surf.set_clip(None)

    close_btn.rect = pygame.Rect(px + PW2 - 108, py + PH2 - 42, 88, 30)
    close_btn.draw(surf, fonts["normal"])

    hint = fonts["tiny"].render("Cuộn chuột để xem thêm", True, (100, 115, 135))
    surf.blit(hint, hint.get_rect(centerx=SW // 2, y=py + PH2 - 16))


# ─────────────────────────────────────────────
# Tạo buttons
# ─────────────────────────────────────────────
def make_buttons(state, lay):
    PW  = lay.PANEL_W
    PAD = 12
    BW  = PW - PAD * 2 - 8
    BH  = 28
    x   = PAD
    btns = {}

    for name in ALGO_NAMES:
        b = Button((x, 0, BW, BH), name)
        b.active = (name == state.selected_algo) and not state.combo_mode
        # Màu nút đặc thù
        if name in ADVERSARIAL_NAMES:
            b.bg = (80, 40, 100) if b.active else None
        elif name in NONDETERMINISTIC_NAMES:
            b.bg = (40, 80, 40) if b.active else None
        btns[f"algo_{name}"] = b

    combo_lbl = "✓ Kết hợp: BẬT" if state.combo_mode else "Kết hợp: TẮT"
    btns["combo_toggle"] = Button((x, 0, BW, BH), combo_lbl)
    btns["combo_toggle"].active = state.combo_mode

    btns["new_map"] = Button((x, 0, BW, 32), "🗺  Sinh bản đồ mới",
                              bg=C["success"], tc=C["text_white"])
    btns["new_map"].active = True
    btns["run"]     = Button((x, 0, BW, 32), "▶  Chạy",
                              bg=C["accent"],  tc=C["text_white"])
    btns["run"].active = True
    btns["reset"]   = Button((x, 0, BW, 32), "↺  Reset")

    btns["view_path"]  = Button((0, 0, 100, BH), "📋 Xem đường đi chi tiết")
    btns["view_path"].active = state.show_path_log
    btns["back_log"]   = Button((x, 0, BW, BH), "← Quay lại")
    btns["help_close"] = Button((0, 0, 88, 30), "✕ Đóng",
                                 bg=C["danger"], tc=C["text_white"])
    btns["help_close"].active = True
    return btns


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    pygame.init()
    SW, SH = 1100, 700
    screen = pygame.display.set_mode((SW, SH), pygame.RESIZABLE)
    pygame.display.set_caption("Mars Robot Explorer")
    clock  = pygame.time.Clock()

    def try_font(names, size, bold=False):
        for name in names:
            try:
                f = pygame.font.SysFont(name, size, bold=bold)
                if f: return f
            except: pass
        return pygame.font.SysFont("arial", size, bold=bold)

    fonts = {
        "heading_lg": try_font(["Segoe UI", "Arial"], 20, bold=True),
        "heading":    try_font(["Segoe UI", "Arial"], 13, bold=True),
        "normal":     try_font(["Segoe UI", "Arial"], 12),
        "small":      try_font(["Segoe UI", "Arial"], 11),
        "small_bold": try_font(["Segoe UI", "Arial"], 11, bold=True),
        "tiny":       try_font(["Segoe UI", "Arial"], 10),
    }

    state       = AppState()
    lay         = Layout(SW, SH)
    btns        = make_buttons(state, lay)
    slider      = Slider(12, 0, lay.PANEL_W - 28)
    slip_slider = Slider(12, 0, lay.PANEL_W - 28, val_max=50, mode="pct")  # Slider xác suất 0-50%

    help_hovered  = False
    left_total_h  = SH
    dragging_sb   = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.VIDEORESIZE:
                SW = max(MIN_W, event.w)
                SH = max(MIN_H, event.h)
                screen = pygame.display.set_mode((SW, SH), pygame.RESIZABLE)
                lay         = Layout(SW, SH)
                btns        = make_buttons(state, lay)
                slip_slider = Slider(12, 0, lay.PANEL_W - 28, val_max=50, mode="pct")
                state.left_scroll = max(0, min(state.left_scroll,
                                               max(0, left_total_h - SH)))

            if event.type == pygame.MOUSEWHEEL:
                mx, _ = pygame.mouse.get_pos()
                if state.show_help:
                    state.help_scroll = max(0, state.help_scroll - event.y * 22)
                elif mx < lay.PANEL_W:
                    state.left_scroll = max(0, min(
                        state.left_scroll - event.y * 22,
                        max(0, left_total_h - SH)
                    ))

            # Slider tốc độ
            new_delay = slider.handle_event(event, state.delay_ms)
            if new_delay != state.delay_ms:
                state.delay_ms = new_delay

            # Slider xác suất trượt (AND-OR)
            if state.selected_algo == "AND-OR":
                cur_slip_val = int(state.slip_prob * 100)
                new_slip_val = slip_slider.handle_event(event, cur_slip_val)
                if new_slip_val != cur_slip_val:
                    state.slip_prob = new_slip_val / 100.0

            # Kéo scrollbar
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                sb_x = lay.PANEL_W - 7
                if sb_x <= mx <= sb_x + 5 and left_total_h > SH:
                    dragging_sb = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_sb = False
            if event.type == pygame.MOUSEMOTION and dragging_sb:
                _, dy = event.rel
                state.left_scroll = max(0, min(
                    state.left_scroll + int(dy * left_total_h / max(1, SH)),
                    max(0, left_total_h - SH)
                ))

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if state.show_help: state.show_help = False
                    else: running = False
                elif not state.show_help:
                    if event.key == pygame.K_SPACE:
                        state.run_algorithm()
                        btns = make_buttons(state, lay)
                    elif event.key == pygame.K_r:
                        state.reset_run()
                        btns = make_buttons(state, lay)
                    elif event.key == pygame.K_n:
                        state.new_map()
                        btns = make_buttons(state, lay)

            if event.type == pygame.MOUSEMOTION:
                help_hovered = pygame.Rect(8, 8, 32, 32).collidepoint(event.pos)
                if not state.show_help:
                    for k, b in btns.items():
                        off = -state.left_scroll if k not in ("view_path","help_close") else 0
                        b.handle(event, off)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                if pygame.Rect(8, 8, 32, 32).collidepoint((mx, my)):
                    state.show_help = not state.show_help
                    state.help_scroll = 0
                    continue

                if state.show_help:
                    if btns["help_close"].rect.collidepoint((mx, my)):
                        state.show_help = False
                    continue

                # Bỏ qua click lên slider
                if slider.dragging:
                    continue

                scroll = state.left_scroll
                for name in ALGO_NAMES:
                    if btns[f"algo_{name}"].hit((mx, my), -scroll):
                        state.selected_algo = name
                        state.combo_mode    = False
                        btns = make_buttons(state, lay)
                        break

                if btns["combo_toggle"].hit((mx, my), -scroll):
                    state.combo_mode = not state.combo_mode
                    btns = make_buttons(state, lay)

                if btns["new_map"].hit((mx, my), -scroll):
                    state.new_map(); btns = make_buttons(state, lay)
                if btns["run"].hit((mx, my), -scroll):
                    state.run_algorithm(); btns = make_buttons(state, lay)
                if btns["reset"].hit((mx, my), -scroll):
                    state.reset_run(); btns = make_buttons(state, lay)

                if btns["view_path"].rect.collidepoint((mx, my)):
                    state.show_path_log = not state.show_path_log
                    state.left_scroll   = 0
                    btns = make_buttons(state, lay)

                if state.show_path_log and btns["back_log"].hit((mx, my), -scroll):
                    state.show_path_log = False
                    state.left_scroll   = 0
                    btns = make_buttons(state, lay)

        state.update_animation()

        # ── Vẽ ──
        screen.fill(C["bg"])
        draw_map(screen, state, lay, fonts)
        left_total_h = draw_left_panel(screen, state, btns, fonts, lay, slider, slip_slider)

        # Hien thi overlay "Dang tinh toan..." khi chay thuat toan nang
        if state.calculating:
            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, (0, 0))
            dots = "." * (int(time.time() * 2) % 4)
            calc_surf = fonts["heading_lg"].render(f"Đang tính toán{dots}", True, (255, 255, 255))
            cx = screen.get_width()  // 2 - calc_surf.get_width()  // 2
            cy = screen.get_height() // 2 - calc_surf.get_height() // 2
            screen.blit(calc_surf, (cx, cy))
        draw_right_panel(screen, state, btns, fonts, lay)

        # Nút ?
        qr = pygame.Rect(8, 8, 32, 32)
        pygame.draw.circle(screen, C["accent"] if help_hovered else C["btn_normal"], qr.center, 16)
        pygame.draw.circle(screen, C["btn_border"], qr.center, 16, 1)
        qt = fonts["heading"].render("?", True, C["text_white"] if help_hovered else C["accent"])
        screen.blit(qt, qt.get_rect(center=qr.center))

        if state.show_help:
            draw_help_overlay(screen, state, fonts, btns["help_close"], lay)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
