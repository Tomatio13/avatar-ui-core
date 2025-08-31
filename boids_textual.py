#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import colorsys
from dataclasses import dataclass

import numpy as np
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.reactive import reactive
from textual.containers import Container


# ====== パラメータ ======
@dataclass
class BoidsParams:
    max_speed: float = 2.8
    max_force: float = 0.08
    align_radius: float = 10.0
    cohere_radius: float = 12.0
    separate_radius: float = 6.0
    count: int = 120
    tank_w: int | None = None   # 水槽幅（セル）
    tank_h: int | None = None   # 水槽高（セル）
    restitution: float = 1.0    # 壁反発係数（1.0=完全反射）


class BoidsWidget(Widget):
    """Textual上でBoidsをASCIIレンダリングするウィジェット（個体ごとに色付け）。

    draw_border を False にすると水槽の壁を描画しません。
    """

    paused: bool = reactive(False)
    fps_target: float = reactive(30.0)

    def __init__(self, params: BoidsParams, draw_border: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        self.draw_border = draw_border
        self.positions: np.ndarray | None = None  # (N,2)
        self.velocities: np.ndarray | None = None  # (N,2)
        self.boid_colors: list[str] | None = None  # (N,) Richスタイル文字列 "#rrggbb"

        self._rng = np.random.default_rng()
        self._frame_count = 0
        self._last_fps_stamp = time.perf_counter()
        self._fps_measured = 0.0

        # 表示用レイアウト（水槽の位置とサイズ）
        self._off_x = 0
        self._off_y = 0
        self._tank_w = 1
        self._tank_h = 1

    # --- ライフサイクル ---
    def on_mount(self) -> None:
        self._compute_layout()
        self._init_flock()
        self.set_interval(1 / self.fps_target, self._tick)

    def on_resize(self, _) -> None:
        self._compute_layout()
        if self.positions is not None:
            self.positions[:, 0] = np.clip(self.positions[:, 0], 0, self._tank_w - 1)
            self.positions[:, 1] = np.clip(self.positions[:, 1], 0, self._tank_h - 1)
        self.refresh()

    # --- レイアウト計算（水槽を中央に配置） ---
    def _compute_layout(self) -> None:
        W = max(self.size.width, 1)
        H = max(self.size.height, 1)
        border = 2 if self.draw_border else 0
        inner_w = max(W - border, 1)  # 枠線ぶんを引いた内寸
        inner_h = max(H - border, 1)

        req_w = self.params.tank_w if self.params.tank_w else inner_w
        req_h = self.params.tank_h if self.params.tank_h else inner_h

        self._tank_w = max(1, min(req_w, inner_w))
        self._tank_h = max(1, min(req_h, inner_h))

        box_w = self._tank_w + (2 if self.draw_border else 0)
        box_h = self._tank_h + (2 if self.draw_border else 0)
        self._off_x = max((W - box_w) // 2, 0)
        self._off_y = max((H - box_h) // 2, 0)

    # --- 色ユーティリティ（鮮やか系HSV→RGB） ---
    def _rand_color_hex(self) -> str:
        h = self._rng.random()  # 0..1
        s = 0.75 + 0.2 * self._rng.random()    # 0.75..0.95
        v = 0.85 + 0.1 * self._rng.random()    # 0.85..0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        R = int(r * 255)
        G = int(g * 255)
        B = int(b * 255)
        return f"#{R:02x}{G:02x}{B:02x}"

    # --- 初期化/リセット ---
    def _init_flock(self) -> None:
        n = self.params.count
        self.positions = self._rng.random((n, 2)) * np.array([self._tank_w, self._tank_h], dtype=np.float64)
        v = self._rng.random((n, 2)) * 2.0 - 1.0
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.velocities = (v / norms) * (self.params.max_speed * 0.6)
        # 個体ごとに色を固定割り当て
        self.boid_colors = [self._rand_color_hex() for _ in range(n)]
        self._frame_count = 0
        self._last_fps_stamp = time.perf_counter()
        self._fps_measured = 0.0

    def reset(self) -> None:
        self._init_flock()
        self.refresh()

    # --- 物理更新（壁バウンド） ---
    def _tick(self) -> None:
        if self.paused or self.positions is None or self.velocities is None:
            return

        pos = self.positions
        vel = self.velocities
        n = pos.shape[0]

        for i in range(n):
            p = pos[i]
            v = vel[i]

            delta = pos - p
            dist = np.linalg.norm(delta, axis=1)
            dist[i] = np.inf

            steer = np.zeros(2, dtype=np.float64)

            # Align
            mask_a = dist < self.params.align_radius
            if np.any(mask_a):
                avg_v = vel[mask_a].mean(axis=0)
                nv = np.linalg.norm(avg_v)
                if nv > 0:
                    desired = (avg_v / nv) * self.params.max_speed
                    s = desired - v
                    ns = np.linalg.norm(s)
                    if ns > self.params.max_force:
                        s = (s / ns) * self.params.max_force
                    steer += s

            # Cohere
            mask_c = dist < self.params.cohere_radius
            if np.any(mask_c):
                center = pos[mask_c].mean(axis=0)
                vec = center - p
                nv = np.linalg.norm(vec)
                if nv > 0:
                    desired = (vec / nv) * self.params.max_speed
                    s = desired - v
                    ns = np.linalg.norm(s)
                    if ns > self.params.max_force:
                        s = (s / ns) * self.params.max_force
                    steer += s

            # Separate
            mask_s = dist < self.params.separate_radius
            if np.any(mask_s):
                diffs = p - pos[mask_s]
                d = np.linalg.norm(diffs, axis=1, keepdims=True)
                d[d == 0] = 1.0
                push = (diffs / d).mean(axis=0)
                npv = np.linalg.norm(push)
                if npv > 0:
                    desired = (push / npv) * self.params.max_speed
                    s = desired - v
                    ns = np.linalg.norm(s)
                    if ns > self.params.max_force:
                        s = (s / ns) * self.params.max_force
                    steer += s

            # 速度更新（クランプ）
            v += steer
            nv = np.linalg.norm(v)
            if nv > self.params.max_speed:
                v[:] = (v / nv) * self.params.max_speed

            # 位置更新
            p += v

            # 壁でバウンド（内寸: 0.._tank_w-1, 0.._tank_h-1）
            if p[0] < 0:
                p[0] = 0
                v[0] = abs(v[0]) * self.params.restitution
            elif p[0] > self._tank_w - 1:
                p[0] = self._tank_w - 1
                v[0] = -abs(v[0]) * self.params.restitution

            if p[1] < 0:
                p[1] = 0
                v[1] = abs(v[1]) * self.params.restitution
            elif p[1] > self._tank_h - 1:
                p[1] = self._tank_h - 1
                v[1] = -abs(v[1]) * self.params.restitution

        # FPS計測
        self._frame_count += 1
        now = time.perf_counter()
        if now - self._last_fps_stamp >= 0.5:
            self._fps_measured = self._frame_count / (now - self._last_fps_stamp)
            self._frame_count = 0
            self._last_fps_stamp = now

        self.refresh()

    # --- 描画（ASCII + per-cell color） ---
    def render(self) -> Text:
        W, H = max(self.size.width, 1), max(self.size.height, 1)
        txt = Text()

        # 文字とスタイルの二重バッファ
        char_rows = [bytearray(b" " * W) for _ in range(H)]
        style_rows: list[list[str | None]] = [[None] * W for _ in range(H)]

        def put(y: int, x: int, ch: str, style: str | None = None):
            if 0 <= y < H and 0 <= x < W:
                b = ch.encode("ascii", "replace")[:1]
                char_rows[y][x:x+1] = b
                if style is not None:
                    style_rows[y][x] = style

        # 水槽枠（ASCII）
        x0, y0 = self._off_x, self._off_y
        if self.draw_border:
            x1, y1 = x0 + self._tank_w + 1, y0 + self._tank_h + 1
            border_style = "bright_black"  # 目立ちすぎない枠

            # 角
            put(y0, x0, "+", border_style); put(y0, x1, "+", border_style)
            put(y1, x0, "+", border_style); put(y1, x1, "+", border_style)
            # 上下辺
            for x in range(x0 + 1, x1):
                put(y0, x, "-", border_style); put(y1, x, "-", border_style)
            # 左右辺
            for y in range(y0 + 1, y1):
                put(y, x0, "|", border_style); put(y, x1, "|", border_style)

        # 個体（色付き）
        if self.positions is not None:
            pts = self.positions.astype(int)
            glyphs = (">", "<", "^", "v")
            for idx, ((px, py), v) in enumerate(zip(pts, self.velocities)):
                if self.draw_border:
                    sx = x0 + 1 + px
                    sy = y0 + 1 + py
                else:
                    sx = x0 + px
                    sy = y0 + py
                if 0 <= sx < W and 0 <= sy < H:
                    if abs(v[0]) >= abs(v[1]):
                        ch = glyphs[0] if v[0] >= 0 else glyphs[1]
                    else:
                        ch = glyphs[2] if v[1] < 0 else glyphs[3]

                    color_style = (self.boid_colors[idx] if self.boid_colors is not None else None)
                    put(sy, sx, ch, color_style)

        # バッファ→Text（スタイルのラン長圧縮で高速化）
        for y in range(H):
            line = Text()
            x = 0
            while x < W:
                cur_style = style_rows[y][x]
                start = x
                while x < W and style_rows[y][x] == cur_style:
                    x += 1
                segment = char_rows[y][start:x].decode("ascii", "replace")
                if cur_style is None:
                    line.append(segment)
                else:
                    line.append(segment, style=cur_style)
            txt.append(line)
            txt.append("\n")

        status = (
            f" Boids: {self.params.count} | FPS: {self._fps_measured:4.1f} | "
            f"Tank: {self._tank_w}x{self._tank_h} | "
            f"[space]=pause  r=reset  +/-=count  q=quit "
        )
        txt.append(status)
        return txt

    # --- 外部操作 ---
    def inc_count(self, delta: int) -> None:
        if self.positions is None or self.velocities is None:
            return
        new_n = max(1, self.params.count + delta)
        if new_n == self.params.count:
            return
        self.params.count = new_n

        cur_n = self.positions.shape[0]
        if new_n > cur_n:
            add = new_n - cur_n
            extra_pos = self._rng.random((add, 2)) * np.array([self._tank_w, self._tank_h], dtype=np.float64)
            extra_vel = (self._rng.random((add, 2)) * 2.0 - 1.0)
            norms = np.linalg.norm(extra_vel, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            extra_vel = (extra_vel / norms) * (self.params.max_speed * 0.6)

            self.positions = np.vstack([self.positions, extra_pos])
            self.velocities = np.vstack([self.velocities, extra_vel])
            if self.boid_colors is None:
                self.boid_colors = []
            self.boid_colors.extend(self._rand_color_hex() for _ in range(add))
        else:
            self.positions = self.positions[:new_n]
            self.velocities = self.velocities[:new_n]
            if self.boid_colors is not None:
                self.boid_colors = self.boid_colors[:new_n]
        self.refresh()


class BoidsApp(App):
    CSS = """
    Screen { align: center middle; }
    #root { width: 100%; height: 100%; }
    """

    BINDINGS = [
        ("space", "toggle_pause", "Pause/Resume"),
        ("r", "reset", "Reset"),
        ("+", "more", "More boids"),
        ("-", "fewer", "Fewer boids"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, params: BoidsParams, fps: float, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        self.fps = fps
        self._widget: BoidsWidget | None = None

    def compose(self) -> ComposeResult:
        self._widget = BoidsWidget(self.params)
        self._widget.fps_target = self.fps
        yield Container(self._widget, id="root")

    # Actions
    def action_toggle_pause(self) -> None:
        if self._widget:
            self._widget.paused = not self._widget.paused

    def action_reset(self) -> None:
        if self._widget:
            self._widget.reset()

    def action_more(self) -> None:
        if self._widget:
            self._widget.inc_count(+10)

    def action_fewer(self) -> None:
        if self._widget:
            self._widget.inc_count(-10)

    def action_quit(self) -> None:
        self.exit()


def main():
    parser = argparse.ArgumentParser(description="Boids Simulation on Textual/Rich (terminal)")
    parser.add_argument("--count", type=int, default=120, help="初期Boid数")
    parser.add_argument("--fps", type=float, default=30.0, help="ターゲットFPS")
    parser.add_argument("--align", type=float, default=10.0, help="整列半径（セル）")
    parser.add_argument("--cohere", type=float, default=12.0, help="結合半径（セル）")
    parser.add_argument("--separate", type=float, default=6.0, help="分離半径（セル）")
    parser.add_argument("--max-speed", type=float, default=2.8, help="最大速度（セル/フレーム）")
    parser.add_argument("--max-force", type=float, default=0.08, help="最大操舵（加速度）")
    parser.add_argument("--tank-w", type=int, default=None, help="水槽の幅（セル）")
    parser.add_argument("--tank-h", type=int, default=None, help="水槽の高さ（セル）")
    parser.add_argument("--restitution", type=float, default=1.0, help="壁の反発係数(0.0〜1.0、1.0で完全反射)")
    args = parser.parse_args()

    params = BoidsParams(
        max_speed=args.max_speed,
        max_force=args.max_force,
        align_radius=args.align,
        cohere_radius=args.cohere,
        separate_radius=args.separate,
        count=args.count,
        tank_w=args.tank_w,
        tank_h=args.tank_h,
        restitution=args.restitution,
    )
    app = BoidsApp(params=params, fps=args.fps)
    app.run()


if __name__ == "__main__":
    main()
