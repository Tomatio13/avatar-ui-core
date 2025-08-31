
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
terminal_text_spectrum.py
文字列を入力として、喋っている風のスペクトラムを端末に描画します（オーディオ不要）。
各文字に応じて周波数帯域の強調やエネルギーを擬似生成します。

使い方:
  echo "こんにちは！今日はいい天気だね。" | python3 terminal_text_spectrum.py
  # 対話的
  python3 terminal_text_spectrum.py --interactive

主なオプション:
  --fps 60                 # 描画FPS
  --cps 12                 # 1秒あたりの文字消費速度（char per second）
  --seed 0                 # 擬似乱数シード（見た目の揺れ）
  --mouth-pipe /tmp/mouth  # このFIFOに各フレームの口開度[0..1]を書き出す（任意）
  --loop                   # 入力文字列をループ再生

補足:
  - 入力はUTF-8想定。改行は文間の小休止。
  - 端末幅に応じて棒グラフ本数を調整。
  - 24bitカラー対応端末ならグラデ表示。
  - 'q' で終了。
"""
import sys, os, time, math, argparse, curses, random
import numpy as np

RESET = "\033[0m"
def color_from_level(level: float) -> str:
    # 青→緑→黄→赤
    r = int(min(255, max(0, 510 * (level - 0.5))))
    g = int(min(255, max(0, 510 * (0.5 - abs(level - 0.5)))) * 2/3 + int(255 * min(level*1.3,1.0))*1/3)
    b = int(min(255, max(0, 510 * (0.5 - level))))
    return f"\033[38;2;{r};{g};{b}m"

# 文字→発音カテゴリ（超簡易）
VOWELS_JA = set(list("あいうえおアイウエオぁぃぅぇぉゃゅょァィゥェォャュョー"))
NASALS_JA = set(list("んン"))
PAUSE = set(list("。、．，・…！？!?。,:;；、\n"))
HARD_CONS = set(list("かきくけこさしすせそたちつてとカキクケコサシスセソタチツテトぱぴぷぺぽバビブベボパピプペポ"))
SOFT_CONS = set(list("なにぬねのまみむめもらりるれろはひふへほやゆよわをがぎぐげござじずぜぞだぢづでどゃゅょゎゐゑ"))
SPACE = set(list(" 　\t"))

def char_class(ch: str) -> str:
    if ch in SPACE: return "space"
    if ch in PAUSE: return "pause"
    if ch in VOWELS_JA: return "vowel"
    if ch in NASALS_JA: return "nasal"
    if ch in HARD_CONS: return "hard"
    if ch in SOFT_CONS: return "soft"
    # ASCII fallback
    if ch.lower() in "aeiou": return "vowel"
    if ch in ".!?," : return "pause"
    return "soft"

# 周波数プロファイル（擬似）: 各カテゴリごとに帯域の重み分布（0..1）
def profile_for(cls: str, n_bars: int) -> np.ndarray:
    x = np.linspace(0,1,n_bars, dtype=np.float32)
    if cls == "vowel":
        # 中高域強め（人声のフォルマントっぽさ）
        prof = np.exp(-((x-0.6)**2)/(2*0.1**2)) + 0.4*np.exp(-((x-0.3)**2)/(2*0.15**2))
    elif cls == "nasal":
        prof = np.exp(-((x-0.25)**2)/(2*0.07**2))
    elif cls == "hard":
        prof = 0.7*np.exp(-((x-0.15)**2)/(2*0.08**2)) + 0.5*np.exp(-((x-0.55)**2)/(2*0.12**2))
    elif cls == "soft":
        prof = 0.6*np.exp(-((x-0.45)**2)/(2*0.12**2))
    elif cls == "pause":
        prof = np.zeros_like(x)
    else:  # space
        prof = np.zeros_like(x)
    return prof / (prof.max() + 1e-6)

def main(stdscr, args):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)

    # 入力テキスト
    if args.interactive:
        print("Enter text. Finish with Ctrl+D (or Ctrl+Z on Windows).", file=sys.stderr)
    text = sys.stdin.read() if not args.interactive else sys.stdin.read()
    if not text:
        text = "こんにちは！テキストから擬似的におしゃべりしています。"

    # 前処理
    chars = list(text)
    idx = 0
    start_time = time.time()
    last_frame = start_time
    # 1文字あたりの基本長（秒）
    base_dur = 1.0 / max(1, args.cps)
    ema = None

    mouth_pipe = None
    if args.mouth_pipe:
        # FIFOが存在しなければ作る（通常は事前作成推奨: mkfifo /tmp/mouth）
        if not os.path.exists(args.mouth_pipe):
            try:
                os.mkfifo(args.mouth_pipe)
            except FileExistsError:
                pass
        # 非ブロッキングで開く
        mouth_pipe = os.open(args.mouth_pipe, os.O_WRONLY | os.O_NONBLOCK) if os.path.exists(args.mouth_pipe) else None

    random.seed(args.seed)
    jitter_phase = random.random()*1000.0

    try:
        while True:
            now = time.time()
            if now - last_frame < 1.0/args.fps:
                time.sleep(max(0, 1.0/args.fps - (now - last_frame)))
            last_frame = time.time()

            h, w = stdscr.getmaxyx()
            n_bars = max(10, w - 10)
            # 現在の文字と残り時間を計算
            ch = chars[idx]
            cls = char_class(ch)

            # 文字クラスごとにエネルギーと持続の係数
            if cls == "vowel":
                dur = base_dur * 1.2
                energy = 0.8
            elif cls == "nasal":
                dur = base_dur * 1.0
                energy = 0.6
            elif cls == "hard":
                dur = base_dur * 0.9
                energy = 0.7
            elif cls == "soft":
                dur = base_dur * 1.0
                energy = 0.65
            elif cls == "pause":
                dur = base_dur * 1.8
                energy = 0.05
            else: # space
                dur = base_dur * 0.8
                energy = 0.0

            # ADSR風エンベロープ(簡易)
            t = (time.time() - start_time) / max(1e-6, dur)
            if t >= 1.0:
                # 次の文字
                start_time = time.time()
                idx += 1
                if idx >= len(chars):
                    if args.loop:
                        idx = 0
                    else:
                        break
                continue
            # Attack/Decay/Sustain/Releaseの擬似（A=0.08, D=0.25, S=0.7, R=0.15）
            A, D, S, R = 0.08, 0.25, 0.7, 0.15
            if t < A:
                env = (t/A)
            elif t < A + D:
                env = 1.0 - (1.0 - S) * ((t-A)/D)
            elif t < 1.0 - R:
                env = S
            else:
                env = S * (1.0 - (t - (1.0 - R))/R)

            # 口開度 [0..1]
            mouth_level = float(max(0.0, min(1.0, energy * env)))

            # 帯域プロファイル
            prof = profile_for(cls, n_bars)
            # 揺らぎ（ホワイトノイズ→平滑）
            jitter = 0.15 * np.sin((jitter_phase + np.arange(n_bars))*0.21) + 0.1*np.random.randn(n_bars)
            jitter_phase += 0.03
            jitter = np.clip(jitter, -0.2, 0.3)

            bars = np.clip(mouth_level * (0.6*prof + 0.4*(prof + jitter)), 0.0, 1.0)

            # 平滑化
            if ema is None or len(ema) != n_bars:
                ema = bars
            else:
                ema = args.smoothing * ema + (1.0 - args.smoothing) * bars

            stdscr.erase()

            # ヘッダ
            status = f" Text Spectrum | char: {repr(ch)} class:{cls} | cps:{args.cps} fps:{args.fps} (q to quit) "
            try:
                stdscr.addnstr(0, max(0, (w - len(status)) // 2), status[:w-1], w-1)
            except curses.error:
                pass

            # 描画
            for x_pos, val in enumerate(ema[:n_bars]):
                bar_h = int(val * (h - 3))
                for y in range(bar_h):
                    level = y / max(1, h - 3)
                    color = color_from_level(level)
                    try:
                        stdscr.addstr(h - 2 - y, x_pos, color + "█" + RESET)
                    except curses.error:
                        pass

            # 現在の単語/先読み提示（任意）
            try:
                preview = "".join(chars[idx: idx+12]).replace("\n", " ")
                stdscr.addnstr(h-1, 1, f"> {preview}", w-2)
            except curses.error:
                pass

            stdscr.refresh()

            # 口開度をFIFOに書く（任意）
            if mouth_pipe is not None:
                try:
                    os.write(mouth_pipe, f"{mouth_level:.3f}\n".encode("utf-8"))
                except OSError:
                    pass

            # 入力
            c = stdscr.getch()
            if c in (ord('q'), ord('Q')):
                break

    finally:
        if mouth_pipe is not None:
            try: os.close(mouth_pipe)
            except: pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--cps", type=float, default=12.0, help="characters per second")
    p.add_argument("--smoothing", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--mouth-pipe", type=str, default=None, help="FIFO path for mouth level [0..1] per frame")
    p.add_argument("--loop", action="store_true")
    args = p.parse_args()

    curses.wrapper(main, args)
