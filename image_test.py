"""
簡易画像レンダリングテスター（縮小時の見栄え調整付き）

例:
  python test.py --path static/images/idle.png --cols 40
  python test.py --path static/images/idle.png --width 30 --height 25 --unsharp 1.2,180,2 --contrast 1.1 --sharpness 1.2
  python test.py --path static/images/idle.png --cols 50 --algo lanczos --edge

優先レンダラ: rich-pixels
フォールバック: rich.image（存在すれば）
"""

import argparse
from pathlib import Path

from rich.console import Console
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

console = Console()

# レンダラの準備
Pixels = None
try:
    from rich_pixels import Pixels as _Pixels  # type: ignore
    Pixels = _Pixels
except Exception:
    Pixels = None

RichImage = None
try:
    from rich.image import Image as _RichImage  # type: ignore
    RichImage = _RichImage
except Exception:
    RichImage = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="static/images/shoebill_35x33.png", help="画像パス")
    p.add_argument("--cols", type=int, default=0, help="文字カラム数の目安（幅）。0で未指定")
    p.add_argument("--width", type=int, default=0, help="ピクセル幅。0で未指定")
    p.add_argument("--height", type=int, default=0, help="ピクセル高さ。0で未指定（未指定時は縦横比維持）")
    p.add_argument("--scale", type=float, default=0.0, help="倍率。0で未指定")
    p.add_argument("--algo", choices=["nearest","bilinear","bicubic","lanczos"], default="lanczos", help="縮小アルゴリズム")
    p.add_argument("--sharpness", type=float, default=1.0, help="シャープネス係数（1.0=無効）")
    p.add_argument("--contrast", type=float, default=1.0, help="コントラスト係数（1.0=無効）")
    p.add_argument("--color", type=float, default=1.0, help="彩度係数（1.0=無効）")
    p.add_argument("--unsharp", type=str, default="", help="UnsharpMask radius,percent,threshold 例: 1.2,180,2")
    p.add_argument("--edge", action="store_true", help="エッジ強調 (EDGE_ENHANCE) を適用")
    p.add_argument("--bg", default="", help="透過をこの色で塗る（例: #000000 または #00ff00）")
    return p.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.path)
    if not img_path.exists():
        console.print(f"[red]画像が見つかりません: {img_path}")
        return

    img = Image.open(img_path)

    # 透過背景の処理
    if args.bg and img.mode in ("RGBA","LA"):
        bg = Image.new("RGBA", img.size, args.bg)
        bg.paste(img, mask=img.split()[-1])
        img = bg.convert("RGB")
    elif img.mode not in ("RGB","RGBA"):
        img = img.convert("RGB")

    # サイズ計算
    w, h = img.size
    new_w, new_h = w, h

    if args.cols > 0:
        # 文字カラム→ピクセルの目安（端末によって変わるが2倍程度が妥当）
        new_w = max(10, args.cols * 2)
        new_h = int(h * (new_w / w))

    if args.width > 0 and args.height > 0:
        new_w, new_h = args.width, args.height
    elif args.width > 0:
        new_w = args.width
        new_h = int(h * (new_w / w))
    elif args.height > 0:
        new_h = args.height
        new_w = int(w * (new_h / h))

    if args.scale and args.scale > 0:
        new_w = max(1, int(new_w * args.scale))
        new_h = max(1, int(new_h * args.scale))

    if (new_w, new_h) != (w, h):
        resample = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[args.algo]
        img = img.resize((new_w, new_h), resample)

    # 画質調整（小さくした時の視認性向上）
    if args.unsharp:
        try:
            r, p, t = args.unsharp.split(",")
            img = img.filter(ImageFilter.UnsharpMask(radius=float(r), percent=int(p), threshold=int(t)))
        except Exception:
            pass

    if args.edge:
        img = img.filter(ImageFilter.EDGE_ENHANCE)

    if args.sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(args.sharpness)
    if args.contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(args.contrast)
    if args.color != 1.0:
        img = ImageEnhance.Color(img).enhance(args.color)

    console.print(f"[dim]render size: {img.size[0]}x{img.size[1]}[/]")

    # レンダリング
    if Pixels is not None:
        renderable = Pixels.from_image(img)
        console.print(renderable)
        return

    if RichImage is not None:
        # RichImage は幅にカラム数を指定
        cols = args.cols if args.cols > 0 else max(20, min(120, int(new_w / 2)))
        console.print(RichImage.from_pil_image(img, width=cols))
        return

    console.print("[red]どの画像レンダラも利用できません（rich-pixels / rich.image）")


if __name__ == "__main__":
    main()
