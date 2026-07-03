from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CASES = {
    "japanese/document": ["第3四半期の利益率は42%に改善", "燃料噴射マップを更新しました"],
    "code/editor": ["def retry_request(timeout=3):", "    return client.get('/api/status')"],
    "powerpoint/slide": ["PROJECT PHOENIX", "Q3 Gross Margin 42%", "Next: calibration review"],
    "dialog/error": ["通信エラー", "ECUとの通信がタイムアウトしました", "再試行    キャンセル"],
    "table/calibration": ["RPM      1000    2000    3000", "Fuel     1.00    1.15    1.23"],
}


def main() -> None:
    output = Path("work/ocr-smoke-dataset")
    japanese_font = ImageFont.truetype(r"C:\Windows\Fonts\meiryo.ttc", 42)
    code_font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 36)
    for relative, lines in CASES.items():
        category, name = relative.split("/")
        folder = output / category
        folder.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (1280, 720), "#f8f8f6")
        draw = ImageDraw.Draw(image)
        for index, line in enumerate(lines):
            font = code_font if category in {"code", "table"} else japanese_font
            draw.text((55, 70 + index * 85), line, fill="#101418", font=font)
        image.save(folder / f"{name}.png")
        (folder / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
