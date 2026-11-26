"""Capture real program output and lay it out as a readable terminal transcript."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
COMMANDS = (
    ("怎么续借",),
    ("图书丢了去哪个柜台", "长宁"),
    ("怎么预约靠窗的座位",),
)


def capture(python):
    sections = []
    for query, *additional in COMMANDS:
        arguments = ["assistant.py", "--query", query]
        command = f'python assistant.py --query "{query}"'
        if additional:
            reply = additional[0]
            arguments += ["--reply", reply]
            command += f' --reply "{reply}"'
        run = subprocess.run([python, "-X", "utf8", *arguments], cwd=ROOT,
                             check=True, encoding="utf-8", capture_output=True)
        sections.append(f"$ {command}\n{run.stdout.rstrip()}")
    return "\n\n".join(sections) + "\n"


def draw_transcript(lines, width, height, subtitle):
    from PIL import Image, ImageDraw, ImageFont

    regular = ImageFont.truetype(r"C:\Windows\Fonts\Deng.ttf", 23)
    bold = ImageFont.truetype(r"C:\Windows\Fonts\Dengb.ttf", 26)
    heading = ImageFont.truetype(r"C:\Windows\Fonts\Dengb.ttf", 36)
    image = Image.new("RGB", (width, height), "#111c2b")
    draw = ImageDraw.Draw(image)
    draw.text((46, 31), "上海工程技术大学图书馆借阅助手", font=heading, fill="#f0f6fc")
    draw.text((48, 81), subtitle, font=regular, fill="#a9bfd5")
    for index, line in enumerate(lines):
        light = line.startswith("$")
        color = "#7fe0cb" if light else "#edf3f8"
        if line.lstrip().startswith("https://"):
            color = "#a6c8ef"
        font = bold if light else regular
        if draw.textlength(line, font=font) > width - 95:
            raise ValueError("Resize the demonstration image for the longest line.")
        draw.text((48, 125 + index * 32), line, font=font, fill=color)
    return image


def save_demo_images(transcript):
    lines = transcript.splitlines()
    draw_transcript(lines, 1430, 145 + 32 * len(lines),
                    "终端输出实录 · 续借 / 校区追问 / 范围外").save(DEMO / "演示.png")

    _, clarification, _ = transcript.strip().split("\n\n$ ")
    _, _, output = clarification.partition("\n")
    conversation = output.splitlines()
    reply_line = next(index for index, line in enumerate(conversation) if line.startswith("补充："))
    stages = (conversation[:1], conversation[:reply_line],
              conversation[:reply_line + 1], conversation)
    frames = [draw_transcript(stage, 1120, 145 + 32 * len(conversation), "实际终端输出逐步回放")
              for stage in stages]
    frames[0].save(DEMO / "完整运行回放.gif", save_all=True, append_images=frames[1:],
                   duration=[1500, 2400, 1600, 7500], loop=0)


if __name__ == "__main__":
    transcript = capture(sys.argv[1] if len(sys.argv) > 1 else sys.executable)
    (DEMO / "终端运行记录.txt").write_text(transcript, encoding="utf-8")
    save_demo_images(transcript)
