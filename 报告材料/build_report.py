"""Render the editable course report source with the measured experiment results."""

import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TITLE = "上海工程技术大学图书馆借阅办事助手"
DARK = colors.HexColor("#17283a")
TEAL = colors.HexColor("#12635f")
MUTED = colors.HexColor("#476479")
LINE = colors.HexColor("#d9e4e9")
LIGHT = colors.HexColor("#f1f7f8")


def load_fonts(directory):
    for label, filename in (("CourseBody", "Deng.ttf"), ("CourseBold", "Dengb.ttf")):
        pdfmetrics.registerFont(TTFont(label, str(directory / filename)))


def build_styles():
    base = dict(fontName="CourseBody", textColor=DARK, alignment=TA_LEFT, wordWrap="CJK")
    return {
        "body": ParagraphStyle("Body", **base, fontSize=11.1, leading=17.8,
                               spaceAfter=9, firstLineIndent=22),
        "heading": ParagraphStyle("Heading", fontName="CourseBold", textColor=DARK,
                                  fontSize=20, leading=28, spaceAfter=16, keepWithNext=True),
        "sub": ParagraphStyle("Sub", fontName="CourseBold", textColor=TEAL, fontSize=13,
                              leading=19, spaceBefore=4, spaceAfter=7, keepWithNext=True),
        "small": ParagraphStyle("Small", **base, fontSize=10, leading=15, spaceAfter=5),
        "table": ParagraphStyle("Table", **base, fontSize=9.5, leading=14),
        "th": ParagraphStyle("TableHeading", fontName="CourseBold", textColor=colors.white,
                             fontSize=9.5, leading=14, wordWrap="CJK"),
        "cover_tag": ParagraphStyle("CoverTag", fontName="CourseBold", textColor=TEAL,
                                    fontSize=16, leading=26, spaceAfter=22),
        "cover_title": ParagraphStyle("CoverTitle", fontName="CourseBold", textColor=DARK,
                                      fontSize=35, leading=48, spaceAfter=17),
        "cover_sub": ParagraphStyle("CoverSub", fontName="CourseBody", textColor=MUTED,
                                    fontSize=16, leading=28, spaceAfter=54),
        "caption": ParagraphStyle("Caption", fontName="CourseBody", fontSize=9.2, leading=14,
                                  textColor=MUTED, alignment=TA_CENTER,
                                  spaceBefore=5, spaceAfter=12),
    }


def paragraph(text, style):
    return Paragraph(escape(text), style)


def box(rows, widths):
    table = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
    ]))
    return table


def ratio(hit, total):
    return f"{hit}/{total}（{hit / total:.0%}）"


def result_table(metrics, styles):
    rows = [[paragraph(item, styles["th"]) for item in (
        "试验与方案", "关键词首位", "TF-IDF 首位", "流程成功", "范围外转介", "追问成功",
    )]]
    for label, key in (
        ("首轮 · 初版", "first_test"),
        ("最终核验 · 初版", "initial_on_audit"),
        ("最终 · TF-IDF", "audit"),
        ("最终 · 关键词", "audit_keyword_workflow"),
        ("最终 · TF-IDF 去阈值", "audit_without_threshold"),
    ):
        row = metrics[key]
        values = [
            label,
            f"{row['keyword_top1']}/{row['answerable']}",
            f"{row['tfidf_top1']}/{row['answerable']}",
            ratio(row["workflow_correct"], row["cases"]),
            f"{row['refer_correct']}/{row['refer_cases']}",
            f"{row['followup_correct']}/{row['followup_cases']}",
        ]
        rows.append([paragraph(value, styles["table"]) for value in values])
    return box(rows, [106, 63, 69, 85, 75, 63])


def load_rows(name):
    import csv
    with (ROOT / "results" / name).open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def blind_table(styles):
    arms = (
        ("等权关键词", "blind_v2_predictions_keyword.csv"),
        ("TF-IDF 检索", "blind_v2_predictions_tfidf.csv"),
        ("逻辑回归分类器", "blind_v2_predictions_classifier.csv"),
    )
    rows = [[paragraph(item, styles["th"]) for item in
             ("方案", "正确", "安全转介", "错答或误追问", "流程成功")]]
    for label, name in arms:
        data = load_rows(name)
        correct = sum(row["correct"] == "1" for row in data)
        referred = sum(row["correct"] != "1" and row["final_action"] == "refer" for row in data)
        wrong = len(data) - correct - referred
        total = len(data)
        rows.append([paragraph(value, styles["table"]) for value in (
            label, str(correct), str(referred), str(wrong),
            ratio(correct, total))])
    return box(rows, [122, 58, 78, 104, 99])


def figure(name, caption, styles, width=461):
    from PIL import Image as Probe

    path = HERE / name
    with Probe.open(path) as probe:
        pixel_width, pixel_height = probe.size
    return [
        Image(str(path), width=width, height=width * pixel_height / pixel_width),
        paragraph(caption, styles["caption"]),
    ]


def small_panel(title, items, styles):
    rows = [[paragraph(title, styles["th"])]]
    rows.extend([paragraph(item, styles["table"])] for item in items)
    return box(rows, [461])


def cover(args, styles):
    items = [
        Spacer(1, 108),
        paragraph("上海工程技术大学  ·  机器学习期末项目", styles["cover_tag"]),
        Paragraph("上海工程技术大学<br/>图书馆借阅办事助手", styles["cover_title"]),
        paragraph("基于 TF-IDF 检索与逻辑回归路由的办事指引", styles["cover_sub"]),
    ]
    data = [
        ("课程", "机器学习"),
        ("姓名", args.name or "____________________________"),
        ("学号", args.student_id or "____________________________"),
        ("组员及分工", args.contribution),
        ("日期", args.date),
    ]
    table = Table([[paragraph(key, styles["small"]), paragraph(value, styles["small"])]
                   for key, value in data], colWidths=[100, 360], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return items + [table, PageBreak()]


def render_body(content, metrics, styles):
    panels = {
        "[FLOW]": ("Agent 运行流程", (
            "中文问题 → 范围检查 → 规则路由（TF-IDF 检索或逻辑回归分类器）→ 匹配判断",
            "证据充足 → 直接给清单或追问身份／校区。证据不足 → 指向官方核实入口。",
        )),
        "[DEMO]": ("终端回放节选 · 见随附完整记录", (
            "输入　图书丢了去哪个柜台",
            "追问　你要在哪个校区处理？",
            "补充　长宁",
            "结果　产教融合楼 329 室服务台。来源为读者手册第 25 页和官网联系页面。",
        )),
        "[COMMANDS]": ("在项目目录运行", (
            "1  python -m pip install -r requirements.txt",
            "2  python assistant.py   （--router classifier 切换路由）",
            "3  python -m unittest discover -s tests -v",
            "4  python evaluate.py",
            "5  python evaluate_blind.py  （校验哈希后跑封存盲测）",
        )),
    }
    items = []
    for block in (part.strip() for part in content.strip().split("\n\n")):
        if not block:
            continue
        if block == "[PAGE]":
            items.append(PageBreak())
        elif block == "[RESULTS]":
            items.extend([result_table(metrics, styles), Spacer(1, 12)])
        elif block == "[BLIND]":
            items.append(KeepTogether([blind_table(styles), Spacer(1, 12)]))
        elif block == "[FIG1]":
            items.extend(figure("图1_盲测结果构成.png",
                                "图 1  封存盲测（20 条）上三种方案的结果构成", styles))
            items.append(Spacer(1, 6))
        elif block == "[FIG2]":
            items.extend(figure("图2_核验与盲测对比.png",
                                "图 2  内部核验与封存盲测的流程成功率对比", styles))
            items.append(Spacer(1, 6))
        elif block == "[FIG3]":
            items.extend(figure("图3_混淆矩阵.png",
                                "图 3  分类器在开发集与测试集上的交叉验证混淆矩阵",
                                styles, width=352))
            items.append(Spacer(1, 6))
        elif block in panels:
            heading, texts = panels[block]
            items.extend([small_panel(heading, texts, styles), Spacer(1, 11)])
        elif block.startswith("@ref|"):
            for line in block.splitlines():
                _, label, url = line.split("|", 2)
                link = f'<link href="{escape(url)}" color="#12635f">{escape(label)}</link>'
                items.append(Paragraph(link, styles["small"]))
            items.append(Spacer(1, 6))
        elif block.startswith("## "):
            items.append(paragraph(block[3:], styles["sub"]))
        elif block.startswith("# "):
            items.append(paragraph(block[2:], styles["heading"]))
        else:
            items.append(paragraph(block, styles["body"]))
    return items


def decorate(canvas, document):
    width, height = A4
    canvas.saveState()
    if document.page == 1:
        canvas.setFillColor(TEAL)
        canvas.rect(0, height - 16, width, 16, fill=True, stroke=False)
    else:
        canvas.setStrokeColor(LINE)
        canvas.line(62, height - 38, width - 62, height - 38)
        canvas.line(62, 42, width - 62, 42)
        canvas.setFont("CourseBody", 9)
        canvas.setFillColor(MUTED)
        canvas.drawString(62, height - 31, "上海工程技术大学  /  机器学习期末项目")
        canvas.drawString(62, 27, "图书馆借阅办事助手")
        canvas.drawRightString(width - 62, 27, str(document.page - 1))
    canvas.restoreState()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="")
    parser.add_argument("--student-id", default="")
    parser.add_argument("--contribution", default="个人项目")
    parser.add_argument("--date", default=datetime.now().strftime("%Y 年 %m 月 %d 日").replace(" 0", " "))
    parser.add_argument("--fonts", type=Path, default=Path(r"C:\Windows\Fonts"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    load_fonts(args.fonts)
    styles = build_styles()
    content = (HERE / "报告正文.md").read_text(encoding="utf-8")
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
    filename = f"{args.student_id or '待填'}_{args.name or '待填'}_图书馆借阅办事助手.pdf"
    target = args.out or ROOT / "output" / "pdf" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(target), pagesize=A4, leftMargin=61, rightMargin=61,
        topMargin=61, bottomMargin=54, title=TITLE, author=args.name,
    )
    document.build(cover(args, styles) + render_body(content, metrics, styles),
                   onFirstPage=decorate, onLaterPages=decorate)
    print(target)


if __name__ == "__main__":
    main()
