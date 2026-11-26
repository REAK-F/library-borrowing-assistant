"""生成当日工作汇报 PDF（给项目作者看，不属于课程报告）。"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FONTS = Path(r"C:\Windows\Fonts")
TARGET = ROOT / "output" / "pdf" / "2026-11-25_工作汇报_图书馆借阅助手.pdf"

DARK = colors.HexColor("#17283a")
TEAL = colors.HexColor("#12635f")
MUTED = colors.HexColor("#476479")
LINE = colors.HexColor("#d9e4e9")
LIGHT = colors.HexColor("#f1f7f8")

pdfmetrics.registerFont(TTFont("Body", str(FONTS / "Deng.ttf")))
pdfmetrics.registerFont(TTFont("Bold", str(FONTS / "Dengb.ttf")))


def styles():
    base = dict(fontName="Body", textColor=DARK, alignment=TA_LEFT, wordWrap="CJK")
    return {
        "title": ParagraphStyle("Title", fontName="Bold", textColor=DARK, fontSize=30,
                                leading=40, spaceAfter=10),
        "meta": ParagraphStyle("Meta", fontName="Body", textColor=MUTED, alignment=TA_LEFT,
                               fontSize=12, leading=20, spaceAfter=18),
        "h": ParagraphStyle("H", fontName="Bold", textColor=TEAL, fontSize=14, leading=21,
                            spaceBefore=16, spaceAfter=8, keepWithNext=True),
        "p": ParagraphStyle("P", **base, fontSize=10.8, leading=17.4, spaceAfter=8,
                            firstLineIndent=21),
        "li": ParagraphStyle("LI", **base, fontSize=10.8, leading=17.4, spaceAfter=6,
                             leftIndent=14, bulletIndent=2),
        "cell": ParagraphStyle("Cell", **base, fontSize=9.6, leading=14.5),
        "th": ParagraphStyle("TH", fontName="Bold", textColor=colors.white, fontSize=9.6,
                             leading=14.5, wordWrap="CJK"),
        "kpi_num": ParagraphStyle("KpiNum", fontName="Bold", textColor=TEAL, fontSize=22,
                                  leading=26, alignment=TA_LEFT),
        "kpi_lbl": ParagraphStyle("KpiLbl", fontName="Body", textColor=MUTED, alignment=TA_LEFT,
                                  fontSize=9.2, leading=13),
        "code": ParagraphStyle("Code", fontName="Body", textColor=colors.HexColor("#0f3d3a"),
                               fontSize=9.4, leading=15, leftIndent=10, spaceAfter=3),
    }


def para(text, style):
    return Paragraph(text, style)


def table(rows, widths, styles_map):
    body = [[para(value, styles_map["th"] if index == 0 else styles_map["cell"]) for value in row]
            for index, row in enumerate(rows)]
    grid = Table(body, colWidths=widths, hAlign="LEFT", repeatRows=1)
    grid.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
    ]))
    return grid


def kpis(items, style):
    numbers = [para(number, style["kpi_num"]) for number, _ in items]
    labels = [para(label, style["kpi_lbl"]) for _, label in items]
    grid = Table([numbers, labels], colWidths=[112] * len(items), hAlign="LEFT")
    grid.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return grid


def decorate(canvas, document):
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(TEAL)
    canvas.rect(0, height - 12, width, 12, fill=True, stroke=False)
    canvas.setStrokeColor(LINE)
    canvas.line(62, 42, width - 62, 42)
    canvas.setFont("Body", 9)
    canvas.setFillColor(MUTED)
    canvas.drawString(62, 28, "机器学习期末项目 / 今日工作报告")
    canvas.drawRightString(width - 62, 28, str(document.page))
    canvas.restoreState()


def build():
    style = styles()
    content = []
    content.append(Spacer(1, 26))
    content.append(para("机器学习期末项目 · 今日工作报告", style["title"]))
    content.append(para("2026 年 11 月 25 日 · 上海工程技术大学图书馆借阅办事助手", style["meta"]))
    content.append(kpis([("9", "页课程报告"), ("3", "张结果图"),
                         ("41", "个交付文件"), ("9", "项测试通过")], style))
    content.append(Spacer(1, 14))

    content.append(para("一、今天处理了什么", style["h"]))
    content.append(para("先核对了原有交付：把项目复制到临时目录重跑评估脚本，生成的指标与逐题 CSV "
                        "和仓库里的文件逐字节一致；按官网逐条比对规则事实，册数、借期、续借、遗失赔偿、"
                        "两校区办理地点与电话都能对上。随后改掉三处会直接影响评分的地方：盲测集被污染、"
                        "机器学习环节只有检索、报告只有表没有图。", style["p"]))
    content.append(para("盲测集。原有 24 条“最终核验”里，“返校再交”“任教”这类措辞只出现在核验题中，"
                        "后来却写进了口语映射和身份别名，等于设计时看过测试。现在把它降级为“内部核验”，"
                        "另按规则写了两轮各 20 条的新问法，写完立即记录文件哈希，只运行一次，"
                        "不根据结果回改规则、映射或阈值。评估脚本每次运行前都会校验哈希。", style["p"]))
    content.append(para("分类器。在原有字符片段特征上训练多项逻辑回归，类目是十条规则加一个"
                        "“范围外”类；训练样本为 50 条标注问法和 78 条规则短语（弱标注）；"
                        "正则强度由三折交叉验证选定。它接进 Agent 的方式是命令行参数 --router classifier，"
                        "默认仍走检索方案。这样“该不该回答”从手调阈值变成了模型判断。", style["p"]))
    content.append(para("报告与工程。重写第三、四、五章，加入盲测协议、两轮结果、错误分析与三张图，"
                        "补两条检索领域的经典文献，正文从 3038 字扩到 4606 字、7 页扩到 9 页。"
                        "同时修掉追问对不上选项时会一直重复同一问题的缺陷（两次后转介），"
                        "表格跨页补表头，构建依赖单独拆出 requirements-report.txt，单元测试从 7 项加到 9 项。",
                        style["p"]))

    content.append(para("二、关键数字", style["h"]))
    content.append(table([
        ["项目", "改动前", "现在"],
        ["报告正文", "3038 字 / 7 页", "4606 字 / 9 页"],
        ["结果图表", "1 张表", "3 张表 + 3 张图"],
        ["评估数据", "内部核验 24 条", "内部 24 条 + 封存盲测 2×20 条"],
        ["主结果", "75%（内部核验）", "70–80%（封存盲测）"],
        ["参考文献", "10 条", "10 条（删 1 补 2）"],
        ["单元测试", "7 项", "9 项"],
        ["交付文件", "20 个", "41 个"],
    ], [118, 168, 190], style))
    content.append(Spacer(1, 6))
    content.append(para("主结果表面上从 75% 变成了 70–80%，是因为换成未参与设计的问法后回落。"
                        "这个差额写进了报告，作为“内部核验偏乐观”的证据，而不是藏起来。", style["p"]))

    content.append(Spacer(1, 10))
    content.append(para("三、删掉与降级的内容", style["h"]))
    content.append(para("按你的确认，删掉了 OpenAI PDF Skill 的复用条目与对应文献：报告附录、参考文献、"
                        "README 三处都已清空，代码里没有任何调用。同一轮还删了 pypdf 与 Poppler 的版本声明，"
                        "因为本机没有安装记录，也没有留下页码核对产物。现在复用清单只剩实际使用的 "
                        "scikit-learn、ReportLab、Pillow 和 matplotlib。", style["p"]))
    content.append(para("“最终核验”这一说法整体降级为“内部核验”，并在报告里写明它的污染来源；"
                        "真正对外的数字改成盲测结果。演示材料仍是终端输出的重绘图片，没有换成真实截图。",
                        style["p"]))

    content.append(para("四、仍然不稳的地方", style["h"]))
    for text in (
        "盲测每轮只有 20 条，分类器 16 条、检索 14 条，两条差距在这么小的样本上不构成显著差异，报告里已注明。",
        "分类器概率没有校准，置信度集中在 0.1 附近，所以弃权没有按概率分段，而是交给了“范围外”类。",
        "手册页码（第 23 至 25 页、§5.4–5.6）没有独立复核：官网手册是嵌入 PDF，本机取不下来。规则内容逐条核过，页码需要你翻一下。",
        "演示图由脚本按真实输出排版，不是屏幕截图；如果老师要求“现场截图”，需要你补一张。",
    ):
        content.append(Paragraph(text, style["li"], bulletText="•"))

    content.append(para("五、接下来你需要做的", style["h"]))
    content.append(para("1. 填封面并改名。姓名、学号现在是空白，文件名还是“待填_待填”。在项目目录运行下面这条，"
                        "会生成按“学号_姓名_项目名称”命名的 PDF。", style["p"]))
    content.append(para("python \"报告材料/build_report.py\" --name \"你的姓名\" --student-id \"你的学号\"",
                        style["code"]))
    content.append(para("2. 翻一遍手册第 23 至 25 页，确认 §5.4、§5.5、§5.6 的编号与页码，报告里共引用了十处。",
                        style["p"]))
    content.append(para("3. 建代码仓库或云盘目录，把链接写进报告封面前的位置和 README；这是提交材料里"
                        "单独列出的第二项，目前还缺。", style["p"]))
    content.append(para("4. 读一遍答辩要点，按上面的五问准备；重点是能说清为什么用字符片段、阈值怎么选、"
                        "为什么关键词不输 TF-IDF、盲测和内部核验的区别。", style["p"]))
    content.append(para("5. 提交前在项目目录重跑这三条，确认结果没变：", style["p"]))
    content.append(para("python -m unittest discover -s tests  ·  python evaluate.py  ·  "
                        "python evaluate_blind.py --blind data/blind_cases_v2.csv "
                        "--manifest data/blind_manifest_v2.json --tag blind_v2", style["code"]))
    content.append(para("6. 报告第七页的 AI 使用说明按事实写成“选题与验收由本人完成，代码与文字在 AI 辅助下完成”。"
                        "如果你想加上自己动手的部分，告诉我具体是哪个文件或哪一段，我按真事补，不要写没做过的。",
                        style["p"]))

    document = SimpleDocTemplate(str(TARGET), pagesize=A4, leftMargin=61, rightMargin=61,
                                 topMargin=48, bottomMargin=54,
                                 title="机器学习期末项目今日工作报告", author="")
    document.build(content, onFirstPage=decorate, onLaterPages=decorate)
    print(TARGET)


if __name__ == "__main__":
    build()
