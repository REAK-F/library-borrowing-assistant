# 上海工程技术大学图书馆借阅办事助手

这是《机器学习》课程的本地 Agent 原型。输入中文问题后，程序检索本校纸质图书借阅规则，必要时询问读者身份或校区，再给出办理步骤和可核对的官网出处。

规则路由有两套实现：默认用 TF-IDF 余弦相似度检索，`--router classifier` 改用字符 n-gram 逻辑回归分类器。两者共用同一条流程（输入检查、范围过滤、追问、出处渲染），区别只在"问题对应哪条规则"这一步。

## 在 VS Code 里运行

1. 在 VS Code 里选择"文件 → 打开文件夹"，打开本项目文件夹。
2. 打开"终端 → 新建终端"。建议使用 Python 3.11 或更新版本。
3. 第一次安装依赖，然后运行助手。

```powershell
python --version
python -m pip install -r requirements.txt
python assistant.py
```

输入 `怎么续借` 试用。输入 `图书丢了去哪个柜台`，再按提示回复 `长宁`。追问时输入 `重来` 可取消当前问题，输入 `q` 退出。连续两次补充都没有对上选项时，程序改为转介。安装依赖后，程序可离线运行。

也可直接运行一条问题，便于截图和复现。

```powershell
python assistant.py --query "怎么续借"
python assistant.py --query "图书丢了去哪个柜台" --reply "长宁"
python assistant.py --query "怎么预约靠窗的座位"
python assistant.py --router classifier --query "怎么续借"
```

## 重跑实验

```powershell
python -m unittest discover -s tests -v
python evaluate.py
python evaluate_blind.py
python evaluate_blind.py --blind data/blind_cases_v2.csv --manifest data/blind_manifest_v2.json --tag blind_v2
```

- `data/rules.json`：10 条规则、官方链接、出处页码及追问选项。以学校《读者手册（2026 秋季版）》为主要来源，按手册印刷页码标注。
- `data/cases.csv`：76 条依据官方规则构造并标注的测试问法。16 条用于首轮校准，36 条用于首轮测试及第二版开发，另留 24 条用于内部核验。
- `data/blind_cases.csv`、`data/blind_cases_v2.csv`：两轮封存盲测，各 20 条，`data/blind_manifest*.json` 记录文件哈希与构造说明。第二轮的规则、口语映射与阈值在第一轮之后未改动。
- `results/metrics.json`：阈值选择、排序命中数和流程成功数。
- `results/first_test.csv`、`results/initial_on_audit.csv`、`results/predictions.csv`：内部核验的逐题输出。末列 `correct` 为整条业务流程的判定。
- `results/keyword_workflow.csv`、`results/without_threshold.csv`：等权关键词端到端基线和去阈值消融的逐题结果。
- `results/blind_v2_predictions_*.csv`、`results/blind_v2_metrics.json`：封存盲测上关键词、TF-IDF、分类器三种方案的逐题结果与汇总。
- `demo/演示.png`、`demo/完整运行回放.gif` 与 `demo/终端运行记录.txt`：由实际终端输出生成，覆盖输入问题、追问、补充和结果。

## 方法

基线用二值字符片段余弦相似度。主方案使用相同的 2 字符和 3 字符片段，用 TF-IDF 加权后计算余弦相似度，并加入少量口语改写和服务范围检查。分类器在同一套片段特征上训练多项逻辑回归，类目为 10 条规则加 1 个"范围外"类；训练样本为开发集与测试集的 50 条标注问法，加上规则表中 78 条检索短语作为弱标注，正则强度由 3 折交叉验证选择。对无法确定的结果，程序给出核实入口。分数只是文本相似度或类别概率。

试用和评估环境为 Python 3.14.6、scikit-learn 1.9.0。scikit-learn 使用 BSD-3-Clause 许可。项目资料库为对官方公开事实和操作步骤的项目化整理，保留出处，原网页及手册版权归权利人。没有使用真实借阅记录或账户信息。本项目为课程练习，业务规则以图书馆官方说明为准。

## 复用来源与验证

| 组件 | 版本或来源 | 许可 | 用途 |
|---|---|---|---|
| scikit-learn | [1.9.0](https://pypi.org/project/scikit-learn/1.9.0/) | BSD-3-Clause | 二值与 TF-IDF 向量化、余弦相似度、逻辑回归 |
| ReportLab | [4.4.9](https://pypi.org/project/reportlab/4.4.9/) | BSD | 生成报告 |
| Pillow | [12.3.0](https://pypi.org/project/Pillow/12.3.0/) | MIT-CMU | 将真实的终端输出制作成演示图及回放 |
| matplotlib | [3.11.0](https://pypi.org/project/matplotlib/3.11.0/) | PSF/BSD 兼容 | 绘制报告中的结果图 |
| 图书馆资料 | [手册及规章](https://lib.sues.edu.cn/94/6f/c14163a300143/page.htm)，2026-11-25 核对 | 原件版权归校方 | 归纳规则与标示出处 |

项目改造了中文片段切分、口语映射和检索后的追问流程。测试检查 IDF 权重、未知词零相似度、分类器路由、盲测集哈希，以及模拟资源缺失时的提示。可用 `--data 路径` 指定另一份规则 JSON 做本地核验。

## 生成式 AI 使用

项目由本人选题，确定需求、范围与验收标准，并对每轮产出做验收。中文片段特征、TF-IDF 检索与追问决策、逻辑回归路由、评估脚本及报告文字在 AI 辅助下完成，已标在 `assistant.py`、`classifier.py` 文件头。测试样例与盲测问法由 AI 按规则另写，实验数值由脚本实际运行产生。

## 报告封面

PDF 单独提供，封面的姓名和学号已留白，默认按个人项目排版。可在 VS Code 编辑 `报告材料/报告正文.md`。在 Windows 的项目目录运行以下命令，可以生成带本人信息且已按要求命名的 PDF。

```powershell
python -m pip install -r requirements-report.txt
python "报告材料/build_report.py" --name "你的姓名" --student-id "你的学号"
```

生成文件位于 `output/pdf/`。结果图可用 `python "报告材料/make_figures.py"` 重新绘制。两人组队时，用 `--contribution "组员姓名及各自分工"` 更新封面。提交前用 VS Code 跑一次演示，再按课程要求将报告 PDF、可访问的代码链接及演示材料提交。
