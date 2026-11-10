# Go2.0 中文 / English 网页

## “强化学习模型（最新）”到底加载哪个

该选项的内部 ID 是 `rl`，来源固定为 Hugging Face 模型仓库 `Jycccc111/Go_RL_policy` 根目录 `last.pth`，不按最大 epoch 编号挑选文件。`AI.load_model()` 在创建 AI 时下载该别名所指向的权重并保存在内存中。

2026-11-09 核查结果：

- 仓库提交：`d2720ad27c3b379260484818c5a4118420c3f89b`
- 文件 SHA-256：`770dfc614383f168451b1423cabc6921a28ac1d22881c8b316323317ab992b85`
- 内容与 `models/epoch-000004-770dfc614383f168451b1423cabc6921a28ac1d22881c8b316323317ab992b85.pth` 相同，即 Epoch 4，内容编号 `770dfc61`。
- [核查时的 last.pth](https://huggingface.co/Jycccc111/Go_RL_policy/blob/d2720ad27c3b379260484818c5a4118420c3f89b/last.pth)

仓库以后上传时该映射会改变。已经加载进 APP 的 AI 不会自动换权重；刷新模型列表只更新列表，不替换正在使用的 AI。重新创建 AI 对局或重启 APP 后重新使用该模型，才加载当时的最新别名。这个核查结果描述仓库文件，不代表此前已加载在用户 APP 内存中的模型一定相同。

## 使用

本地桌面项目已修改。重启 `app.py` 并刷新浏览器：

```sh
cd /Users/jiangyuncong/Desktop/Go2.0
python3 app.py
```

页面顶部的语言下拉框可选择中文或 English；也可使用同一 APP 地址加参数：`/?lang=zh-CN` 和 `/?lang=en`。浏览器记住语言偏好。

语言切换通过读取页面完成，沿用服务器现有棋局。棋盘、黑白双方待选模型、贴目和 SGF 回放位置保留；已计算的 SGF 胜率曲线在本标签页切换时复用。临时推荐标记和点目覆盖层不自动重新计算，需要时再次点击相应功能。AI 对局可继续；有请求正在执行时会暂时禁用语言选择。

按钮、对局提示、模型列表、推荐位置、点目、胜率预测、SGF 回放和图表文本均提供两种语言。模型 ID、棋盘坐标和请求参数保持原值，不翻译为另一套规则。已知服务器错误会翻译；未知中文异常在英文界面显示通用错误，并把原文保留在浏览器控制台，供排查。

`Go2.0-web-bilingual.zip` 解压覆盖其他项目副本后也可使用。必须一起更新 `app.py`、`web_i18n.py`、HTML、`static/i18n.js`、`static/translations.json` 与 `static/script.js`，不要只复制 HTML。没有推送 GitHub，也没有上传模型或启动训练。

## 修改与验证

- `templates/show.html`：中文、英文共用同一布局；增加语言选择和 latest 来源说明。
- `web_i18n.py`、`static/translations.json`：共享翻译文本，支持服务端首次渲染。
- `static/i18n.js`：文本、参数和动态模型名称翻译，以及语言偏好和切换暂存。
- `static/script.js`：运行时消息与图表翻译；恢复语言切换前的界面设置。
- `app.py`：语言参数与偏好读取；新增只读 `/sgf/state`，不推进回放。
- CSS：语言选择控件及长文本换行。

81 项 Python 测试通过，另有无第三方依赖的 JavaScript 检查。实际浏览器验证了英文、中文、黑白不同模型选择、贴目以及落子后的语言切换；没有为了测试下载大模型或执行 AI 训练。SGF 位置保持由自动测试验证。

```sh
python3 -m unittest discover -s tests -v
# 可选，仅用于开发验证，不是 APP 运行依赖：
node tests/test_i18n.js
```

## English

“Reinforcement learning (latest)” loads `Jycccc111/Go_RL_policy/last.pth` when the AI is constructed. It does not choose the largest epoch number. As checked on 2026-11-09 at commit `d2720ad27c3b379260484818c5a4118420c3f89b`, that file matches Epoch 4 with content identity `770dfc61`. Future uploads may change this mapping. An AI already in memory keeps its loaded weights; refreshing the catalog does not hot-swap them. Create a new AI game or restart the app to load the current alias.

Restart `app.py`, refresh the browser and use the top language selector. Direct links are `/?lang=zh-CN` and `/?lang=en`; the browser remembers the preference. Both versions share game logic and model IDs. Switching languages preserves the existing board, pending model choices, komi and SGF replay position. The current tab also retains the computed SGF curve. Temporary recommendation/scoring overlays may be requested again. Language selection is disabled while a request is pending.

Translations cover controls, model labels, game status, scoring, predictions, SGF replay, diagnostics and chart text. Unknown Chinese server errors get a generic English message while preserving details in the browser console. Deploy all files together, including `web_i18n.py` and the static translation catalog. No GitHub push, model upload or training was performed.

Eighty-one Python tests and the JavaScript checks passed. Browser verification covered both languages, independent model choices, komi and board preservation. SGF replay preservation is covered by automated tests. The JavaScript test requires Node only for development, not for running the app.

## 缓慢播放 AI 对弈 / Slow AI self-play

棋盘下方勾选“缓慢播放 AI 对弈”，选择 0.5、1、2 或 3 秒（默认 1 秒）；默认不勾选，保持原速度。黑白双方均为 AI 时，显示上一手后按所选间隔等待，再请求下一手。AI 推理时间另计；人类与 AI 对局不添加等待。对弈中可以调整，下一次等待读取新设置；当前已开始的等待最多保留原来的 3 秒。终局、投降或请求出错后不再请求下一手。偏好保存在当前浏览器，切换中英文时也保留。

默认关闭。此功能只调节网页观看速度，不改变模型选点、训练或原规则。

Enable “Slow AI self-play” below the board and choose 0.5, 1, 2 or 3 seconds (default interval: 1 second). It is initially off to retain the existing speed. The interval is a wait after the previous position is displayed and before requesting the next AI move; inference time is additional. It only affects AI-vs-AI games. Changes apply to subsequent waits; an already started wait retains its original duration. No extra move is requested after termination, resignation or an error. Preferences persist in the browser and across language changes.

The playback helper does not change model decisions, training or game rules. Python tests pass, along with JavaScript tests covering slow/fast modes, live interval changes, terminal stopping and errors. A local browser preview uses a simulated AI to verify pacing without downloading weights or training.
