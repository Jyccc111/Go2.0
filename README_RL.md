# Go2.0 REINFORCE — Epoch 训练 / Epoch training

## 中文

### 单位与入口

入口：`python3 -m rl.train`；`python3 Reinforcement_learning.py` 使用相同参数。
导入不会训练，不依赖 `app.py` 或 `AI` 实例。安装与检查：

```sh
python3 -m pip install -r requirements-rl.txt
python3 -m unittest discover -s tests -v
python3 -m rl.train --help
```

**一个默认 epoch = 4096 次有效策略优化器更新，不是 4096 盘棋。**
`--epochs 1` 表示再完成一个 epoch；`--updates-per-epoch` 默认 4096。
一盘有效棋局最多调用一次 `optimizer.step()`。跳过局、不含学习方动作的棋局不更新，
也不计入 4096 次。完成一轮可能需要超过 4096 盘。
有确定标签但学习方尚未行动便结束的棋局仍可提供一个 value 样本，所以样本数也可能超过 4096。
`--episodes`、`--snapshot-every` 已删除，避免把棋局、更新和 epoch 混用。
可用 `--max-games-per-epoch` 限制一轮尝试局数；达到上限而更新不足时会报错，不能冒充完成。

仅做短检查时可显式使用 `--updates-per-epoch 2`；这会改变该次运行的 epoch 大小。
正常训练示例（需你主动执行）：

```sh
python3 -m rl.train --output runs/rl-001 --epochs 1 --updates-per-epoch 4096 --device cuda
```

`--output` 必须是新目录；设备默认 CPU，可选 `cuda` 或 `mps`。
常用参数：`--max-moves 800 --komi 7.5 --lr 0.00001 --temperature 1 --seed 0`。
Colab 应先更新 GitHub 项目；输出可放在 Drive，`--cache-dir /content/go2-hf-cache`
放在本地磁盘，避免从 Drive 反复读取历史模型。不要用旧版 Notebook 的 `--episodes`。

### 学习方与历史对手

默认 `--opponent-mode history`。学习方从 `Jycccc111/Go_RL_policy` 最新模型初始化，
启动时固定 HEAD 提交 SHA；下载使用固定版本，整次运行持续更新同一个学习方。
对手每盘从该仓库较早提交或保留的旧检查点中均匀随机选择，按 LFS/Git blob 标识去重，
排除当前学习方及其内容相同的备份。候选列表在启动时固定，不随仓库实时变化。
对手冻结不更新，学习方交替执黑白。逐局记录双方来源、提交 SHA、文件名和学习方颜色。

如果可访问的 RL 仓库确实没有模型，沿用最初回退源
`Jycccc111/Go-Policy/best_policy_network_128.pth`。网络、权限和损坏文件会明确报错。
若没有不同的旧模型，可显式用 `--bootstrap-opponent` 将最初监督模型归档进 RL 仓库，
建立初始历史对手；不覆盖根目录最新模型，且不能与 `--no-upload` 同用。
没有历史且未选择初始化时会报错，不会把当前模型冒充历史。

其他模式：`--opponent-mode self` 让学习策略执双方；
`--opponent-mode pool --pool-size 4` 使用本次运行的冻结副本池，每完成一个 epoch 加入一个新副本。
默认仓库历史模式不把本次新模型临时加入启动候选，它们上传后供下次运行发现。
`--checkpoint /path/policy.pth` 显式覆盖学习方来源；要自动使用仓库最新模型时省略它。

保持原 `Train_net.Policy_network` 和八层编码，按权重识别通道数与残差块数，
严格加载旧格式检查点及 `network.` 前缀。所有合法 logits 按温度采样，pass 与投降始终可选。
BatchNorm 在采样和反向传播时都用 eval；每盘重新计算动作概率，
损失为 `-mean(log pi(a|s) * reward)`，无基线、gamma=1，分块反传并把梯度范数裁剪到 1。

### 原规则与逐局抽样

双方累计动作包含落子、pass 和投降。少于 200 步投降，投降方负，按 −1/+1 训练；
少于 200 步连续双 pass，跳过计分和更新。至少 200 步结束时，调用原
`Game.cal_result()` 的深拷贝计分，不新增死活判断；投降动作单独按点目检查：
实际落后时投降 +1，实际领先时误投降 −1。达到上限而未自然结束的局跳过。
恰好第 200 步使用原计分，自然终局优先于同一步的上限截断。

**每局仅均匀随机选一个落子前盘面，不保存整盘所有盘面。**
范围是双方所有仍可行动的位置，包含初始空盘，排除终盘；使用常量内存的蓄水池抽样，
抽样随机源独立，不影响策略采样和对手选择。已知胜负且未跳过的局，给选中盘面标注绝对颜色标签：
黑胜 1、白胜 0。早期投降标签来自投降，至少 200 步的标签来自原点目。
未知或跳过局不伪造标签；仍在本地保留该局唯一选中盘面的 PNG 和结果 JSON。

`rl/value_data.py` 每 epoch 累积一份 `value.npz`：

| 字段 | 内容 |
| --- | --- |
| `states` | `(N,8,19,19)`，`int16`，每个有效标签棋局一个抽样盘面 |
| `results` | `(N,)`，`int8`，黑胜 1／白胜 0 |
| `boards` | 原盘面 `(N,19,19)` |
| `episodes`, `move_indices`, `players` | 棋局编号、抽样时已完成动作数、当时行棋颜色 |
| `metadata` | 抽样与棋局来源说明 |

`N` 是有标签的样本数，不是更新次数。数据兼容 `Train_value.validate_value_arrays`。
不会自动训练 value 网络；当前 `Value_network.fit` 的训练/验证划分需要至少两份数据文件，
应累积至少两个完整 epoch 数据块，不能复制同一份文件来凑训练和验证。

### 每 epoch 保存、上传与恢复

每盘的选中盘面和结果保留在 `<output>/games/episode-.../`，不再逐局向 HF 提交。
**只有一轮完成全部有效更新后，才保存和发布这一轮模型与数据：**

```text
<output>/epochs/epoch-000001/
  policy.pth       该轮策略与优化器、计数状态
  value.npz        该轮 value 样本数据块
  metadata.json    尝试局数、样本数、有效更新次数和 complete 状态
  boards.zip       所有尝试局唯一抽样盘面的 PNG + JSON
  ready.json       模型与数据的校验和，标记完整配对
  uploads.json     两个仓库的上传状态
<output>/last.pth   最近完整 epoch 模型的原子复制别名
```

模型每 epoch 只序列化一次到 `policy.pth`，再原子复制到 `last.pth`，不每盘保存权重。
数据集一次提交写入 `Jycccc111/Go_value_data/runs/<run_id>/epochs/epoch-.../`；
模型仓库一次提交更新根目录 `last.pth`、本次运行的 `last.pth`、该轮 `policy.pth` 和明确列出的源码，
写入 `Jycccc111/Go_RL_policy`。源码在开始时冻结于 `<output>/code/`，不会上传密钥或无关文件。
`boards.zip` 包含所有尝试局的单个抽样盘面；value 数据块只包含有确定标签且未跳过的样本。

`--no-upload` 只保留本地完整 epoch 输出。上传失败保留配对文件及状态，可只重传：

```sh
python3 -m rl.train --output runs/rl-001 --upload-only
```

补传仅处理完整 epoch，校验配对文件，并跳过已经成功的仓库；不继续训练。
如果一轮中断或达到尝试局数限制，本地可保留 `complete=false` 的局部 value 数据和清单，
但不上传，也不把它算作完成的 epoch。**由于按要求每 epoch 才保存模型，中断轮已做的更新会丢失；
只能从上一个完整 epoch 恢复。**

```sh
python3 -m rl.train --checkpoint runs/rl-001/last.pth --resume --output runs/rl-002 --epochs 1
```

新检查点恢复模型、优化器、棋局计数、epoch 和累计更新次数；旧检查点缺失的新计数从 0 开始，
不从旧棋局数推算 epoch。恢复的优化器学习率优先于 `--lr`；历史候选重新固定，
不恢复 Python/GPU 随机源或旧内存池，因此不保证逐位重放。
Hugging Face 登录用 `hf auth login` 或环境中的 `HF_TOKEN`，需读取模型及写入两个目标仓库。
不把 token 写进代码或配置。测试验证流程，不证明棋力提升，也不替原死活算法做正确性保证。

## English

### Units and commands

Use `python3 -m rl.train`; `Reinforcement_learning.py` accepts the same options.
Imports never train or construct the UI/AI.
Install `requirements-rl.txt`, run `python3 -m unittest discover -s tests -v`, and inspect `--help`.

**One default epoch means 4096 effective policy optimizer steps, not 4096 games.**
`--epochs 1` completes one additional epoch; `--updates-per-epoch` defaults to 4096.
An eligible game makes at most one optimizer step. Skipped games and games with no
learner action do not update or count toward 4096. Attempted games can exceed 4096.
A labelled game that ends before the learner acts may still yield one value sample,
so the sample count can also exceed 4096. Removed options `--episodes` and
`--snapshot-every` must not be used. Optional `--max-games-per-epoch` fails clearly
if the attempt limit is reached before the required updates; it never marks that
partial epoch complete.

For a short check, explicitly set `--updates-per-epoch 2`; that changes this run's
epoch size. A standard training command, to be executed deliberately, is:

```sh
python3 -m rl.train --output runs/rl-001 --epochs 1 --updates-per-epoch 4096 --device cuda
```

Use a new output directory. CPU is default; `cuda` and `mps` are optional devices.
Defaults: `--max-moves 800 --komi 7.5 --lr 0.00001 --temperature 1 --seed 0`.
In Colab update GitHub first, put output on Drive, and keep
`--cache-dir /content/go2-hf-cache` on local disk.
The older Notebook's `--episodes` command is obsolete.

### Latest learner and historical opponents

Default `--opponent-mode history` initializes from the latest model in
`Jycccc111/Go_RL_policy`, pins its starting HEAD SHA, and keeps updating that same
learner. Each game uniformly samples a distinct older checkpoint from retained
files or previous commits. Deduplicate LFS/Git blob identities and exclude the
learner plus identical backups. Freeze the candidate list at startup, freeze the
opponent weights, alternate learner color, and log both model sources and learner
color. Do not replace the learner's accumulated progress between games.

An accessible RL repo genuinely lacking weights retains the original fallback to
`Jycccc111/Go-Policy/best_policy_network_128.pth`. Network, access and corrupt-file
errors remain errors. If no earlier model exists, explicitly use
`--bootstrap-opponent` to archive the original supervised policy as initial history;
it does not replace the latest root policy and conflicts with `--no-upload`.
Without history or explicit initialization, fail instead of reusing the current
policy as a supposedly older opponent.

`--opponent-mode self` trains both sides of the same policy.
`--opponent-mode pool --pool-size 4` uses frozen in-run copies and adds one after
each completed epoch. Repository-history mode keeps its startup list unchanged;
newly published policies become candidates on the next run.
`--checkpoint /path/policy.pth` explicitly overrides the latest learner source.

Keep the original `Train_net.Policy_network`, architecture inference and eight
input planes. Strictly accept legacy state dictionaries and `network.` prefixes.
Sample from all legal logits, including pass and resignation. BatchNorm stays in
eval for rollout and backward. Recompute probabilities per game and use
`-mean(log pi(a|s) * reward)`, gamma 1, no baseline, chunked backward and norm-1
clipping. Frozen opponent actions never receive learner gradients.

### Original rules and one random board per game

Count both colors' actions, including pass/resign. Resignation before 200 actions
loses and trains with −1/+1; two passes before 200 skip scoring and updates.
At 200+ actions, score a deep copy through original `Game.cal_result()` with no
replacement life/death rules. For the final resignation action only, correct
resignation gets +1 and mistaken resignation −1 according to that board score.
Move-cap truncations skip. Exactly 200 uses original scoring; natural termination
at the cap takes precedence over truncation.

**Uniformly sample one pre-action board per game, across both colors.** Include
the initial board and exclude the terminal board. A constant-memory reservoir
uses independent RNG so it does not change policy or opponent sampling. A known,
non-skipped result labels that one position Black win=1, White win=0. Early resign
labels use resignation; 200+ labels use original scoring. Unknown/skipped results
never fabricate value labels, but their chosen PNG and result JSON remain local.

Each epoch's `value.npz` contains `states (N,8,19,19)` as int16 and binary
`results (N,)` as int8, plus `boards`, `episodes`, `move_indices`, `players` and
metadata. `N` counts labelled games, not optimizer steps. It passes the expected `Train_value.validate_value_arrays` schema. Value training is separate. Existing
`Value_network.fit` needs at least two files for its train/validation split:
accumulate two or more complete epoch chunks rather than duplicating a file.

### Epoch save, upload and resume

Keep every attempted game's one sampled PNG and result under
`<output>/games/episode-.../`; there are no per-game HF commits. Only after all
required updates complete, finalize `<output>/epochs/epoch-000001/` with
`policy.pth`, `value.npz`, `metadata.json`, `boards.zip`, `ready.json` checksums and
`uploads.json` status. Serialize policy weights once per epoch, then atomically
copy that file to `<output>/last.pth`. There are no per-game weight checkpoints.

One dataset commit writes the epoch's value chunk, manifest and optional board
archive to `Jycccc111/Go_value_data/runs/<run_id>/epochs/epoch-.../`. One model
commit updates root `last.pth`, run `last.pth`, the epoch policy and allowlisted
source in `Jycccc111/Go_RL_policy`. Source is frozen at run start under `code/`;
credentials and unrelated files are excluded. The board archive contains the one
chosen board for all attempted games; the value chunk includes only known,
non-skipped labels.

`--no-upload` retains complete epochs locally. For a failed upload, run
`python3 -m rl.train --output runs/rl-001 --upload-only`. Retry validates complete
model/data pairs and skips the repository that already succeeded; it does not
continue training. Partial epochs may retain local value data and a
`complete=false` manifest, but are never uploaded or counted as complete.
**Interrupting an epoch loses its unsaved policy updates; resume only from the
last completed epoch, as required by once-per-epoch saving.**

```sh
python3 -m rl.train --checkpoint runs/rl-001/last.pth --resume --output runs/rl-002 --epochs 1
```

New checkpoints restore model, optimizer, game counter, completed epochs and total
updates. Missing new counters in old checkpoints start at zero; never infer epochs
from old game counts. The saved optimizer LR overrides `--lr`. Rediscover pinned
history; Python/GPU RNG and old in-memory pool are not restored, so resume is not
a bitwise replay. Authenticate with `hf auth login` or `HF_TOKEN`, with read access
to models and write access to both target repos. Never save tokens in source or
configuration. Smoke tests verify mechanics, not playing strength or the original
life/death algorithm's correctness.
