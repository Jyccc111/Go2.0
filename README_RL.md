# REINFORCE 自博弈训练 / REINFORCE self-play

## 中文指南

### 入口与依赖

训练入口是 `python3 -m rl.train`，`python3 Reinforcement_learning.py` 接受相同参数。
导入模块不会启动训练，也不依赖 `AI.py`、`app.py` 或 Flask。
依赖为 `torch`、`numpy`、`huggingface_hub`；盘面 PNG 由轻量渲染器生成，不需要图形界面。

文件职责：

| 文件 | 职责 |
| --- | --- |
| `rl/game.py` | 原棋盘编码、合法动作、终局与奖励；调用原 `Game.cal_result()` |
| `rl/policy.py` | 兼容旧检查点、带合法动作掩码的采样、原子保存 |
| `rl/train.py` | 自博弈、REINFORCE 更新、历史对手池和命令行 |
| `rl/hub.py` | Hugging Face 上传、逐局盘面和结果文件 |
| `rl/opponents.py` | 固定最新模型版本、发现与随机加载仓库历史对手 |
| `tests/test_rl*.py` | 短小的 CPU 冒烟测试与模拟网络测试 |

### 最新学习方与随机历史对手

默认 `--opponent-mode history`：学习方从 `Jycccc111/Go_RL_policy` 当前主分支的最新
`last.pth` 初始化；启动时固定该次提交 SHA，下载也使用这个 SHA，避免加载过程中仓库更新
导致来源混杂。整次训练继续更新这一个学习方，不会每盘重新加载覆盖已学到的参数。
如果可访问的 RL 仓库确实没有任何模型，沿用最初的回退规则，加载
`Jycccc111/Go-Policy/best_policy_network_128.pth`。网络、权限或模型损坏不会当成“没有模型”。

对手每盘从同一仓库的旧版本中均匀随机抽取：包括启动时仓库保留的旧检查点，以及之前
Git 提交中的模型版本。按 LFS 内容哈希或 Git blob 标识去重，排除最新学习方本身；
根目录 `last.pth` 与 `runs/.../last.pth` 的同一份备份不会被算成两个版本。
候选名单在运行开始时固定，本次训练新生成的检查点在上传后供下一次运行使用。
每盘只加载一个冻结对手，学习方交替执黑、执白，只对学习方的动作更新梯度。
日志和逐局元数据记录双方仓库、提交 SHA、文件名与学习方颜色，便于复查抽到了哪个版本。

如果还没有旧版本，程序会明确报错，绝不把最新模型的备份冒充历史对手。
首次建立历史池可显式加 `--bootstrap-opponent`：把最初的监督模型
`Go-Policy/best_policy_network_128.pth` 保存为 `Go_RL_policy` 内
`bootstrap/<digest>/initial.pth`，作为初始历史对手；不会覆盖根目录的最新 `last.pth`。
这个选项会写入模型仓库，需要写权限；已有历史候选时不再创建重复的初始版本。

本次检查仓库时，只有一份实际模型：根目录和 `runs/.../last.pth` 内容相同，
更早提交仅为空仓库，因此首次启用严格历史对手需要先建立初始历史版本。

`--checkpoint /absolute/path/policy.pth` 是显式覆盖学习方来源；`--resume` 继续指定的本地
RL 检查点。想始终从仓库最新模型开始时，不传这两个选项。
加载保持 `Train_net.Policy_network` 原结构，按权重推断通道数和残差块数；
接受原始 `state_dict`、含 `model_state_dict` 的检查点及带 `network.` 前缀的权重。
使用严格加载和 `weights_only=True`，不修改输入模型文件。

### 如何运行

先在终端通过 `hf auth login` 登录有仓库写权限的 Hugging Face 账号，
或使用运行环境中已有的 `HF_TOKEN`。不要把 token 写进代码、命令参数或配置文件。

```sh
cd /Users/jiangyuncong/Desktop/Go2.0
python3 -m pip install -r requirements-rl.txt
python3 -m unittest discover -s tests -v
python3 -m rl.train --help
```

明确执行以下命令才开始训练。默认自动加载 Hub 模型、每局上传数据，结束后上传代码和模型：

```sh
python3 -m rl.train --output runs/rl-001 --episodes 1
```

当前仓库历史池尚空，首次运行应明确选择建立初始历史版本：

```sh
python3 -m rl.train --output runs/rl-first-history --episodes 1 --bootstrap-opponent
```

如果不想写入初始历史版本，先自行向仓库加入旧模型；在此之前默认历史模式会报错。
`--bootstrap-opponent` 与 `--no-upload` 不能同时使用。之后已有历史候选时通常不再需要该选项。
`--output` 必须是新目录。CPU 为默认设备；可用时用 `--device mps`（Apple）
或 `--device cuda`（NVIDIA）。`--no-upload` 保留所有本地输出并关闭本次训练结果上传；
默认历史模式仍需联网读取模型和历史。完全离线时使用本地 `--checkpoint`、
`--opponent-mode self` 和 `--no-upload`。

Colab 使用 GitHub 上更新后的源码；先更新干净的克隆再运行 Notebook。旧入口不支持这些新选项。

常用参数：`--max-moves 800`、`--komi 7.5`、`--lr 0.00001`、
`--temperature 1`、`--seed 0`。默认 `--episodes 1`，不会自动循环无限训练。

默认无需传池参数，即为仓库历史对手模式。其他模式必须显式选择：

- `--opponent-mode self`：同一学习策略执双方，双方按各自奖励更新。
- `--opponent-mode pool --pool-size 4`：保留旧的本次运行内存池；从学习方冻结副本开始，
  按 `--snapshot-every` 加入新副本并淘汰最旧副本。它与仓库历史池是不同的模式。

所有模式均每隔 `--snapshot-every 20` 局保存本地策略快照。运行开始也保存
`initial.pth`，正常结束后保留在仓库 `runs/<run_id>/initial.pth`，供将来的历史对手发现。

恢复训练时指定 RL 检查点和新的输出目录，例如：

```sh
python3 -m rl.train --checkpoint runs/rl-001/last.pth --resume --output runs/rl-002 --episodes 10
```

恢复模型、优化器、已完成局数和 CPU Torch 随机状态；恢复的优化器学习率优先于 `--lr`。
其他运行参数按本次命令设置；默认历史模式重新发现并固定仓库候选，内存池模式从恢复模型重建。
不恢复 Python/GPU 随机状态，因此不是逐位重放。省略 `--resume` 时只用检查点初始化权重，优化器重新开始。

### 终局、200 步规则和奖励

步数按双方累计动作计算，包含落子、pass 和投降。动作 `0..360` 为 `row*19+col`，
`361` 是 pass，`362` 是投降；pass 和投降在整盘中都可选，`(0,0)` 是正常落子。
连续两次 pass 或任意一方投降即结束。

| 结束情况 | 胜负与训练 |
| --- | --- |
| 少于 200 步投降 | 直接按投降结果判负，不点目；投降方 −1、对方 +1，包括投降动作 |
| 少于 200 步双方连续 pass | 不点目、不更新，但保存并上传盘面及跳过原因 |
| 至少 200 步，pass 或投降结束 | 调用原 `Game.cal_result()` 判断胜负，普通动作按点目胜负得到 +1/−1 |
| 至少 200 步的投降动作 | 单独使用投降方点目奖励的相反数：实际落后时投降 +1，实际领先时误投降 −1 |
| 达到步数上限但未自然结束 | 截断并跳过更新，结果标签未知；仍保存盘面和状态 |

恰好第 200 步结束使用原点目逻辑。天然终局与上限同时发生时，优先按天然终局处理。
200 步前的投降动作没有反转奖励，避免强化立即投降。
原计分在终局 `Game` 的深拷贝上执行并缓存结果，不修改保存的实际盘面。
不新增死活判断或面积计分，保留原 `black_win` 及其平局处理方式。

八层输入与原 `AI.encode_board` 一致：黑子、白子、`turn_since`、`air.board`、
`cluster_air`、当前颜色可提子数、对方可提子数、`islegal`，无新增归一化或视角变换。
合法动作使用当前行棋方原规则，并额外去掉已占交点；规则能力沿用原 `Game`。

从所有合法动作的 logits 按温度进行分类采样，不限制为前五。
每盘保存不带计算图的状态、掩码、动作和颜色，更新时重新计算相同分布。
BatchNorm 在采样和更新时均保持 eval，梯度仍可传播。
损失为 `-mean(log pi(a|s) * reward)`，`gamma=1`，无基线或回合内奖励标准化；
分块反向传播，梯度范数裁剪到 1，非有限值会报错并中止更新。

### 每局数据与上传

每一盘（包括被跳过的棋局）先保存在本地：

```text
<output>/games/episode-.../
  board.png       最终实际盘面及结果文字
  result.json     结束原因、步数、胜负来源、是否跳过、双方版本和学习方颜色等
  position.npz    终局编码、原盘面及可用的 value 标签
```

有明确胜负时，`states` 形状为 `(1,8,19,19)`，`results` 为 `(1,)`，
标签黑胜为 1、白胜为 0；未知或跳过的对局使用空 `states/results`，
仍保留原 `board` 和 `encoded_state`，不会把未知结果伪装成平局。
这是每盘一个终局样本，不是整盘所有中间局面的截图或训练样本。

元数据区分投降规则下的胜者与原点目胜者。200 步前投降的 value 标签来自投降结果；
至少 200 步的标签来自原点目，即使它与投降结果不同。应按 `result_source` 和
跳过标记筛选数据，而不是忽略来源混用。

默认每局将三份文件上传到数据集 `Jycccc111/Go_value_data`，位于
`runs/<run_id>/games/...`，避免不同运行互相覆盖。训练正常结束后，
将最后检查点和明确列出的训练代码、原 `Game.py`、`Train_net.py`、指南及测试
保存进 `Jycccc111/Go_RL_policy`；最新模型为仓库根目录 `last.pth`，
另保留本次运行的初始模型、最后模型及定期快照。不会上传整个工作目录、`.env`、token 或无关文件。

训练开始前将本次使用的明确代码清单冻结到 `<output>/code/`；
上传和重试均使用这一份代码，避免中途编辑项目后上传错版本。
本地 `last.pth` 每盘原子替换，保留上一个完整检查点直到新文件写入成功。
上传失败会明确报错，本地文件保留。修复网络或权限后，可只重传、不继续训练：

```sh
python3 -m rl.train --output runs/rl-001 --upload-only
```

此任务验证了短测试、仓库元数据读取和真实检查点：已下载并加载
`best_policy_network_128.pth`，自动识别为 128 通道、8 个残差块；
单次无梯度前向输出形状为 `(1,363)`，全部为有限值。没有启动长训练，
也没有实际上传测试数据或模型。上传交互用模拟测试覆盖；
原死活算法本身的正确性和棋力不由冒烟测试证明。

## English guide

### Entry points and dependencies

Run `python3 -m rl.train`; `python3 Reinforcement_learning.py` accepts the same options.
Imports never start training. The pipeline does not import `AI.py`, `app.py` or Flask.
Dependencies are `torch`, `numpy` and `huggingface_hub`. A lightweight renderer writes
board PNGs without a GUI.

`rl/game.py` adapts the existing game and rewards; `rl/policy.py` handles compatible
checkpoints, masked sampling and atomic saves; `rl/train.py` runs self-play and
updates; `rl/hub.py` handles uploads and per-game artifacts; `rl/opponents.py`
pins the current learner and discovers/downloads historical opponents.
`tests/test_rl*.py` contains short CPU and mocked-network tests.

### Latest learner and random historical opponents

The default is `--opponent-mode history`. Initialize the learner from the latest
root `last.pth` on `Jycccc111/Go_RL_policy`. Pin the starting commit SHA and use
that revision when downloading. Continue updating this same learner throughout
the run; do not reload and discard progress before each game. If the accessible
RL repository genuinely contains no checkpoint, retain the original fallback to
`Jycccc111/Go-Policy/best_policy_network_128.pth`. Network/authentication errors
and invalid checkpoints fail visibly instead of silently selecting a fallback.

For each game, uniformly sample one distinct earlier model from the repository:
retained old checkpoints at the starting HEAD and models from previous commits.
Deduplicate by LFS content hash or Git blob identity and exclude the current
learner. A root checkpoint and its identical run backup are one version, not two.
Freeze the candidate list at run start; newly produced snapshots become available
to the next run after upload. Load one frozen opponent at a time, alternate learner
color, and update learner actions only. Logs and game metadata identify learner
and opponent repository, full revision, filename and learner color.

If no old version exists, fail clearly rather than pretending that a backup of
the latest model is historical. To initialize an empty history explicitly, add
`--bootstrap-opponent`: archive the original supervised
`Go-Policy/best_policy_network_128.pth` into the RL repository at
`bootstrap/<digest>/initial.pth`. It is then an initial historical opponent;
root `last.pth` is not replaced. This writes to the model repository and needs
write access. Existing historical candidates avoid duplicate initialization.
At inspection time the repository contained only one actual checkpoint, duplicated
at root and a run path; the previous commit had no weights. The first strict
historical run therefore needs an initial historical version.

`--checkpoint /absolute/path/policy.pth` explicitly overrides the learner source;
`--resume` continues that specified local RL checkpoint. Omit both to start from
the repository's latest model.

The loader preserves `Train_net.Policy_network`, infers channels/residual blocks
from weights, and accepts raw state dictionaries, `model_state_dict` checkpoints
and `network.` prefixes. It loads strictly with `weights_only=True` and leaves the
source checkpoint untouched.

Authenticate with `hf auth login` using an account with repository write access,
or provide `HF_TOKEN` through the existing environment. Never put tokens in code,
CLI arguments or saved configuration.

```sh
cd /Users/jiangyuncong/Desktop/Go2.0
python3 -m pip install -r requirements-rl.txt
python3 -m unittest discover -s tests -v
python3 -m rl.train --output runs/rl-001 --episodes 1
```

The training command downloads the latest model, samples historical opponents,
uploads each game's data, and uploads code/model after successful completion.
The current repository has no distinct old weights. To explicitly initialize history:

```sh
python3 -m rl.train --output runs/rl-first-history --episodes 1 --bootstrap-opponent
```

Alternatively, add an older checkpoint to the repository yourself; default historical
mode fails until a previous version exists. `--bootstrap-opponent` cannot be combined
with `--no-upload`. It is normally unnecessary once historical candidates exist.
The output directory must be new. CPU is the default; use `--device mps` or
`--device cuda` when available. `--no-upload` keeps training outputs local;
default historical mode still needs networking. For completely offline use,
combine a local `--checkpoint`, `--opponent-mode self` and `--no-upload`.

In Colab, update the clean GitHub checkout before using these options; the old
entry point does not contain the historical-opponent implementation.

Defaults include `--episodes 1`, `--max-moves 800`, `--komi 7.5`, `--lr 0.00001`,
`--temperature 1` and `--seed 0`. There is no endless training loop.
No pool options are needed for default repository history. Other modes are
explicit: `--opponent-mode self` plays both sides with the learning policy and
per-color rewards. `--opponent-mode pool --pool-size 4` retains the earlier
in-run memory pool: start with a frozen learner copy, add snapshots at the
interval and evict the oldest when full. This differs from repository history.
All modes save snapshots every `--snapshot-every 20` games. Preserve the starting
checkpoint as local `initial.pth` and publish it at `runs/<run_id>/initial.pth`
after successful completion, making initial versions discoverable in later runs.

To resume, use `--checkpoint runs/rl-001/last.pth --resume` and a new output folder.
This restores weights, optimizer, episode counter and CPU Torch RNG; the saved
optimizer learning rate overrides `--lr`. Other options are supplied again.
Default historical mode rediscovers and pins the repository candidates; memory
pool mode restarts from resumed weights. Python/GPU RNG and prior pool contents
are not restored, so resume is not a bitwise replay. Without `--resume`, only weights
are initialized from the checkpoint and the optimizer starts fresh.

### Termination and reward rules

Count both players' actions, including pass and resignation. Actions `0..360`
mean `row*19+col`, 361 is pass and 362 is resignation. Both special actions remain
available before and after move 200; `(0,0)` is an ordinary board point.
Two consecutive passes or any resignation terminate a game.

| Ending | Result and update |
| --- | --- |
| Resignation before 200 actions | Resigning side loses; ordinary −1/+1 returns, including −1 for resignation; no scoring call |
| Two passes before 200 actions | Skip scoring and updates; archive the board and skip status |
| Pass/resignation at 200 or more actions | Use original `Game.cal_result()`; ordinary actions get board winner +1, loser −1 |
| Resignation action at 200 or more actions | Override only that action with the negative of the resigning side's board return: correct resignation +1, mistaken resignation −1 |
| Move cap without natural termination | Skip updates and leave the outcome unknown; archive the game |

Exactly 200 actions uses original scoring. A natural ending at the move cap takes
precedence over truncation. Early resignation has no sign override, avoiding
reinforcement of immediate resignation.

Run original scoring on a deep copy and cache its result, preserving the actual
terminal board. No replacement life/death or area scorer is introduced. The
original `black_win` behavior, including its tie convention, is retained.

The eight input planes match `AI.encode_board`: black, white, `turn_since`,
`air.board`, `cluster_air`, current-color captures, opponent captures and
`islegal`. There is no new normalization or perspective transform. The mask uses
the actual side-to-move's existing legality map and excludes occupied points;
move/capture/suicide/ko behavior remains that of the original `Game`.

Sample from all legal logits at the configured temperature. Store detached
states/masks/actions/colors and recompute the same distribution for one update
per eligible episode. BatchNorm stays in eval during rollout and backward;
gradients still flow. The loss is `-mean(log pi(a|s) * reward)`, with gamma 1,
no baseline or within-episode normalization. Chunked backward limits graph
memory, gradient norm is clipped to 1, and nonfinite values abort the update.

### Artifacts, upload and recovery

Every game, including skipped ones, first writes a local folder under
`<output>/games/episode-.../` containing `board.png`, `result.json` and
`position.npz`. The PNG shows the actual terminal board and result text; JSON
records termination, action count, winner/source, skip status, learner/opponent
versions and learner color.
This stores one terminal position per game, not all intermediate positions.

For known outcomes, NPZ `states` has shape `(1,8,19,19)` and `results` has shape
`(1,)`, with Black win 1 and White win 0. Unknown/skipped games have empty
`states/results` while retaining raw `board` and `encoded_state`; they are never
mislabeled as draws. Metadata distinguishes the official resignation winner from
the adjudicated board winner. Early-resignation labels use resignation results;
labels at 200+ actions use original board scoring, even when they disagree with
resignation. Filter by source/skip status when consuming the dataset.

By default, upload all three files after every game to dataset
`Jycccc111/Go_value_data` under `runs/<run_id>/games/...`. At successful training
completion, upload the final model plus an explicit allowlist of training code,
original `Game.py`/`Train_net.py`, guide and tests to `Jycccc111/Go_RL_policy`.
The latest checkpoint is root `last.pth`; retain run-scoped initial weights,
final weights and interval snapshots for future historical opponents.
Never upload the whole workspace, `.env`, tokens or unrelated files.

Before training, freeze the explicit source allowlist into `<output>/code/`.
Uploads and retries use that same frozen source, so later project edits cannot
silently change the code associated with the run.
Local `last.pth` is atomically replaced after each episode, preserving the previous
complete checkpoint until the new write succeeds. Failed uploads raise visibly
and retain local artifacts. After fixing access/networking, retry without training:

```sh
python3 -m rl.train --output runs/rl-001 --upload-only
```

Verification included short tests, real repository metadata reads and the actual
`best_policy_network_128.pth`: loading inferred 128 channels and 8 residual blocks;
a single no-gradient forward pass produced finite logits of shape `(1,363)`.
No long training or live test-data/model upload was performed. Upload behavior
is covered with mocks; smoke tests do not establish playing strength or validate
the original life/death algorithm itself.
