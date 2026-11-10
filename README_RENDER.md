# Render 部署 / Render deployment

本项目部署为 Python Web Service，提供网页、围棋规则和 CPU 模型推理。强化学习仍在本地或 Colab 执行，不在 Web Service 中启动训练。

## 部署配置

仓库根目录已提供 `render.yaml`，可在 Render 选择 **New → Blueprint**，连接 `Jyccc111/Go2.0` 的 `main` 分支。也可以手动创建 Web Service：

| 设置 | 值 |
| --- | --- |
| Runtime | Python 3 |
| Python | `.python-version` 固定为 3.12 系列 |
| Build Command | `pip install --no-cache-dir -r requirements.txt` |
| Start Command | `gunicorn --config gunicorn.conf.py app:app` |
| Health Check | `/healthz` |
| Instance | Free（先验证，不自动升级） |

Blueprint 自动生成 `SECRET_KEY`，开启 `GO_ISOLATE_SESSIONS=1`，限定 4 个浏览器会话，闲置会话保留 1 小时，value 批量推理设为 16，CPU 计算线程设为 1。手动部署时按 `render.yaml` 填写这些环境变量。`PORT` 由 Render 提供，无需写死。

`requirements.txt` 在 Linux 安装 CPU 版 PyTorch；Mac 使用普通的 Mac wheel。依赖版本与本地验证环境一致；不要改用 Colab 的 CUDA 安装步骤。公开 Hugging Face 模型无需提供 token，尤其不要把训练用 write token 放进代码。如果以后加载私有仓库，在 Render 环境变量中单独配置只读 `HF_TOKEN`。

## 上线后的行为

- 中文、English、黑白独立模型选择和慢放均可使用。
- “强化学习模型（最新）”仍加载 `Jycccc111/Go_RL_policy/last.pth`。模型首次使用时下载；正在内存中使用的模型不随上传自动更换。
- Render 上每个浏览器使用签名 cookie 指向自己的服务器内存棋局与 SGF 回放，不共用棋盘。桌面默认保留原行为；设置 `GO_ISOLATE_SESSIONS=1` 也能在本地启用隔离。
- 当前架构必须保持 **1 个 worker、1 个实例**。Gunicorn 配置包含多线程与状态锁；不要增加 worker 或水平扩容，除非先接入共享会话存储。
- 浏览器对局存在内存中。重启、部署或服务休眠后对局可能丢失；闲置过久或超过会话容量时旧会话会被回收。过期对局的落子请求返回提示，需要刷新后重新开始。
- 模型缓存位于临时磁盘，重启后可能重新下载。不要把模型权重、Hugging Face token 或 `runs/` 的本地训练数据提交到 GitHub。
- SGF 上传上限 2 MB，超过时返回 JSON 错误。

## 免费服务与资源

2026-11-10 核查：Free 实例为 512 MB，闲置 15 分钟会休眠，恢复时有冷启动。PyTorch、双方 AI、value 网络和推理峰值可能超过这个内存额度；因此免费部署用于验证，不能保证完整 AI 功能在该额度下稳定。

如果日志显示超内存，再考虑 2 GB 实例；当日官方列价约 $25/月，需要用户自行确认费用后升级。本配置不包含付费磁盘、数据库或付费实例。不要仅为保活而添加循环访问。

参考：[Flask 部署](https://render.com/docs/deploy-flask)、[免费服务限制](https://render.com/docs/free)、[实例价格](https://render.com/pricing)、[Python 版本](https://render.com/docs/python-version)。

## 本地验证

```sh
python3 -m unittest discover -s tests -v
GO_ISOLATE_SESSIONS=1 PORT=10000 gunicorn --config gunicorn.conf.py app:app
```

健康检查不加载模型，也不创建浏览器棋局。新增测试覆盖浏览器棋盘和模型选择隔离、SGF 隔离、会话过期、容量与活跃请求保护，以及上传限制。发布时保留原点目、投降和 RL 奖励代码。

## English

Deploy a Python Web Service from the `main` branch with the supplied `render.yaml`, or use the build/start/health settings above. Python 3.12 and CPU-only Linux PyTorch are configured. The web service performs inference; training remains on your Mac or Colab. Public HF models require no token; use a separate read-only `HF_TOKEN` only for private models.

Browser games and SGF replays are isolated using signed cookies and bounded in-memory state. Run exactly one worker and one instance. Games are not persistent: deployments, restarts, idle expiry or capacity eviction may discard them. Expired move requests ask the browser to refresh. The default Blueprint retains up to four sessions for an hour of inactivity. Model caches are temporary, and SGF uploads are capped at 2 MB.

The Blueprint selects Free and creates no paid databases/disks. As checked on 2026-11-10, Free has 512 MB and sleeps after 15 minutes of inactivity. AI memory peaks may exceed that budget. Test first and review logs; a 2 GB instance was listed at about $25/month and must be explicitly approved before upgrading. The service is not yet deployed merely because these files exist; verify the actual build, public URL, health endpoint and AI operation in Render.
