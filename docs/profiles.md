# Profile 配置系统

Profile 系统将常用的参数组合命名保存在 `profiles.yaml` 中，避免每次运行都要写长命令行。

## 快速开始

```bash
# 查看所有可用 profile
python request.py --list-profiles

# 使用 profile 运行
python request.py --profile comt_vsp --max_tasks 20

# CLI 参数覆盖 profile 中的值
python request.py --profile comt_vsp --model custom-model --max_tasks 5
```

## 参数优先级

从高到低：

1. **CLI 显式指定** — 命令行中写的参数，始终优先
2. **Profile 值** — `profiles.yaml` 中定义的参数
3. **Defaults** — `profiles.yaml` 中的 `defaults` 段
4. **argparse 默认值** — `request.py` 中 `parser.add_argument` 的 default

> **注意**: 当 CLI 参数值恰好等于 argparse 默认值时（如 `--model gpt-5`），系统无法区分是显式指定还是未指定，此时 profile 值会生效。实际使用中基本不会遇到此情况。

## profiles.yaml 结构

```yaml
# 全局默认值，所有 profile 继承
defaults:
  provider: openrouter
  model: gpt-5
  consumers: 20
  # ...

# Profile 定义
comt_vsp:
  mode: comt_vsp
  model: "qwen/qwen3-vl-235b-a22b-instruct"
  consumers: 3

# 继承另一个 profile（单级继承）
comt_vsp_cf:
  _inherit: comt_vsp
  tunnel: cf
```

### 参数名映射

Profile 中的参数名与 CLI 参数名（`--xxx`）一致，去掉 `--` 前缀即可。例如：

| CLI 参数 | Profile 键名 |
|---------|-------------|
| `--mode direct` | `mode: direct` |
| `--provider self` | `provider: self` |
| `--model X` | `model: X` |
| `--consumers 3` | `consumers: 3` |
| `--sampling_rate 0.12` | `sampling_rate: 0.12` |
| `--vsp_postproc` | `vsp_postproc: true` |
| `--llm_base_url URL` | `llm_base_url: URL` |
| `--tunnel cf` | `tunnel: cf` |
| `--image_types SD TYPO` | `image_types: [SD, TYPO]` |

## 继承

使用 `_inherit` 键继承另一个 profile 的所有设置，然后覆盖需要修改的部分：

```yaml
comt_vsp:
  mode: comt_vsp
  model: "qwen/qwen3-vl-235b-a22b-instruct"
  consumers: 3

comt_vsp_cf:
  _inherit: comt_vsp     # 继承上面的全部设置
  tunnel: cf              # 只改 tunnel
```

合并顺序：`defaults` → 父 profile → 当前 profile

仅支持单级继承（不能 A → B → C）。

## 验证

Profile 加载时自动检查参数组合的兼容性：

| 规则 | 级别 | 说明 |
|------|------|------|
| `vsp_postproc` + `mode=direct` | 错误 | VSP 后处理仅在 VSP/CoMT-VSP 模式下有效 |
| `provider=self` 无 `llm_base_url` | 错误 | 自部署 provider 必须指定 LLM 端点 |
| `comt_sample_id` + `mode=direct` | 警告 | CoMT 参数在 direct 模式下无效 |
| `provider=openrouter` + `llm_base_url` | 警告 | 建议使用 `provider: self` |

## 预定义 Profile

| Profile | 说明 |
|---------|------|
| `direct` | OpenRouter 直接调用（最简模式） |
| `qwen235b` | Qwen 235B via OpenRouter + Alibaba |
| `comt_vsp` | CoMT-VSP 标准模式（SSH tunnel） |
| `comt_vsp_cf` | CoMT-VSP + Cloudflare Tunnel |
| `comt_vsp_prebaked_ask` | CoMT-VSP + prebaked 后处理（ASK fallback） |
| `comt_vsp_prebaked_sd_good` | CoMT-VSP + prebaked 后处理（SD good fallback） |
| `comt_vsp_prebaked_sd_bad` | CoMT-VSP + prebaked 后处理（SD bad fallback） |
| `autodl_qwen` | 自部署 Qwen（AutoDL，CF tunnel） |
| `autodl_comt_vsp` | 自部署 Qwen + CoMT-VSP |

## 预览配置

不实际运行，只显示解析后的完整配置：

```bash
# 单次运行预览
python request.py --profile comt_vsp --tunnel cf --show-config

# batch 配置对比表格（显示所有组合的差异列）
python batch_request.py --show-config
```

`batch_request.py --show-config` 会以表格形式对比所有组合，只显示有差异的参数列。

## 在 batch_request.py 中使用

`args_combo` 中可以直接使用 `--profile`：

```python
args_combo = [
    "--tunnel cf --sampling_rate 0.01",
    [
        '--profile qwen235b',
        '--profile comt_vsp_prebaked_ask',
        '--profile comt_vsp_prebaked_sd_good',
        '--profile comt_vsp_prebaked_sd_bad',
    ],
]
```

## 自定义 Profile 文件

默认读取项目根目录的 `profiles.yaml`，也可以指定其他文件：

```bash
python request.py --profile-file my_profiles.yaml --profile my_profile
```

## 相关文件

| 文件 | 说明 |
|------|------|
| `profiles.yaml` | Profile 定义文件 |
| `profile_loader.py` | 加载、继承解析、验证、应用逻辑 |
| `request.py` | `--profile`/`--profile-file`/`--list-profiles`/`--show-config` 参数入口 |
| `batch_request.py` | 支持 `--show-config` 对比表格 |
