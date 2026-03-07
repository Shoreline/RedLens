"""
Profile 配置加载器

从 profiles.yaml 加载预定义参数组合，支持继承和 CLI 覆盖。

优先级（从高到低）：CLI 显式指定 > profile 值 > defaults > argparse 默认值
"""

import os
import yaml


def load_profiles(path="profiles.yaml"):
    """加载 profiles.yaml，返回 {name: dict} 映射（含 'defaults'）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Profile 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Profile 文件格式错误: 顶层必须是字典")
    return data


def resolve_profile(name, profiles):
    """
    解析 profile，合并 defaults → 父 profile → 当前 profile。
    支持单级 _inherit。

    Returns:
        合并后的参数字典（不含 _inherit 键）
    """
    if name not in profiles:
        available = [k for k in profiles if k != "defaults"]
        raise KeyError(f"Profile '{name}' 不存在。可用的 profile: {', '.join(available)}")

    defaults = profiles.get("defaults", {})
    current = profiles[name]

    # 处理继承
    parent_name = current.get("_inherit")
    if parent_name:
        if parent_name not in profiles:
            raise KeyError(f"Profile '{name}' 继承的 '{parent_name}' 不存在")
        if profiles[parent_name].get("_inherit"):
            raise ValueError(f"不支持多级继承: '{name}' -> '{parent_name}' -> '{profiles[parent_name]['_inherit']}'")
        parent = profiles[parent_name]
    else:
        parent = {}

    # 合并：defaults → parent → current
    merged = {}
    merged.update(defaults)
    merged.update(parent)
    merged.update(current)
    merged.pop("_inherit", None)

    return merged


def get_cli_explicit_args(parser, argv=None):
    """
    检测哪些参数是 CLI 显式指定的（非默认值来源）。

    通过对比 parse_args 结果和 parser 默认值来判断。
    对于 store_true 参数，只有在 CLI 中显式出现才视为显式指定。

    Returns:
        set: 显式指定的参数名集合（使用 dest 名称）
    """
    import sys

    # 获取所有参数的默认值
    defaults = {}
    for action in parser._actions:
        if action.dest != "help":
            defaults[action.dest] = action.default

    # 解析实际命令行
    args = parser.parse_args(argv)

    # 找出哪些和默认值不同的参数
    explicit = set()
    for dest, default_val in defaults.items():
        actual_val = getattr(args, dest, None)
        if actual_val != default_val:
            explicit.add(dest)

    return explicit, args


# profile 参数名 → argparse dest 名的映射
# 大多数参数名一致，只有少数需要映射
_PARAM_MAP = {
    "temp": "temp",
    "consumers": "consumers",
    "image_types": "image_types",
    "categories": "categories",
    "vsp_postproc": "vsp_postproc",
}


def apply_profile(args, profile_dict, explicit_args):
    """
    将 profile 值应用到 argparse namespace。
    CLI 显式指定的参数不会被覆盖。

    Args:
        args: argparse.Namespace
        profile_dict: resolve_profile 返回的合并字典
        explicit_args: CLI 显式指定的参数名集合

    Returns:
        修改后的 args（原地修改）
    """
    applied = []
    skipped = []

    for key, value in profile_dict.items():
        # 映射参数名
        dest = _PARAM_MAP.get(key, key)

        if not hasattr(args, dest):
            # 未知参数（可能是 profile 专用的元数据），跳过
            continue

        if dest in explicit_args:
            skipped.append(f"{key}={value}")
            continue

        setattr(args, dest, value)
        applied.append(f"{key}={value}")

    return args, applied, skipped


def validate_profile(resolved_dict):
    """
    检查 profile 参数组合的兼容性。

    Returns:
        (errors: list[str], warnings: list[str])
    """
    errors = []
    warnings = []

    mode = resolved_dict.get("mode", "direct")
    provider = resolved_dict.get("provider", "openrouter")

    # vsp_postproc + direct mode
    if resolved_dict.get("vsp_postproc") and mode == "direct":
        errors.append("vsp_postproc=true 不能与 mode=direct 一起使用")

    # comt_sample_id + direct mode
    if resolved_dict.get("comt_sample_id") and mode == "direct":
        warnings.append("comt_sample_id 在 mode=direct 下无效，将被忽略")

    # provider=self 必须有 llm_base_url
    if provider == "self" and not resolved_dict.get("llm_base_url"):
        errors.append("provider=self 需要指定 llm_base_url")

    # provider=openrouter + llm_base_url
    if provider == "openrouter" and resolved_dict.get("llm_base_url"):
        warnings.append("provider=openrouter 同时设置了 llm_base_url，建议使用 provider=self")

    return errors, warnings


def list_profiles(profiles):
    """列出所有可用的 profile（排除 defaults），使用解析后的值"""
    result = []

    for name in profiles:
        if name == "defaults":
            continue
        resolved = resolve_profile(name, profiles)
        mode = resolved.get("mode", "direct")
        provider = resolved.get("provider", "openrouter")
        model = resolved.get("model", "")
        inherit = profiles[name].get("_inherit", "")

        desc = f"mode={mode}, provider={provider}, model={model}"
        if inherit:
            desc += f" (继承自 {inherit})"
        result.append((name, desc))

    return result
