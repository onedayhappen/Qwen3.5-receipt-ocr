"""训练与推理共用的纯业务提示词，不包含图像占位符。"""

import json

from .schema import get_schema


def build_prompt(schema_version="receipt_v1", fields=None):
    schema = get_schema(schema_version, fields)
    item_instructions = ""
    if "items" in schema.get("properties", {}):
        item_instructions = (
            "逐行提取商品明细，不漏行、不重复、不虚构商品；"
            "name、quantity、price 缺失或不可读时用 null，不默认数量为 1。"
            "price 保留票面值；无明确上下文时不要认定它是单价或行金额，不自行乘数量。"
            "主商品及有名称的子商品按票面顺序作为独立行列出，保留原始名称，不把套餐子项价格加回主项。"
            "items 仅在确认票面没有商品明细时使用 []，无法确定时使用 null。"
        )
    return (
        "读取所提供票据图像，只输出一个符合以下 JSON schema 的对象。"
        "不要输出 Markdown、解释、推理过程或额外文本。"
        "只提取 schema 允许的字段，不输出其他字段。"
        "忠实保留票面原文、金额单位和字段归属，不改写或纠正商户、编号或金额。"
        "没有印出或无法读取的信息用 null，不依据常识、加总关系或其他字段猜测补全。"
        "日期存在歧义时保留原文，不猜测日月顺序。"
        + item_instructions + "\nJSON schema:\n"
        + json.dumps(schema, ensure_ascii=False, sort_keys=True)
    )
