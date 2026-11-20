REWARD_METADATA_FIELDS = (
    "source_id", "sample_id", "schema_version", "annotation", "ground_truth", "requested_fields",
)


def has_reasoning(value):
    return isinstance(value, str) and bool(value.strip())


def validate_media_sample(sample, *, single_turn_image=False, preference=False):
    if "image" in sample and "video" in sample:
        raise ValueError("同一样本不能混合 image/video。")
    if preference:
        texts = [sample.get("prompt")]
        for key in ("chosen", "rejected"):
            if not isinstance(sample.get(key), str) or not sample[key].strip():
                raise ValueError(f"DPO {key} 必须是非空字符串。")
            if "<image>" in sample[key] or "<video>" in sample[key]:
                raise ValueError("DPO回答不能包含媒体占位符。")
    else:
        turns = sample.get("conversations")
        if not isinstance(turns, list) or not turns or len(turns) % 2:
            raise ValueError("conversations 必须是非空的 human/gpt 偶数成对列表。")
        if single_turn_image and len(turns) != 2:
            raise ValueError("GRPO业务主线只支持单轮 human/gpt 图片样本。")
        texts = []
        for index, turn in enumerate(turns):
            expected = "human" if index % 2 == 0 else "gpt"
            if not isinstance(turn, dict) or turn.get("from") != expected:
                raise ValueError(f"第{index}轮角色必须是 {expected}。")
            value = turn.get("value")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"第{index}轮 value 必须是非空字符串。")
            if expected == "gpt" and ("<image>" in value or "<video>" in value):
                raise ValueError("回答不能包含媒体占位符。")
            if expected == "human":
                texts.append(value)
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("prompt 必须是非空字符串。")
    for media in ("image", "video"):
        files = sample.get(media, [])
        if isinstance(files, str):
            files = [files]
        if not isinstance(files, list) or any(not isinstance(path, str) or not path for path in files):
            raise ValueError(f"{media} 必须是路径字符串或非空路径列表。")
        if media in sample and not files:
            raise ValueError(f"{media} 不能为空。")
        count = sum(text.count(f"<{media}>") for text in texts)
        if count != len(files):
            raise ValueError(f"{media} 占位符数量 {count} 与文件数量 {len(files)} 不一致。")
    if single_turn_image and ("image" not in sample or "video" in sample):
        raise ValueError("GRPO业务主线要求图片，拒绝视频或纯文本样本。")
    return sample


def require_token_budget(length, maximum, *, sample_id=None):
    if maximum is None or maximum <= 0:
        raise ValueError("必须提供正整数token预算。")
    if length > maximum:
        raise ValueError(
            f"样本 {sample_id!r} 实际token长度 {length} 超出预算 {maximum}；"
            "禁止静默截断视觉tokens，请调整像素预算或序列长度。"
        )
    return length


def reward_metadata(sample):
    return {key: sample.get(key) for key in REWARD_METADATA_FIELDS}
