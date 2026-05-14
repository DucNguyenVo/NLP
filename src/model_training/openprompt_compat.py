"""
Phải được import trước bất kỳ module `openprompt` nào.

OpenPrompt 1.0.1 kỳ vọng hằng `SPECIAL_TOKENS_MAP_FILE` trong
`transformers.tokenization_utils` (transformers cũ). Từ khoảng 4.30+ hằng này
đã bị gỡ; thêm lại để tránh ImportError khi dùng transformers 4.36–4.45.
"""


def patch_transformers_for_openprompt() -> None:
    import transformers.tokenization_utils as _tu

    if not hasattr(_tu, "SPECIAL_TOKENS_MAP_FILE"):
        _tu.SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"


patch_transformers_for_openprompt()
