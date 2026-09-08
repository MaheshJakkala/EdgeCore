# """Qwen2 chat template application for EdgeCore."""
# from __future__ import annotations
# import json
# from pathlib import Path

# QWEN_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}{{ '<|system|>\n' + messages[0]['content'] + '<|im_end|>\n' }}{% else %}{{ '<|system|>\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"""


# def apply_qwen_chat_template(messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
#     """Apply Qwen2 chat template to messages."""
#     result = []
    
#     if messages and messages[0]['role'] == 'system':
#         result.append('<|system|\n' + messages[0]['content'] + '<|im_end|\n')
#         start_idx = 1
#     else:
#         result.append('<|system|\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|\n')
#         start_idx = 0
    
#     for msg in messages[start_idx:]:
#         role = msg['role']
#         content = msg['content']
#         if role == 'user':
#             result.append('<|user|\n' + content + '<|im_end|\n')
#         elif role == 'assistant':
#             result.append('<|assistant|\n' + content + '<|im_end|\n')
    
#     if add_generation_prompt:
#         result.append('<|assistant|\n')
    
#     return ''.join(result)


# def load_chat_template_from_hf(hf_dir: Path) -> str:
#     """Load chat template from HF tokenizer config."""
#     tokenizer_config = hf_dir / "tokenizer_config.json"
#     if tokenizer_config.exists():
#         with open(tokenizer_config) as f:
#             config = json.load(f)
#         return config.get("chat_template", QWEN_CHAT_TEMPLATE)
#     return QWEN_CHAT_TEMPLATE


# if __name__ == "__main__":
#     messages = [
#         {"role": "user", "content": "Explain what a CPU cache is in simple terms."}
#     ]
#     formatted = apply_qwen_chat_template(messages)
#     print(formatted)

"""Qwen2 chat template application for EdgeCore."""
from __future__ import annotations
import json
from pathlib import Path

# Matches the actual Qwen2.5-Instruct chat_template.jinja shipped by HF:
# role delimiters are <|im_start|>{role}\n ... <|im_end|>\n
QWEN_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}"
    "{% else %}"
    "{{ '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
    "You are a helpful assistant.<|im_end|>\n' }}"
    "{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


def apply_qwen_chat_template(messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    """Apply Qwen2 chat template to messages.

    Uses <|im_start|>{role}\\n ... <|im_end|>\\n exactly as Qwen2/Qwen2.5
    were trained, since <|im_start|>/<|im_end|> are the only role
    delimiters present in the tokenizer's special-token vocab. Anything
    else (e.g. <|system|>) isn't a real vocab entry and gets shredded by
    BPE instead of passed through as a single control token.
    """
    result = []

    if messages and messages[0]['role'] == 'system':
        result.append('<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n')
        start_idx = 1
    else:
        result.append(
            '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. '
            'You are a helpful assistant.<|im_end|>\n'
        )
        start_idx = 0

    for msg in messages[start_idx:]:
        role = msg['role']
        content = msg['content']
        if role == 'user':
            result.append('<|im_start|>user\n' + content + '<|im_end|>\n')
        elif role == 'assistant':
            result.append('<|im_start|>assistant\n' + content + '<|im_end|>\n')

    if add_generation_prompt:
        result.append('<|im_start|>assistant\n')

    return ''.join(result)


def load_chat_template_from_hf(hf_dir: Path) -> str:
    """Load chat template from HF tokenizer config."""
    tokenizer_config = hf_dir / "tokenizer_config.json"
    if tokenizer_config.exists():
        with open(tokenizer_config) as f:
            config = json.load(f)
        return config.get("chat_template", QWEN_CHAT_TEMPLATE)
    return QWEN_CHAT_TEMPLATE


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Explain what a CPU cache is in simple terms."}
    ]
    formatted = apply_qwen_chat_template(messages)
    print(formatted)
    assert "<|system|" not in formatted and "<|user|" not in formatted
    assert formatted.count("<|im_start|>") == formatted.count("<|im_end|>") + 1  # generation prompt has no closing tag