INJECTION_PATTERNS = [
    "ignore previous instructions",
    "disregard the above",
    "reveal the system prompt",
    "system prompt",
    "do not cite",
    "override",
    "jailbreak",
]

def looks_like_prompt_injection(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in INJECTION_PATTERNS)
