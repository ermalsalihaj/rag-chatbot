INJECTION_PATTERNS = [
    "ignore previous instructions",
    "disregard the above",
    "reveal the system prompt",
    "system prompt",
    "do not cite",
    "override",
    "jailbreak",
    "developer message",
    "act as",
    "you are chatgpt",
    "follow these instructions",
]

DOC_INJECTION_PATTERNS = [
    "ignore all previous",
    "system prompt",
    "developer instructions",
    "do not answer using",
    "bypass",
    "jailbreak",
]

def looks_like_prompt_injection(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in INJECTION_PATTERNS)

def doc_looks_malicious(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in DOC_INJECTION_PATTERNS)
