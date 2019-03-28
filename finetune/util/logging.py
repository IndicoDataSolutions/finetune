
def truncate_text(text, max_chars=100):
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text
