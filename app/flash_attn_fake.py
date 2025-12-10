# Fake flash_attn module to bypass Florence-2 import check
# Florence-2 checks for flash_attn but we don't need it on CPU

def flash_attn_func(*args, **kwargs):
    raise NotImplementedError("flash_attn not available on CPU")

def flash_attn_varlen_func(*args, **kwargs):
    raise NotImplementedError("flash_attn not available on CPU")
