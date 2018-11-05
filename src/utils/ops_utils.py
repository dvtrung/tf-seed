def evaluate(target, result, decode_fn, metrics="default"):
    if metrics == "default":
        return int(target != result), decode_fn(target), decode_fn(result)