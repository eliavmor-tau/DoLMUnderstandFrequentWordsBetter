from hashlib import md5


def string_to_md5(input: str) -> str:
    encoded_input = input.encode()
    return str(md5(encoded_input).hexdigest())