def ask_int(s: str, default: int = 0) -> int:
    try:
        answr = int(input(s))
    except ValueError:
        answr = default

    return answr

def ask_str(s: str, file_extension: str, default: str ='temporal') -> str:
    answr = input(s)
    if len(answr) == 0 or len(answr) == file_extension:
        return default + file_extension

    if answr[len(file_extension):] == file_extension:
        return answr
    return answr + file_extension
