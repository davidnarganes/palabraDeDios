import re
import string

def preprocess(biblia):
    # 1. Join in one string
    biblia = '\n'.join(biblia)
    # 5. Lowercase
    biblia = biblia.lower()
    # 6. Split chars
    biblia = list(biblia)
    # 6. Replace chars
    remplace_dict = {
        'á':['a','tilde'],
        'é':['e','tilde'],
        'í':['i','tilde'],
        'ó':['o','tilde'],
        'ú':['u','tilde'],
        'ü':['u','tilde'],
        '\n':['end_line'],
        ' ':['white_space'],
        }
    biblia = [b if not b in remplace_dict else remplace_dict[b] for b in biblia]
    # 8. Flat list
    biblia = [item for sublist in biblia for item in sublist]
    # 9. Replace non-valid chars
    valid_chars = string.ascii_lowercase + string.digits + '.,:;?!()-¡¿ñ'
    valid_chars = list(valid_chars)
    valid_chars.extend(['tilde','dieresis','white_space'])
    biblia = ['<unknown>' if c not in valid_chars else c for c in biblia]
    return biblia

def preprocess_all(biblia_string):
    """
    Function to preprocess a string
    Steps:
        1. Strip strings (e.g. 'moises ' -> 'moises')
        2. Join by '\n'
        3. Clean multiple separation (e.g. 'hey   moises' -> 'hey moises')
        4. Lowercase
        5. Split into characters (e.g. 'moises' into ['m','o','i','s','e','s'])
        6. Clean characters: check dict
        7. 
    """

    # 1. Strip text
    biblia_string = [b.strip() for b in biblia_string if b.strip()]
    # 1. Join in one string
    biblia_string = '\n'.join(biblia_string)
    # 2. Clean tabulation and symbols
    biblia_string = re.sub('\s+',' ', biblia_string)
    biblia_string = re.sub('([?!.])\s([A-Z]\w+|[1-9]+\s)',r'\1\n', biblia_string)
    biblia_string = re.sub('\n\s(([A-Z]\w+|[1-9]+\s))',r'\n\1', biblia_string)
    # 5. Lowercase
    biblia_string = biblia_string.lower()
    # 6. Split chars
    biblia_string = list(biblia_string)
    # 6. Replace chars
    remplace_dict = {
        'á':['a','tilde'],
        'é':['e','tilde'],
        'í':['i','tilde'],
        'ó':['o','tilde'],
        'ú':['u','tilde'],
        'ü':['u','tilde'],
        '\n':['end_line'],
        ' ':['white_space'],
        }
    biblia_string = [b if not b in remplace_dict else remplace_dict[b] for b in biblia_string]
    # 8. Flat list
    biblia_string = [item for sublist in biblia_string for item in sublist]
    # 9. Replace non-valid chars
    valid_chars = string.ascii_lowercase + string.digits + '.,:;?!()-¡¿ñ'
    valid_chars = list(valid_chars)
    valid_chars.extend(['tilde','dieresis','white_space'])
    biblia_string = ['<unknown>' if c not in valid_chars else c for c in biblia_string]
    return biblia_string


if __name__ == "__main__":
    in_filepath = '../../data/Biblia/procesado_1/biblia_no_encabezados.txt'

    with open(in_filepath, 'r', encoding='latin1') as handle:
        biblia_raw = handle.readlines()

    biblia_preprocessed = preprocess_all(biblia_raw)
    chars = set(biblia_preprocessed)