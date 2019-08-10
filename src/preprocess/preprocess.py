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

def preprocess_all(biblia):
    # 1. Strip text
    biblia = [b.strip() for b in biblia if b.strip()]
    # 1. Join in one string
    biblia = '\n'.join(biblia)
    # 2. Clean tabulation and symbols
    biblia = re.sub('\s+',' ', biblia)
    biblia = re.sub('([?!.])\s([A-Z]\w+|[1-9]+\s)',r'\1\n', biblia)
    biblia = re.sub('\n\s(([A-Z]\w+|[1-9]+\s))',r'\n\1', biblia)
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

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result