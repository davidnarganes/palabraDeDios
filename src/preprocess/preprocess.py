import os
import re
import sys
import string

sys.path.append(os.path.join("..","utils"))
from utils import *

def preprocess_func(biblia_string, replace_unknown=True):
    """
    Function to preprocess a string
    Steps:
        1. Strip strings (e.g. "moises " -> "moises")
        2. Join by "\n"
        3. Clean multiple separation (e.g. "hey   moises" -> "hey moises")
        4. Lowercase
        5. Split into characters (e.g. "moises" into ["m","o","i","s","e","s"])
        6. Clean characters: check dict
        7. Make flat lists:
            [["m","o","i","s","e","s"],["j","e","s","u","s"]] -> ["m","o","i","s","e","s","j","e","s","u","s"]
        8. Replace any unknown char by "<unknown>"
    
    Args:
        - string
    
    Returns:
        - list of preprocessed chars
    """

    # 1. Strip text
    biblia_string = [b.strip() for b in biblia_string if b.strip()]
    # 2. Join in one string
    biblia_string = "\n".join(biblia_string)
    # 3. Clean tabulation and symbols
    if replace_unknown:
        biblia_string = re.sub("\s+"," ", biblia_string)
        biblia_string = re.sub("([?!.])\s([A-Z]\w+|[1-9]+\s)",r"\1\n", biblia_string)
        biblia_string = re.sub("\n\s(([A-Z]\w+|[1-9]+\s))",r"\n\1", biblia_string)
    # 4. Lowercase
    biblia_string = biblia_string.lower()
    # 5. Split chars
    biblia_string = list(biblia_string)
    # 6. Replace chars
    remplace_dict = {
        "á":["a","<tilde>"],
        "é":["e","<tilde>"],
        "í":["i","<tilde>"],
        "ó":["o","<tilde>"],
        "ú":["u","<tilde>"],
        "ü":["u","<tilde>"],
        "\n":["<end_line>"],
        " ":["<white_space>"],
        }
    biblia_string = [b if not b in remplace_dict else remplace_dict[b] for b in biblia_string]
    # 7. Flat list
    biblia_string = [item for sublist in biblia_string for item in sublist]
    # 8. Replace non-valid chars
    if replace_unknown:
        valid_chars = string.ascii_lowercase + string.digits + ".,:;?!()-¡¿ñ"
        valid_chars = list(valid_chars)
        valid_chars.extend(["<tilde>","<dieresis>","<white_space>"])
        biblia_string = ["<unknown>" if c not in valid_chars else c for c in biblia_string]
    return biblia_string

def save2tsv(out_filepath, string_preprocessed):
    """
    Function to save to TSV

    Args:
        - `out_filepath` to save
        - `string_preprocessed` a list of str for characters

    Returns:
        - Saves to `out_filepath`
    """
    if type(string_preprocessed) != list:
        raise ValueError("`string_preprocessed must be a list of str")
    with open(out_filepath, "w",  encoding='latin1') as outfile:
        outfile.write("\t".join(string_preprocessed))

if __name__ == "__main__":
    in_filepath = os.path.join("..","..","data","Biblia","procesado_1","biblia_no_encabezados.txt")
    out_directory = os.path.join("..","..","data","Biblia","AA_preprocesado")
    mknewdir(out_directory)
    out_filepath = os.path.join(out_directory, "biblia_preprocessed.txt")

    with open(in_filepath, "r", encoding="latin1") as handle:
        biblia_raw = handle.readlines()

    biblia_preprocessed = preprocess_func(biblia_raw)
    chars = set(biblia_preprocessed)
    print("Used chars:\n%s" % "\n".join(chars))

    save2tsv(out_filepath, biblia_preprocessed)