import sys
import math, random
from tqdm import tqdm

raw_traces_file = sys.argv[1]

formulas_set = set()

val_dist= {}

with open(raw_traces_file) as raw_trace:
    for line in tqdm(raw_trace.readlines()):
        if ":=" not in line:
            continue
        # if "0x" == line[:2]:continue
        rhs = line.split(":=")[1]
        formula_tokens = rhs.split()
        
        if len(formula_tokens) < 5 or len(formula_tokens) > 286:
            continue
        
        for tok_idx in range(len(formula_tokens)):
            tok = formula_tokens[tok_idx]
            if "BV" in tok:
                formula_tokens[tok_idx] = "bv"
            elif "0x" in tok:
                value = int(tok, 16)
                if value == 0:
                    formula_tokens[tok_idx] = "0"
                elif value <= 64:
                    formula_tokens[tok_idx] = "constant " + str(value)
                else:
                    formula_tokens[tok_idx] = "constant 2e"+str(int(math.log2(value)))
            elif "Reverse" in tok:
                formula_tokens[tok_idx] = "reverse"
            elif "Concat" in tok:
                formula_tokens[tok_idx] = "concat"
            elif "Extract" in tok:
                formula_tokens[tok_idx] = "extract"
            elif "SMod" in tok:
                formula_tokens[tok_idx] = "smod"
            elif "SDiv" in tok:
                formula_tokens[tok_idx] = "sdiv"
            elif "LShR" in tok:
                formula_tokens[tok_idx] = "lshr"
            elif tok.count("__") == 2:
                op = tok[2:]
                end = op.index("__")
                formula_tokens[tok_idx] = op[:end]
            elif tok == "(" or tok == ")" or tok == ",":
                pass
            else:
                print(tok)
                raise NotImplementedError()
        formulas_set.add(" ".join(formula_tokens) + "\n")

data_size = len(formulas_set)
print(data_size)
formulas_list = list(formulas_set)
random.shuffle(formulas_list)

with open("data/trace.txt", "w") as f:
    f.writelines(formulas_list)