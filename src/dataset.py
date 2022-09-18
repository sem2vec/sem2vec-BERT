from torch.utils.data import Dataset
from pathlib import Path

from src.tokenizer import load_tokenizer

class FormulaDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = load_tokenizer()

        self.examples = []

        src_files = [str(x) for x in Path("./data/text/").glob("*.txt")]
        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])