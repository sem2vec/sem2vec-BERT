from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import dill

def train_tokenizer():

    # paths = [str(x) for x in Path("./data/text/").glob("*.txt")]
    paths = ["./data/traces.txt"]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()
    # Customize training
    tokenizer.train(files=paths, vocab_size=512, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    tokenizer.save_model("FoBERT-wwm")

def load_tokenizer(debug=False):
    tokenizer = ByteLevelBPETokenizer(
        "./FoBERT/vocab.json",
        "./FoBERT/merges.txt",
    )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)    
    return tokenizer

if __name__ == "__main__":
    train_tokenizer()
    load_tokenizer(True)
    