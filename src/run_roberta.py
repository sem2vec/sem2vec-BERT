from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import RobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import pipeline

def train():
    config = RobertaConfig(
        vocab_size=512,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    tokenizer = RobertaTokenizerFast(vocab_file="FoBERT/vocab.json", merges_file="FoBERT/merges.txt")
    model = RobertaForMaskedLM(config=config)
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="./data/constraints.txt",
        block_size=128,
    )

    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer, mlm_probability=0.15
    )
    training_args = TrainingArguments(
        output_dir="./FoBERT",
        overwrite_output_dir=True,
        num_train_epochs=512,
        per_device_train_batch_size=128,
        save_steps=10000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    trainer.save_model("./FoBERT")


def eval():
    fill_mask = pipeline(
        "fill-mask",
        model="./FoBERT",
        tokenizer="./FoBERT"
    )
    formula = "and ( eq ( bv , constant 2e64 ) , eq ( bv , 0 ) , eq ( bv , constant 2e64 ) , not ( <mask> ( bv , constant 2e64 ) ) , eq ( bv , 0 ) , ge ( add ( constant 32 , bv ) , bv ) )"
    for f in fill_mask(formula): print(f)

if __name__ == "__main__":
    train()
    # eval()