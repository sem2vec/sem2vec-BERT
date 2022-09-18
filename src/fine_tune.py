from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import pickle,math
from numpy import dot
from numpy.linalg import norm

model_path = "./FoBERT/checkpoint-290000"
batch_size = 16
num_epoch = 64

def load_data():
    with open("./data/pair/test.pkl", "rb") as f:
        test = pickle.load(f)
    with open("./data/pair/train.pkl", "rb") as f:
        train = pickle.load(f)
    test_samples = []
    for pair in test:
        label = 1.0 if pair[2] else 0.0
        inp_exp = InputExample(texts=[pair[0][:510], pair[1][:510]], label=label)
        test_samples.append(inp_exp)
    train_samples = []
    for pair in train:
        label = 1.0 if pair[2] else 0.0
        inp_exp = InputExample(texts=[pair[0][:510], pair[1][:510]], label=label)
        train_samples.append(inp_exp)
    return train_samples, test_samples


def finetune():
    wd_model = models.Transformer(model_path)
    pooling_model = models.Pooling(wd_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[wd_model, pooling_model])

    train_loss = losses.CosineSimilarityLoss(model=model)
    train_samples, test_samples = load_data()
    # train_samples += load_data2()
    print(f"Num of train: {len(train_samples)}")
    train_data_loader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-dev')
    warmup_steps = math.ceil(len(train_data_loader) * num_epoch  * 0.1) #10% of train data for warm-up
    print(f"Warmup steps: {warmup_steps}")
    model.fit(train_objectives=[(train_data_loader, train_loss)],
          evaluator=evaluator,
          epochs=num_epoch,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path="./FoBERT-SS")

def run():
    model = SentenceTransformer("./FoBERT-SS")
    sentences = ['and ( eq ( bv , add ( constant 2e64 , mul ( constant 2e64 , bv ) ) ) , eq ( extract ( constant 55 , constant 48 , bv ) , constant 2e6 ) , eq ( reverse ( concat ( extract ( constant 63 , constant 8 , reverse ( bv ) ) , extract ( constant 63 , constant 56 , bv ) ) ) , sub ( bv , constant 1 ) ) )']
    sentence_embedding = model.encode(sentences)[0]
    print(sentence_embedding)

if __name__ == "__main__":
    finetune()
    #run()
