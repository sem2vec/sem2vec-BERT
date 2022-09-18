import pickle, random

if __name__ == "__main__":
    with open("all.pkl", "rb") as f:
        all_pairs = pickle.load(f)

    random.shuffle(all_pairs)
    data_num = len(all_pairs)
    test = all_pairs[:int(0.1*data_num)]
    train = all_pairs[int(0.1*data_num):]
    with open("test.pkl", "wb") as f:
        pickle.dump(test, f)
    with open("train.pkl", "wb") as f:
        pickle.dump(train, f)
    