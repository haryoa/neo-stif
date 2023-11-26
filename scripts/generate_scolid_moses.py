

def main():
    import pandas as pd

    df_train = pd.read_csv("data/scolid/train.csv")
    df_dev = pd.read_csv("data/scolid/val.csv")
    df_test = pd.read_csv("data/scolid/test.csv")

    # output df_train.formal to train.for and train.informal to train.inf
    # also do for dev and test
    # output them to data/scolid/moses
    # mkdir
    import os
    os.makedirs("data/scolid/moses", exist_ok=True) 
    df_train.formal.to_csv("data/scolid/moses/train.for", index=False, header=False)
    df_train.informal.to_csv("data/scolid/moses/train.inf", index=False, header=False)
    df_dev.formal.to_csv("data/scolid/moses/dev.for", index=False, header=False)
    df_dev.informal.to_csv("data/scolid/moses/dev.inf", index=False, header=False)
    df_test.formal.to_csv("data/scolid/moses/test.for", index=False, header=False)
    df_test.informal.to_csv("data/scolid/moses/test.inf", index=False, header=False)


if __name__ == "__main__":
    main()