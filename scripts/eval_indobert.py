import fire
from indobenchmark import IndoNLGTokenizer
from transformers import MBartForConditionalGeneration
from tqdm import tqdm
import pandas as pd
from sacrebleu import corpus_bleu
from evaluate import load
import numpy as np


def main(model_name, csv_path, out_path):

    tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indobart-v2")
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    df = pd.read_csv(csv_path)
    inf_result = []
    for inf in tqdm(df.informal):
        inp_to_model = tokenizer(inf, return_tensors='pt')['input_ids']
        result = model.generate(
            inp_to_model, num_beams=10, max_length=100
        )
        inf_result.append(tokenizer.decode(result[0], skip_special_tokens=True))
    
    df_out = pd.DataFrame({'informal': df.informal, 'formal': df.formal, 'formal_pred': inf_result})
    df_out.to_csv(out_path, index=False)

    # use sacrebleu
    ## detokenize
    print(corpus_bleu(inf_result, [df.formal.to_list()], lowercase=True))
    bertscore = load("bertscore")
    predictions = inf_result
    references = df.formal.to_list()
    results = bertscore.compute(predictions=predictions, references=references, lang="id")
    print(np.mean(results['f1']))


if __name__ == "__main__":
    fire.Fire(main)
