python -m scripts.eval_indobert --model_name haryoaw/stif-i-f-scratch --csv-path data/stif_indo/test_with_pointing.csv --out_path data/pred/stif-indo_scratch --device cuda
python -m scripts.eval_indobert --model_name haryoaw/stif-i-f-indobart-v2 --csv-path data/stif_indo/test_with_pointing.csv --out_path data/pred/stif-indo_indobart --device cuda

python -m scripts.eval_indobert --model_name haryoaw/lexicol-scratch --csv-path data/stif_indo/test_with_pointing.csv --out_path data/pred/lexicol_scratch --device cuda
python -m scripts.eval_indobert --model_name haryoaw/lexicol-indobart-v2 --csv-path data/stif_indo/test_with_pointing.csv --out_path data/pred/lexicol_indobart --device cuda

