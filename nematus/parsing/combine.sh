#!/bin/bash
#SBATCH --mem=48g
#SBATCH -c16
#SBATCH --time=2-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=/cs/snapless/oabend/borgr/TG/slurm/combine-%j.out

# source /cs/snapless/oabend/borgr/envs/SSMT/bin/activate
# source /cs/snapless/oabend/borgr/envs/pytorch/bin/activate
source /cs/snapless/oabend/borgr/envs/tf15/bin/activate

python3 /cs/snapless/oabend/borgr/TG/nematus/parsing/corpus.py --conllu "/cs/snapless/oabend/borgr/TG/en-de/output/tmp.clean.unesc.tok.tc.conllu.en" --bpe "/cs/snapless/oabend/borgr/TG/en-de/output/tmp.train.clean.unesc.tok.tc.bpe.en" --out "/cs/snapless/oabend/borgr/TG/en-de/output/"
python3 /cs/snapless/oabend/borgr/TG/nematus/parsing/corpus.py --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD/train.clean.unesc.tok.tc.conllu.en" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/train.clean.unesc.tok.tc.bpe.en" --out "/cs/snapless/oabend/borgr/TG/en-de/output/"
python3 /cs/snapless/oabend/borgr/TG/nematus/parsing/corpus.py --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD/newstest2012.en.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/newstest2012.unesc.tok.tc.bpe.en" --out "/cs/snapless/oabend/borgr/TG/en-de/output/"
python3 /cs/snapless/oabend/borgr/TG/nematus/parsing/corpus.py --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD/newstest2013.en.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/newstest2013.unesc.tok.tc.bpe.en" --out "/cs/snapless/oabend/borgr/TG/en-de/output/"
python3 /cs/snapless/oabend/borgr/TG/nematus/parsing/corpus.py --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD/newstest2014.en.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/newstest2014.unesc.tok.tc.bpe.en" --out "/cs/snapless/oabend/borgr/TG/en-de/output/"
python3 /cs/snapless/oabend/borgr/TG/nematus/parsing/corpus.py --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD/newstest2015.en.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/newstest2015.unesc.tok.tc.bpe.en" --out "/cs/snapless/oabend/borgr/TG/en-de/output/"
echo done