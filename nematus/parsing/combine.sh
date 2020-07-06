#!/bin/bash
#SBATCH --mem=256g
#SBATCH -c16
#SBATCH --time=2-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=/cs/snapless/oabend/borgr/TG/slurm/combine-%j.out

# source /cs/snapless/oabend/borgr/envs/SSMT/bin/activate
# source /cs/snapless/oabend/borgr/envs/pytorch/bin/activate
source /cs/snapless/oabend/borgr/envs/tg/bin/activate
corpus=/cs/snapless/oabend/borgr/TG/nematus/parsing/corpus.py
choose_lines=/cs/snapless/oabend/borgr/TG/nematus/parsing/choose_lines_by_idx.py
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/TG/en-de/output/tmp.clean.unesc.tok.tc.conllu.${trg}" --bpe "/cs/snapless/oabend/borgr/TG/en-de/output/tmp.train.clean.unesc.tok.tc.bpe.${trg}" --out "/cs/snapless/oabend/borgr/TG/en-de/output/"
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD/train.clean.unesc.tok.tc.conllu.${trg}" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/train.clean.unesc.tok.tc.bpe.${trg}" --out "/cs/snapless/oabend/borgr/TG/en-de/output/"
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD/newstest2012.${trg}.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/newstest2012.unesc.tok.tc.bpe.${trg}" --out "/cs/snapless/oabend/borgr/TG/en-de/output/"
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD/newstest2013.${trg}.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/newstest2013.unesc.tok.tc.bpe.${trg}" --out "/cs/snapless/oabend/borgr/TG/en-de/output/"
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD/newstest2014.${trg}.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/newstest2014.unesc.tok.tc.bpe.${trg}" --out "/cs/snapless/oabend/borgr/TG/en-de/output/"
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/UD/newstest2015.${trg}.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_de/5.8/newstest2015.unesc.tok.tc.bpe.${trg}" --out "/cs/snapless/oabend/borgr/TG/en-de/output/"

##hebrew en-he
#src=en
#trg=he
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/train.clean.unesc.tok.$trg.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/train.clean.unesc.tok.bpe.$trg" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/" --force
#python3 $choose_lines -i s/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/train.clean.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/train.clean.unesc.tok.tc.bpe.$src -o s/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/train.clean.unesc.tok.tc.$src
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/dev.unesc.tok.$trg.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/dev.unesc.tok.bpe.$trg" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/"
#python3 $choose_lines -i s/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/dev.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/dev.unesc.tok.tc.bpe.$src -o s/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/dev.clean.unesc.tok.tc.$src
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/test.unesc.tok.$trg.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/test.unesc.tok.bpe.$trg" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/"
#python3 $choose_lines -i s/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/test.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/test.unesc.tok.tc.bpe.$src -o s/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/test.clean.unesc.tok.tc.$src
#
#
##english en-he
#src=he
#trg=en
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/train.clean.unesc.tok.he.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/train.clean.unesc.tok.bpe.he" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/" --force
#python3 $choose_lines -i s/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/train.clean.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/train.clean.unesc.tok.tc.bpe.$src -o s/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/train.clean.unesc.tok.tc.$src
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/dev.unesc.tok.he.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/dev.unesc.tok.bpe.he" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/"
#python3 $choose_lines -i s/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/dev.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/dev.unesc.tok.tc.bpe.$src -o s/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/dev.clean.unesc.tok.tc.$src
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/test.unesc.tok.he.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/test.unesc.tok.bpe.he" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/"
#python3 $choose_lines -i s/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/test.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/test.unesc.tok.tc.bpe.$src -o s/snapless/oabend/borgr/SSMT/preprocess/data/en_he/12.8.19/UD/test.clean.unesc.tok.tc.$src

#
###arabic en-ar
#src=en
#trg=ar
##python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/train.clean.unesc.tok.$trg.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/train.clean.unesc.tok.bpe.$trg" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/" --force
#python3 $choose_lines -i /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/train.clean.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/train.clean.unesc.tok.tc.bpe.$src -o /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/train.clean.unesc.tok.tc.$src
##python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/dev.unesc.tok.$trg.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/dev.unesc.tok.bpe.$trg" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/"
#python3 $choose_lines -i /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/dev.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/dev.unesc.tok.tc.bpe.$src -o /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/dev.clean.unesc.tok.tc.$src
##python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/test.unesc.tok.$trg.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/test.unesc.tok.bpe.$trg" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/"
#python3 $choose_lines -i /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/test.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/test.unesc.tok.tc.bpe.$src -o /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/test.clean.unesc.tok.tc.$src

##english en-ar
src=en
trg=ar
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/train.clean.unesc.tok.${trg}.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/train.clean.unesc.tok.bpe.${trg}" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/" --force
#python3 $choose_lines -i /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/train.clean.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/train.clean.unesc.tok.tc.bpe.$src -o /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/train.clean.unesc.tok.tc.$src
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/dev.unesc.tok.${trg}.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/dev.unesc.tok.bpe.${trg}" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/"
#python3 $choose_lines -i /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/dev.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/dev.unesc.tok.tc.bpe.$src -o /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/dev.clean.unesc.tok.tc.$src
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/test.unesc.tok.${trg}.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/test.unesc.tok.bpe.${trg}" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/" --force
#python3 $choose_lines -i /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/test.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/test.unesc.tok.tc.bpe.$src -o /cs/snapless/oabend/borgr/SSMT/preprocess/data/en-ar/19.06.23/UD/test.clean.unesc.tok.tc.$src



##arabic ar-he
src=ar
trg=he
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/train.clean.unesc.tok.${trg}.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/train.clean.unesc.tok.bpe.${trg}" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/" --force
#python3 $choose_lines -i /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/train.clean.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/train.clean.unesc.tok.bpe.$src -o /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/train.clean.unesc.tok.$src
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/dev.unesc.tok.${trg}.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/dev.unesc.tok.bpe.${trg}" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/"
#python3 $choose_lines -i /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/dev.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/dev.unesc.tok.bpe.$src -o /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/dev.clean.unesc.tok.$src
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/test.unesc.tok.${trg}.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/test.unesc.tok.bpe.${trg}" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/" --force
#python3 $choose_lines -i /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/test.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/test.unesc.tok.bpe.$src -o /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/test.clean.unesc.tok.$src

##hebrew ar-he
src=he
trg=ar
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/train.clean.unesc.tok.${trg}.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/train.clean.unesc.tok.bpe.${trg}" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/" --force
#python3 $choose_lines -i /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/train.clean.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/train.clean.unesc.tok.bpe.$src -o /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/train.clean.unesc.tok.$src
python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/dev.unesc.tok.${trg}.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/dev.unesc.tok.bpe.${trg}" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/"
python3 $choose_lines -i /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/dev.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/dev.unesc.tok.bpe.$src -o /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/dev.clean.unesc.tok.$src
#python3 $corpus --conllu "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/test.unesc.tok.${trg}.conllu" --bpe "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/test.unesc.tok.bpe.${trg}" --out "/cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/" --force
#python3 $choose_lines -i /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/test.unesc.tok.$trg.conllu.trns1.ids -s /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/test.unesc.tok.bpe.$src -o /cs/snapless/oabend/borgr/SSMT/preprocess/data/ar_he/20.07.21/UD/test.clean.unesc.tok.$src

echo "done"