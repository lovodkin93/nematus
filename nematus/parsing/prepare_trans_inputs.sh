#!/bin/bash
#SBATCH --mem=256g
#SBATCH -c16
#SBATCH --time=14-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=/cs/snapless/oabend/borgr/TG/slurm/combine-%j.out

# change those when changing pairs:
#ud_dir=${working_dir}/UD
#
#working_dir=ar_he/20.07.21

#working_dir=en_ar/20.07.21
#working_dir=/cs/snapless/oabend/borgr/SSMT/preprocess/data/${working_dir}/
#src=en
#trg=ar
#trg_name=arabic-padt
#files=("train.clean" dev test)
#udpipe_models=/cs/snapless/oabend/borgr/SSMT/udpipe_models/udpipe-ud-2.5-191206/
#ud_dir=${working_dir}/UD2.5

src=en
trg=ru
working_dir=en_ru/17.08.20
working_dir=/cs/snapless/oabend/borgr/SSMT/preprocess/data/${working_dir}/
#UD2
ud_dir=${working_dir}/UD
udpipe_models=/cs/snapless/oabend/borgr/SSMT/udpipe_models/udpipe-ud-2.0-170801/
trg_name=russian
#UD2.5
ud_dir=${working_dir}/UD2.5
udpipe_models=/cs/snapless/oabend/borgr/SSMT/udpipe_models/udpipe-ud-2.5-191206/
trg_name="russian-syntagrus"
#files=("train.clean" dev test)
#files=("train_en_ko.clean" dev_en_ko.cleaned test_en_ko.cleaned)
files=(newstest2013 newstest2014-ruen newstest2015-enru newstest2015-ruen newstest2016-enru newstest2016-ruen newstest2017-enru newstest2017-ruen newstest2018-enru newstest2018-ruen newstest2019-enru newstest2019-ruen)
files+=("0.train.clean" "1.train.clean" "2.train.clean" "3.train.clean" "4.train.clean" "5.train.clean" "6.train.clean" "7.train.clean" "8.train.clean" "9.train.clean" )
files=("0.train.clean" "4.train.clean" )

#src=de
#trg=en
##trg_name=english
#trg_name=english-gum
#working_dir=en_de/5.8
##UD2.5
#working_dir=/cs/snapless/oabend/borgr/SSMT/preprocess/data/${working_dir}/
#ud_dir=${working_dir}/UD2.5
#udpipe_models=/cs/snapless/oabend/borgr/SSMT/udpipe_models/udpipe-ud-2.5-191206/
#files=(newstest2012 newstest2013 newstest2014 newstest2015)
#files+=("train.clean")



echo "src $src trg $trg udmodel $trg_name"
capitalized_langs=(en de fr es ru)
corpus=/cs/snapless/oabend/borgr/TG/nematus/parsing/corpus.py
choose_lines=/cs/snapless/oabend/borgr/TG/nematus/parsing/choose_lines_by_idx.py
udpipe=/cs/snapless/oabend/borgr/SSMT/udpipe.py


echo "working_dir is ${working_dir}"
echo "writing UD to " ${ud_dir}
#python2
source /cs/snapless/oabend/borgr/envs/py2/bin/activate

if [[ " ${capitalized_langs[@]} " =~ " ${trg} " ]]; then
  trg_tc=".tc"
else
  trg_tc=""
fi
if [[ " ${capitalized_langs[@]} " =~ " ${src} " ]]; then
  src_tc=".tc"
else
  src_tc=""
fi

for file in "${files[@]}"
do
    # language has truecasing
#    python $udpipe -i "${working_dir}/${file}.unesc.tok${trg_tc}.${trg}" -o "${ud_dir}/${file}.unesc.tok${trg_tc}.${trg}.conllu" -m "${udpipe_models}/${trg_name}-ud-2.0-170801.udpipe"
    python $udpipe -i "${working_dir}/${file}.unesc.tok${trg_tc}.${trg}" -o "${ud_dir}/${file}.unesc.tok${trg_tc}.${trg}.conllu" -m "${udpipe_models}/${trg_name}-ud-2.5-191206.udpipe"
done

#python3
for file in "${files[@]}"
do
  source /cs/snapless/oabend/borgr/envs/tg/bin/activate
  echo "python3 $corpus --conllu ${ud_dir}/${file}.unesc.tok${trg_tc}.${trg}.conllu --bpe ${working_dir}/${file}.unesc.tok${trg_tc}.bpe.${trg} --out ${ud_dir}/ --force"
  python3 $corpus --conllu "${ud_dir}/${file}.unesc.tok${trg_tc}.${trg}.conllu" --bpe "${working_dir}/${file}.unesc.tok${trg_tc}.bpe.${trg}" --out "${ud_dir}/" --force
  echo "python3 $choose_lines -i ${ud_dir}/${file}.unesc.tok${trg_tc}.${trg}.conllu.trns1.ids -s ${working_dir}/${file}.unesc.tok${src_tc}.bpe.${src} -o ${ud_dir}/${file}.unesc.tok.bpe${src_tc}.${src}"
  python3 $choose_lines -i "${ud_dir}/${file}.unesc.tok${trg_tc}.${trg}.conllu.trns1.ids" -s "${working_dir}/${file}.unesc.tok${src_tc}.bpe.${src}" -o "${ud_dir}/${file}.unesc.tok.bpe${src_tc}.${src}"
done