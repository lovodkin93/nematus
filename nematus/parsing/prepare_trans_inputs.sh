#!/bin/bash
#SBATCH --mem=256g
#SBATCH -c64
#SBATCH --time=7-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=leshem.choshen@mail.huji.ac.il
#SBATCH --output=/cs/snapless/oabend/borgr/TG/slurm/combine-%j.out

# change those when changing pairs:
working_dir=ar_he/20.07.21
working_dir=en_ko/20.06.07
src=ko
trg=en
trg_name=english
echo "src $src trg $trg $trg_name"
capitalized_langs=(en de fr es)
corpus=/cs/snapless/oabend/borgr/TG/nematus/parsing/corpus.py
choose_lines=/cs/snapless/oabend/borgr/TG/nematus/parsing/choose_lines_by_idx.py
udpipe=/cs/snapless/oabend/borgr/SSMT/udpipe.py
udpipe_models=/cs/snapless/oabend/borgr/SSMT/udpipe_models/udpipe-ud-2.0-170801/
working_dir=/cs/snapless/oabend/borgr/SSMT/preprocess/data/${working_dir}/
files=("train.clean" dev test)
files=("train_en_ko.clean" dev_en_ko.cleaned test_en_ko.cleaned)

echo "working_dir is ${working_dir}"

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
    python $udpipe -i "${working_dir}/${file}.unesc.tok${trg_tc}.${trg}" -o "${working_dir}/UD/${file}.unesc.tok${trg_tc}.${trg}.conllu" -m "${udpipe_models}/${trg_name}-ud-2.0-170801.udpipe"
done

#python3
for file in "${files[@]}"
do
  source /cs/snapless/oabend/borgr/envs/tg/bin/activate
  echo "python3 $corpus --conllu ${working_dir}/UD/${file}.unesc.tok${trg_tc}.${trg}.conllu --bpe ${working_dir}/${file}.unesc.tok${trg_tc}.bpe.${trg} --out ${working_dir}/UD/ --force"
  python3 $corpus --conllu "${working_dir}/UD/${file}.unesc.tok${trg_tc}.${trg}.conllu" --bpe "${working_dir}/${file}.unesc.tok${trg_tc}.bpe.${trg}" --out "${working_dir}/UD/" --force
  echo "python3 $choose_lines -i ${working_dir}/UD/${file}.unesc.tok${trg_tc}.${trg}.conllu.trns1.ids -s ${working_dir}/${file}.unesc.tok${src_tc}.bpe.${src} -o ${working_dir}/UD/${file}.unesc.tok.bpe${src_tc}.${src}"
  python3 $choose_lines -i "${working_dir}/UD/${file}.unesc.tok${trg_tc}.${trg}.conllu.trns1.ids" -s "${working_dir}/${file}.unesc.tok${src_tc}.bpe.${src}" -o "${working_dir}/UD/${file}.unesc.tok.bpe${src_tc}.${src}"
done