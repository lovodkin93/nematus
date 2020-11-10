import locale
import logging
import os
import random
import subprocess
import sys

import pandas as pd
from metrics.chrF.chrFpp import computeChrF

pd.set_option('display.max_columns', None)


def set_logger():
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = set_logger()
NEMATUS = os.path.abspath(__file__ + "/../../")  # "/cs/snapless/oabend/borgr/TG/"
VALID_SCRIPT = os.path.join(NEMATUS, "en-de/scripts/evaluate_seq.sh")
POSTPROCESS_SCRIPT = os.path.join(NEMATUS, "en-de/scripts/postprocess_seq.sh")


def get_model_checkpoint(model_dir, model_name, steps, graceful=False):
    if steps == "best":
        model_name += ".best-valid-script"
    else:
        change = 0
        steps = int(steps)
        while True:
            tmp = f"{model_name}-{steps + change}"
            tmp_index = os.path.join(model_dir, f"{tmp}.index")
            tmp_json = os.path.join(model_dir, f"{tmp}.json")
            if os.path.isfile(tmp_index) and os.path.isfile(tmp_json):
                model_name = tmp
                break
            tmp = f"{model_name}-{steps - change}"
            tmp_index = os.path.join(model_dir, f"{tmp}.index")
            tmp_json = os.path.join(model_dir, f"{tmp}.json")
            if os.path.isfile(tmp_index) and os.path.isfile(tmp_json):
                model_name = tmp
                break
            change += 1000
            if change > steps:
                error = f"model not found {model_name} in dir {model_dir}"
                if graceful:
                    logger.info(error)
                    return ""
                raise ValueError(error)
    model_checkpoint = os.path.join(model_dir, model_name)
    logger.info(f"model checkpoint: {model_checkpoint}")
    return model_name


def create_df(df, output_path, combine=None, force=False):
    if combine is None:
        combine = []
    df = pd.DataFrame(df, columns=["model", "data", "BLEU", "chrf", "source", "target", "checkpoint"])
    if not force and os.path.isfile(output_path):
        combine.append(pd.read_csv(output_path))
    if combine:
        df = pd.concat(combine + [df])
    # logger.critical(f"df before drop {df}")
    df.drop_duplicates(inplace=True)
    df.to_csv(output_path, index=False)
    return df


def shuffle_zip(gen):
    l = list(gen)
    return random.sample(l, len(l))


def main():
    recalculate_output = True
    recalculate_output = False
    # logger = logging.getLogger('validate_all')
    logger.setLevel(logging.DEBUG)
    logger.info("Starting validations...")
    local_run = True
    models = ("0gcn", "sgcn", "seq", "unlung",)
    models = ["bpe256", "0gcn", "parent", "sgcn", "unlung", "unlabeled"]
    # model_names = ["model_seq_trans.npz-60000", "model_seq_trans.npz-74000", "model.npz-60000"]
    model_names = ["model.npz", "model_seq_trans.npz", "model_parent_attn.npz", "model_seq_trans.npz",
                   "model_seq_trans.npz", "model_seq_trans.npz"]
    steps = "90000"
    # german only
    lang_pair_names = ("en-de", "en-de_rev")
    preprocessed_paths = ["en_de/5.8", "en_de/5.8"]
    lang_pairs = [("de", "en"), ("en", "de")]
    shuffled_order = True
    shuffle = shuffle_zip if shuffled_order else lambda x: x
    # all pairs
    lang_pair_names = ("en-ko", "ko-en", "en-ar", "en-he", "ar-he", "en-de", "en-de_rev", "en-ru")
    preprocessed_paths = ["en_ko/20.06.07", "en_ko/20.06.07", "en-ar/20.07.21", "en_he/20.07.21",
                          "ar_he/20.07.21", "en_de/5.8", "en_de/5.8", "en_ru/17.08.20"]
    lang_pairs = [("en", "ko"), ("ko", "en"), ("en", "ar"), ("en", "he"), ("ar", "he"), ("de", "en"), ("en", "de"),
                  ("en", "ru")]
    challenges = ["Books_3000_de-en", "de_reflexive", "de_reflexive_news", "de_particle", "de_particle_news",
                  "en_preposition_stranding", "en_preposition_stranding_news",
                  "en_particle", "en_particle_news", "en_reflexive", "en_reflexive_news", "ko3_relative", "ko_relative"]
    out_dir = os.path.join(NEMATUS, "val_output")
    os.makedirs(out_dir, exist_ok=True)
    out_name = "bleu_scores"
    df_path = os.path.join(out_dir, f"{out_name}.csv")
    last_path = os.path.join(out_dir, f"{out_name}.last.csv")
    tmp_path = os.path.join(out_dir, f"tmp_{out_name}.csv")

    recalculate_output = True
    # steps = 200000
    # lang_pairs = lang_pairs[-1:]
    # lang_pair_names = lang_pair_names[-1:]
    # preprocessed_paths = preprocessed_paths[-1:]

    if os.path.isfile(tmp_path):
        tmp_df = pd.read_csv(tmp_path)
    else:
        tmp_df = None
    if os.path.isfile(last_path):
        last_df = pd.read_csv(last_path)
        tmp_df = pd.concat([last_df, tmp_df])
    else:
        last_df = None
    if os.path.isfile(df_path):
        old_df = pd.read_csv(df_path)
    else:
        old_df = None
    logger.info(f"loaded available results {old_df}")
    logger.info(f"loaded stopped last run results {tmp_df}")
    # challenges = []
    assert len(lang_pairs) == len(lang_pair_names) == len(preprocessed_paths)
    assert len(models) == len(model_names)
    df = []
    # long distance dependencies
    for model, model_name in shuffle(zip(models, model_names)):
        for lang_pair, lang_pair_name, preprocessed_path in shuffle(
                zip(lang_pairs, lang_pair_names, preprocessed_paths)):
            # long distance dependencies not created
            if lang_pair not in [("de", "en"), ("en", "de"), ("ko", "en"), ("en", "ko")]:
                continue

            src, trg = lang_pair
            scripts_dir = os.path.join(NEMATUS, lang_pair_name, "scripts")
            # command = sbatch + os.path.join(scripts_dir, "translate_seq.sh")
            data_dir = f"/cs/snapless/oabend/borgr/SSMT/preprocess/data/{preprocessed_path}/challenge_short/"

            model_dir = os.path.join(scripts_dir, "models", model)
            checkpoint = get_model_checkpoint(model_dir, model_name, steps, graceful=True)
            if not checkpoint:
                logger.warning(f"Model not found {model}")
                continue
            model_file = os.path.join(model_dir, checkpoint + ".index")
            for challenge in shuffle(challenges):
                output_path = os.path.join(NEMATUS, lang_pair_name, "output", model,
                                           f"{challenge}.{checkpoint}.{trg}")
                # if not os.path.isdir(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                ref_path = os.path.join(data_dir, f"{challenge}.tok.{trg}")
                input_path = os.path.join(data_dir, f"{challenge}.bpe.{src}")
                if os.path.isfile(input_path):
                    score = evaluate_model(data_dir, input_path, output_path, model, model_file, ref_path, checkpoint,
                                           src, trg, scripts_dir, local_run, recalculate_output=recalculate_output,
                                           dfs=[tmp_df, old_df])
                    df.append((model, challenge, score[0], score[1], src, trg, checkpoint))
                    create_df(df, tmp_path, [tmp_df])
                else:
                    logger.info(f"Challenge {challenge} not found for {lang_pair}")

    logger.info(f"challenges results: {df}")

    # test sets
    inputs = ["newstest2013", "newstest2014", "newstest2015", "dev", "test", "dev_en_ko.cleaned", "test_en_ko.cleaned",
              "newstest2015-enru", "newstest2014-enru"]
    for model, model_name in shuffle(zip(models, model_names)):
        for lang_pair, lang_pair_name, preprocessed_path in shuffle(
                zip(lang_pairs, lang_pair_names, preprocessed_paths)):
            src, trg = lang_pair
            scripts_dir = os.path.join(NEMATUS, lang_pair_name, "scripts")
            model_dir = os.path.join(scripts_dir, "models", model)
            checkpoint = get_model_checkpoint(model_dir, model_name, steps, graceful=True)
            model_file = os.path.join(model_dir, checkpoint + ".index")
            if not checkpoint:
                logger.warning(f"Model not found {model}")
                continue
            # command = sbatch + os.path.join(scripts_dir, "translate_seq.sh")
            data_dir = f"/cs/snapless/oabend/borgr/SSMT/preprocess/data/{preprocessed_path}"
            for input_path in inputs:
                output_path = os.path.join(NEMATUS, lang_pair_name, "output", model,
                                           f"{input_path}.{model_name}.{trg}")
                ref_path = os.path.join(data_dir, f"{input_path}.ref.{trg}")
                trg_tc = get_tc(trg)
                if not os.path.isfile(ref_path):
                    process_path = os.path.join(data_dir, f"{input_path}.unesc.tok{trg_tc}.bpe.{trg}")
                    if not os.path.isfile(process_path):
                        logger.info(f"Skipping {lang_pair_name} test file not found: {process_path}")
                        continue
                    args = [POSTPROCESS_SCRIPT, process_path, ref_path]
                    logger.info(f"calling {' '.join(args)}")
                    subprocess.call(args)
                    logger.info(f"created, {ref_path} from {process_path}")
                assert os.path.isfile(ref_path), f"ref path not created {ref_path}"
                src_tc = get_tc(src)
                data = os.path.basename(input_path).split(".")[0]  # equivalent to "challenge" for challenge sets
                input_path = f"{input_path}.unesc.tok{src_tc}.bpe.{src}"
                score = evaluate_model(data_dir, input_path, output_path, model, model_file, ref_path, checkpoint, src,
                                       trg, scripts_dir, local_run, recalculate_output=recalculate_output,
                                       dfs=[tmp_df, old_df])
                df.append((model, data, score[0], score[1], src, trg, checkpoint))
                create_df(df, tmp_path, [tmp_df])
    df = create_df(df, output_path=last_path)
    # os.remove(tmp_path)
    logger.info(f"DF:\n{df}")
    # combine with old results
    if old_df is not None:
        df_all = df.merge(old_df.drop_duplicates(), on=["model", "data", "source", "target"],
                          how='outer')
        df_all["BLEU"] = df_all.apply(lambda row: row["BLEU_x"] if not pd.isnull(row["BLEU_x"]) else row["BLEU_y"],
                                      axis=1)
        df_all["chrf"] = df_all.apply(lambda row: row["chrf_x"] if not pd.isnull(row["chrf_x"]) else row["chrf_y"],
                                      axis=1)
        df_all["checkpoint"] = df_all.apply(
            lambda row: row["checkpoint_x"] if not pd.isnull(row["checkpoint_x"]) else row["checkpoint_y"], axis=1)
        # logger.info(f"columns {df_all.columns}")
        df_all.drop(["BLEU_x", "BLEU_y", "chrf_x", "chrf_y", "checkpoint_x", "checkpoint_y"], axis=1, inplace=True)
    else:
        df_all = df

    # logger.info(f"columns {df_all.columns}")
    # df_all.apply(lambda x: logger.info(f"x {x}"), axis=1)

    df_all.drop_duplicates(inplace=True)
    df_all.to_csv(df_path, index=False)
    logger.info(f"DF including past:\n{df_all}")
    # logger.info(f'pretty?\n{df_all.sort_values("BLEU").pivot(columns=["data", "src", "trg"])}')


def get_tc(lang):
    latin_langs = ("en", "de", "ru", "fr", "es")
    return ".tc" if lang in latin_langs else ""


def chrf_scorer(hyp, ref, beta=3, nworder=0, ncorder=6, sentence_level_scores=None):
    with open(ref) as fpRef:
        with open(hyp) as fpHyp:
            try:
                chrf = computeChrF(fpRef, fpHyp, nworder, ncorder, beta, sentence_level_scores=sentence_level_scores)
            except UnboundLocalError as e:
                logger.critical(f"Failed on chrf for {fpRef}, {fpHyp}")
                return 0
            res = chrf[0] * 100
            if isinstance(res, list):
                logger.warning(f"Chrf is list {res}, from chrf {chrf}")
                res = res[0]
            return res


def file_len(fname):
    if not os.path.isfile(fname):
        logger.warning(f"File {fname} not found")
        return 0
    i = 0
    with open(fname) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1


def translate(input_path, output_path, model, checkpoint, scripts_dir,
              local_run, recalculate_output=False, batch_size=""):
    sbatch = "" if local_run else "sbatch"
    command = sbatch + os.path.join(scripts_dir, "translate_seq.sh")
    if os.path.isfile(output_path) and (file_len(output_path) == file_len(input_path)) and not recalculate_output:
        logger.info(f"output path exists, skipping translation of {output_path}")
        return
    elif os.path.isfile(output_path) and not recalculate_output:
        logger.info(f"output path exists, but recalculating as length differs from source")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    args = [command, model, output_path, input_path, checkpoint]
    if batch_size:
        args.append(str(batch_size))
    elif model in ["unlung", "unlabeled"]:
        batch_size = 20
        args.append(str(batch_size))
    elif model in ["sgcn"]:
        batch_size = 5
        args.append(str(batch_size))

    logger.info("calling translation script " + " ".join(args))
    subprocess.call(args)


def evaluate_model(data_dir, input_file, output_path, model, model_file, ref_path, checkpoint, src, trg, scripts_dir,
                   local_run, recalculate_output=False, dfs=None, batch_size="", scorers=(chrf_scorer,)):
    """

    :param data_dir:
    :param input_file:
    :param output_path:
    :param model:
    :param model_file:
    :param ref_path:
    :param checkpoint:
    :param src:
    :param trg:
    :param scripts_dir:
    :param local_run:
    :param recalculate_output:
    :param dfs: list of dfs in which the results might be already found
    :return:
    """
    model_dir = os.path.join(scripts_dir, "models", model)
    input_path = os.path.join(data_dir, f"{input_file}")
    # checked if result in df
    logger.info(f"searching for:{src} {trg} {checkpoint} {input_path} {model}")
    for df in dfs:
        if df is not None:
            same_data = (df["data"] == os.path.basename(input_path)) | (
                    df["data"] == os.path.basename(input_path).split(".")[0])
            same_checkpoint = df["checkpoint"] == os.path.basename(checkpoint)
            same_src = df["source"] == src
            same_trg = df["target"] == trg
            same_model = df["model"] == model
            df_idx = (same_src) & (same_trg) & (same_checkpoint) & (same_data) & (same_model)
            # logger.info(
            #     f" same_src[{same_src[0]} same_trg[{same_trg[0]} same_checkpoint[{same_checkpoint[0]} same_data[{same_data[0]} same_model{same_model[0]}")
            # logger.info(f"df_idx:{(df_idx).any()}\n{df_idx}")
            # logger.info(f"recalculate_output {recalculate_output} {(df_idx).any() and not recalculate_output}")
            if (df_idx).any() and not recalculate_output:
                logger.info(f"skipping {output_path} already exists in old df")
                bleu = df[df_idx]["BLEU"]
                bleu = [bl for bl in bleu.unique() if bl]
                chrf = df[df_idx]["chrf"]
                chrf = [bl for bl in chrf.unique() if bl]

                if bleu and chrf:
                    # assert len(bleu) == 1, bleu
                    bleu = bleu[0]
                    chrf = chrf[0]
                    if bleu and chrf:
                        return bleu, chrf
                logger.info(f"Cancelled due to empty bleu {bleu} or chrf {chrf}")

        # logger.info(f"not found in :{df}")

    score = [0] * (len(scorers) + 1)
    if not os.path.isdir(model_dir) and not recalculate_output:
        logger.info(f"model not found: {model_dir}")
    elif not os.path.isfile(model_file) and not recalculate_output:
        logger.info(f"model checkpoint not found: {model_file}")
    else:
        # translate
        translate(input_path, output_path, model, checkpoint, scripts_dir,
                  local_run, recalculate_output, batch_size)

        # evaluate
        if os.path.isfile(output_path):
            logger.info(f"evaluating {output_path} vs. {ref_path}")
            score = evaluate_file(VALID_SCRIPT, output_path, ref_path)
        if int(score) == 0 and ((not batch_size) or int(batch_size) > 1):
            logger.warning("Failed translation. trying with minimal batch size of 1")
            batch_size = "1"
            translate(input_path, output_path, model, checkpoint, scripts_dir,
                      local_run, recalculate_output, batch_size)
            if os.path.isfile(output_path):
                logger.info(f"evaluating {output_path} vs. {ref_path}")
            score = evaluate_file(VALID_SCRIPT, output_path, ref_path)
            if int(score) == 0:
                logger.critical(f"Failed totally to evaluate {model} on {input_path}")
        score = [score] + [scorer(output_path, ref_path) for scorer in scorers]

    return score


def evaluate_file(valid_script, output_path, ref_path):
    args = [valid_script, output_path, ref_path]
    logger.info(f"calling for evaluation {' '.join(args)}")
    proc = subprocess.Popen(args, stdin=None, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout_bytes, stderr_bytes = proc.communicate()
    encoding = locale.getpreferredencoding()
    stdout = stdout_bytes.decode(encoding=encoding)
    stderr = stderr_bytes.decode(encoding=encoding)

    if len(stderr) > 0:
        logger.info("Validation script wrote the following to standard "
                    "error:\n" + stderr)
    if proc.returncode != 0:
        logger.warning("Validation script failed (returned exit status of "
                       "{}).".format(proc.returncode))
        return None
    try:
        score = float(stdout.split()[0])
    except:
        logger.warning("Validation script output does not look like a score: "
                       "{}".format(stdout))
        return None
    logger.info("Validation script score: {}".format(score))
    return score


if __name__ == '__main__':
    # a = chrf_scorer("/home/leshem/PycharmProjects/lab/TG/nematus/metrics/chrF/example.hyp.en", "/home/leshem/PycharmProjects/lab/TG/nematus/metrics/chrF/example.ref.en")
    # print(a)
    # raise
    main()
