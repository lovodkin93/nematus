import locale
import logging
import os
import subprocess

import pandas as pd

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)

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
                    print(error)
                    return ""
                raise ValueError(error)
    model_checkpoint = os.path.join(model_dir, model_name)
    print(f"model checkpoint: {model_checkpoint}")
    return model_name


def main():
    print("Starting validations...")
    local_run = True
    models = ("0gcn", "sgcn", "seq", "unlung", "unlabeled")
    models = ["0gcn", "bpe256", "parent"]
    # model_names = ["model_seq_trans.npz-60000", "model_seq_trans.npz-74000", "model.npz-60000"]
    model_names = ["model_seq_trans.npz", "model.npz", "model_parent_attn.npz", ]
    steps = "90000"
    print("Warning, a step might not be a comparable measure")
    # german only
    # lang_pair_names = ("en-de", "en-de_rev", "en-ko", "en-ar", "en-he", "he-ar",)
    # preprocessed_paths = ["en_de/5.8", "en_de/5.8"]
    # lang_pairs = [("de", "en"), ("en", "de")]

    lang_pair_names = ("en-ko", "ko-en", "en-ar", "en-he", "ar-he", "en-de", "en-de_rev")
    preprocessed_paths = ["en_ko/20.06.07", "en_ko/20.06.07", "en-ar/20.07.21", "en_he/20.07.21",
                          "ar_he/20.07.21", "en_de/5.8", "en_de/5.8"]
    lang_pairs = [("en", "ko"), ("ko", "en"), ("en", "ar"), ("en", "he"), ("ar", "he"), ("de", "en"), ("en", "de")]
    challenges = ["de_reflexive_100", "de_reflexive", "de_reflexive_news", "de_particle", "en_preposition_stranding",
                  "en_particle", "en_particle_news", "en_reflexive"]
    # challenges = []
    assert len(lang_pairs) == len(lang_pair_names) == len(preprocessed_paths)
    assert len(models) == len(model_names)
    df = []
    # long distance dependencies
    for model, model_name in zip(models, model_names):

        for lang_pair, lang_pair_name, preprocessed_path in zip(lang_pairs, lang_pair_names, preprocessed_paths):
            # long distance dependencies not created
            if lang_pair not in [("de", "en"), ("en", "de")]:
                continue

            src, trg = lang_pair
            scripts_dir = os.path.join(NEMATUS, lang_pair_name, "scripts")
            model_dir = os.path.join(scripts_dir, "models", model)
            # command = sbatch + os.path.join(scripts_dir, "translate_seq.sh")
            data_dir = f"/cs/snapless/oabend/borgr/SSMT/preprocess/data/{preprocessed_path}/challenge/"
            checkpoint = get_model_checkpoint(model_dir, model_name, steps)
            model_file = os.path.join(model_dir, checkpoint + ".index")
            for challenge in challenges:
                output_path = os.path.join(NEMATUS, lang_pair_name, "output", model,
                                           f"{challenge}.{checkpoint}.{trg}")
                # if not os.path.isdir(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                ref_path = os.path.join(data_dir, f"{challenge}.tok.{trg}")
                input_path = os.path.join(data_dir, f"{challenge}.bpe.{src}")
                score = 0
                score = evaluate_model(data_dir, input_path, output_path, model, model_file, ref_path, checkpoint, src,
                                       trg,
                                       scripts_dir, local_run)
                df.append((model, challenge, score, src, trg, checkpoint))
    print(f"challenges results: {df}")

    # test sets
    inputs = ["newstest2013", "newstest2014", "newstest2015", "dev", "test", "dev_en_ko.cleaned", "test_en_ko.cleaned"]
    for model, model_name in zip(models, model_names):
        for lang_pair, lang_pair_name, preprocessed_path in zip(lang_pairs, lang_pair_names, preprocessed_paths):
            src, trg = lang_pair
            scripts_dir = os.path.join(NEMATUS, lang_pair_name, "scripts")
            # command = sbatch + os.path.join(scripts_dir, "translate_seq.sh")
            model_dir = os.path.join(scripts_dir, "models", model)
            data_dir = f"/cs/snapless/oabend/borgr/SSMT/preprocess/data/{preprocessed_path}"
            checkpoint = get_model_checkpoint(model_dir, model_name, steps, graceful=True)
            model_file = os.path.join(model_dir, checkpoint + ".index")
            for input_path in inputs:
                output_path = os.path.join(NEMATUS, lang_pair_name, "output", model,
                                           f"{input_path}.{model_name}.{trg}")
                ref_path = os.path.join(data_dir, f"{input_path}.ref.{trg}")
                trg_tc = get_tc(trg)
                if not os.path.isfile(ref_path):
                    process_path = os.path.join(data_dir, f"{input_path}.unesc.tok{trg_tc}.bpe.{trg}")
                    if not os.path.isfile(process_path):
                        print(f"Skipping {lang_pair_name} test file not found: {process_path}")
                        continue
                    args = [POSTPROCESS_SCRIPT, process_path, ref_path]
                    print(f"calling {' '.join(args)}")
                    subprocess.call(args)
                    print(f"created, {ref_path} from {process_path}")
                assert os.path.isfile(ref_path), f"ref path not created {ref_path}"
                src_tc = get_tc(src)
                input_path = f"{input_path}.unesc.tok{src_tc}.bpe.{src}"
                score = evaluate_model(data_dir, input_path, output_path, model, model_file, ref_path, checkpoint, src,
                                       trg, scripts_dir, local_run)
                df.append((model, input_path, score, src, trg, checkpoint))
    df = pd.DataFrame(df, columns=["model", "data", "BLEU", "source", "target", "checkpoint"])
    print(df)
    df.to_csv(os.path.join(NEMATUS, "en-de", "output", "bleu_scores.csv"), index=False)


def get_tc(lang):
    latin_langs = ("en", "de")
    return ".tc" if lang in latin_langs else ""


def evaluate_model(data_dir, input_file, output_path, model, model_file, ref_path, checkpoint, src, trg, scripts_dir,
                   local_run, recalculate_output=False):
    sbatch = "" if local_run else "sbatch"
    model_dir = os.path.join(scripts_dir, "models", model)
    command = sbatch + os.path.join(scripts_dir, "translate_seq.sh")
    input_path = os.path.join(data_dir, f"{input_file}")

    if os.path.isfile(output_path) and not recalculate_output:
        print(f"output path exists, skipping {output_path}")
    elif not os.path.isdir(model_dir) and not recalculate_output:
        print(f"model not found: {model_dir}")
    elif not os.path.isfile(model_file) and not recalculate_output:
        print(f"model checkpoint not found: {model_file}")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        args = [command, model, output_path, input_path, checkpoint]
        print("calling " + " ".join(args))
        subprocess.call(args)
    if os.path.isfile(output_path):
        print(f"evaluating {output_path} vs. {ref_path}")
    score = evaluate_file(VALID_SCRIPT, output_path, ref_path)
    return score


def evaluate_file(valid_script, output_path, ref_path):
    args = [valid_script, output_path, ref_path]
    print(f"calling for evaluation {' '.join(args)}")
    proc = subprocess.Popen(args, stdin=None, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout_bytes, stderr_bytes = proc.communicate()
    encoding = locale.getpreferredencoding()
    stdout = stdout_bytes.decode(encoding=encoding)
    stderr = stderr_bytes.decode(encoding=encoding)

    if len(stderr) > 0:
        print("Validation script wrote the following to standard "
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
    print("Validation script score: {}".format(score))
    return score


if __name__ == '__main__':
    main()
