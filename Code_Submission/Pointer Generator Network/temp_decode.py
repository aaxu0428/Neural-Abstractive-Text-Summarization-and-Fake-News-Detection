from rouge import Rouge
import os
import numpy as np, scipy.stats as st
from pathlib2 import Path
import re

def temp_decode(rouge_ref_dir, rouge_dec_dir):
    # rouge_ref_dir = "/data/home/gxpeh/1/finished_files/log/decode_model_4000_1551496645/rouge_ref"
    # rouge_ref_dir = "/Volumes/GoogleDrive/My Drive/CS224N_Shared_Folder/GPU_Runs/decode_model_4000_1551496645/rouge_ref"

    # rouge_dec_dir = "/data/home/gxpeh/1/finished_files/log/decode_model_4000_1551496645/rouge_dec_dir"
    # rouge_dec_dir = "/Volumes/GoogleDrive/My Drive/CS224N_Shared_Folder/GPU_Runs/decode_model_4000_1551496645/rouge_dec_dir"

    rouge_ref_files = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(rouge_ref_dir)
        for f in files if f.endswith('.txt')]

    num_files = len(rouge_ref_files)

    rouge1_f = []
    rouge1_p = []
    rouge1_r = []
    rouge2_f = []
    rouge2_p = []
    rouge2_r = []
    rougeL_f = []
    rougeL_p = []
    rougeL_r = []

    rouge = Rouge()
    for i in range(len(rouge_ref_files)):
        ref_contents = Path(rouge_ref_files[i]).read_text()
        title_search = re.search(rouge_ref_dir + '(.*)_reference.txt', rouge_ref_files[i], re.IGNORECASE)
        if title_search:
            title = title_search.group(1)
        decode_contents = Path(rouge_dec_dir + "/" + title + "_decoded.txt").read_text()

        scores = rouge.get_scores(decode_contents, ref_contents)[0]
        rouge1_f.append(scores["rouge-1"]["f"])
        rouge1_p.append(scores["rouge-1"]["p"])
        rouge1_r.append(scores["rouge-1"]["r"])

        rouge2_f.append(scores["rouge-2"]["f"])
        rouge2_p.append(scores["rouge-2"]["p"])
        rouge2_r.append(scores["rouge-2"]["r"])

        rougeL_f.append(scores["rouge-l"]["f"])
        rougeL_p.append(scores["rouge-l"]["p"])
        rougeL_r.append(scores["rouge-l"]["r"])

        # if i % 1000: 
        #     print(str((i * 100) / num_files) + "% complete")

    rouge1_f = np.array(rouge1_f)
    rouge1_p = np.array(rouge1_p)
    rouge1_r = np.array(rouge1_r)
    rouge2_f = np.array(rouge2_f)
    rouge2_p = np.array(rouge2_p)
    rouge2_r = np.array(rouge2_r)
    rougeL_f = np.array(rougeL_f)
    rougeL_p = np.array(rougeL_p)
    rougeL_r = np.array(rougeL_r)

    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), st.sem(a)
        h = se * st.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h

    CI_triple = mean_confidence_interval(rouge1_f)
    print("rouge1_f: ") 
    print("mean: " + str(CI_triple[0]) + " confidence_interval: (" + str(CI_triple[1]) + "," + str(CI_triple[2]) + ")")

    CI_triple = mean_confidence_interval(rouge1_p)
    print("rouge1_p: ") 
    print("mean: " + str(CI_triple[0]) + " confidence_interval: (" + str(CI_triple[1]) + "," + str(CI_triple[2]) + ")")

    CI_triple = mean_confidence_interval(rouge1_r)
    print("rouge1_r: ") 
    print("mean: " + str(CI_triple[0]) + " confidence_interval: (" + str(CI_triple[1]) + "," + str(CI_triple[2]) + ")")

    CI_triple = mean_confidence_interval(rouge2_f)
    print("rouge2_f: ") 
    print("mean: " + str(CI_triple[0]) + " confidence_interval: (" + str(CI_triple[1]) + "," + str(CI_triple[2]) + ")")

    CI_triple = mean_confidence_interval(rouge2_p)
    print("rouge2_p: ") 
    print("mean: " + str(CI_triple[0]) + " confidence_interval: (" + str(CI_triple[1]) + "," + str(CI_triple[2]) + ")")

    CI_triple = mean_confidence_interval(rouge2_r)
    print("rouge2_r: ") 
    print("mean: " + str(CI_triple[0]) + " confidence_interval: (" + str(CI_triple[1]) + "," + str(CI_triple[2]) + ")")

    CI_triple = mean_confidence_interval(rougeL_f)
    print("rougeL_f: ") 
    print("mean: " + str(CI_triple[0]) + " confidence_interval: (" + str(CI_triple[1]) + "," + str(CI_triple[2]) + ")")

    CI_triple = mean_confidence_interval(rougeL_p)
    print("rougeL_p: ") 
    print("mean: " + str(CI_triple[0]) + " confidence_interval: (" + str(CI_triple[1]) + "," + str(CI_triple[2]) + ")")

    CI_triple = mean_confidence_interval(rougeL_r)
    print("rougeL_r: ") 
    print("mean: " + str(CI_triple[0]) + " confidence_interval: (" + str(CI_triple[1]) + "," + str(CI_triple[2]) + ")")
