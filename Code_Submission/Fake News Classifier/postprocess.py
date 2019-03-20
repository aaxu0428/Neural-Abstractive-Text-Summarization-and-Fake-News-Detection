import pandas as pd
import os
import re

#fake_directory = "/data/home/cs224n/fake_news_data/finished_files/fake_log/decode_model_350000_1551699098/rouge_dec_dir"
fake_directory = "/Volumes/GoogleDrive/My Drive/CS224N_Shared_Folder/GPU_Codes/fake_news_data/finished_files/fake_log/decode_model_350000_1551699098/rouge_dec_dir"
ref_directory = "/Volumes/GoogleDrive/My Drive/CS224N_Shared_Folder/GPU_Codes/fake_news_data/finished_files/fake_log/decode_model_350000_1551699098/rouge_ref"
#ref_directory = "/data/home/cs224n/fake_news_data/finished_files/fake_log/decode_model_350000_1551699098/rouge_ref"
#fake_or_real_path = "/data/home/cs224n/fake_news_data/fake_or_real_news.csv"
fake_or_real_path = "/Volumes/GoogleDrive/My Drive/CS224N_Shared_Folder/GPU_Codes/fake_news_data/fake_or_real_news.csv"

# print("contents of fake_directory", os.walk(fake_directory))
# for dirpath, dirnames, files in os.walk(fake_directory):
    #print("dirpath", dirpath)
file_names = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(fake_directory) for f in files if f.endswith('.txt')]
#print("file_names", file_names)
temp_df = pd.DataFrame()
fake_or_real = pd.read_csv(fake_or_real_path)
print(fake_or_real)

original_initial_texts = []
for i in range(fake_or_real.shape[0]):
    original_initial_texts.append(str(fake_or_real["text"][i])[:100].lower())
fake_or_real["initial_texts"] = original_initial_texts

initial_texts = []
summaries = []

for i in range(len(file_names)):
    with open(file_names[i], 'r') as file:
        name = re.search(fake_directory + "/(.*)_decoded.txt", file_names[i], re.IGNORECASE)
        #print("name", name) # a MATCH object
        if name:
            file_id = name.group(1)
            print("file_id", file_id)
        with open(ref_directory + "/" + str(file_id) + "_reference.txt", 'r') as ref_file:
            initial_texts.append(ref_file.read())
        summaries.append(file.read().replace('\n', ''))

temp_df["initial_texts"] = initial_texts
temp_df["summarized_text"] = summaries

temp_initial_texts = []
for i in range(temp_df.shape[0]):
    temp_initial_texts.append(str(temp_df["initial_texts"][i])[:100].lower())
temp_df["initial_texts"] = temp_initial_texts

merged_df = fake_or_real.merge(temp_df, how = "inner", on = "initial_texts")

alternative_headlines = []
for i in range(merged_df.shape[0]):
    summary_word_list = merged_df["summarized_text"][i].split(" ")
    new_headline = " ".join(summary_word_list[:10])
    alternative_headlines.append(new_headline)

merged_df["alternative_headline"] = alternative_headlines

merged_df.to_csv("merged.csv")

summarized_texts = merged_df["summarized_text"]
total_num_words = 0
for summary in summarized_texts:
	word_list = summary.split(' ')
	total_num_words += len(word_list)
	
print("Average number of words in summary: ", (total_num_words / len(summarized_texts)))
