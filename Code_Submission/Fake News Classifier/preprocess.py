import pandas as pd

fake_df = pd.read_csv("fake_or_real_news.csv")

for i in range(fake_df.shape[0]):
    with open("/home/cs224n/fake_news_data/data/" + str(fake_df["id"][i]) + ".txt", 'w+') as file:
        file.write(str(fake_df["text"][i]))