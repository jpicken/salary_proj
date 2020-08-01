
import glassdoor_scraper as gs
import pandas

path = r"G:\coding\salary_proj/chromedriver"

## Denver, CO --> 1148170

df = gs.get_jobs('data scientist', 1000, False, path, 6)

print(df.iloc[0])

data_path = r"G:\coding\salary_proj\data_new.csv"

df.to_csv(data_path, index = False)
