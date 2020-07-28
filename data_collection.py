
import glassdoor_scraper as gs
import pandas

path = r"G:\coding\salary_proj/chromedriver"

## Denver, CO --> 1148170

df = gs.get_jobs('data scientist', 15, False, path)

print(df.iloc[0])
<<<<<<< HEAD

data_path = r"G:\coding\salary_proj"

df.to_csv(path = data_path, index = False)
=======
>>>>>>> d6904845e72118e27196b4ecbb9b8dc17fbebb87
