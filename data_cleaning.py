
import pandas as pd 

df = pd.read_csv('data_7-30.csv')

"""
Columns: ['Job Title', 'Salary Estimate', 'Job Description', 'Rating',
          'Company Name', 'Location', 'Headquarters', 'Size', 'Founded',
          'Type of ownership', 'Industry', 'Sector', 'Revenue', 'Competitors']

"""

# Job Title is Clean

# Remove null Salary Estimate values
df = df [ df['Salary Estimate'] != '-1']

df['employer provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided' in x.lower() else 0)
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
trimmed = df['Salary Estimate'].apply(lambda x: x.replace('Employer Provided Salary:','').replace(' Per Hour',''))

# Salary Estimate needs cleaning
# format: "$##K - $##K"
salary  = trimmed.apply(lambda x: x.split('(')[0])
salary  = salary.apply(lambda x: x.strip())
cleaned = salary.apply(lambda x: x.replace('K','').replace('$','').replace('\\n',''))

df['min_salary'] = cleaned.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = cleaned.apply(lambda x: int(x.split('-')[1]))
df['average_salary'] = (df.min_salary+df.max_salary)/2



# Company Name, text only

df['company_text'] = df.apply(lambda x: x['Company Name'] if x['Rating'] == -1 else x['Company Name'][:-3], axis = 1).apply(lambda x: x.strip())

# Location: parse city and state
df['job_state'] = df['Location'].apply(lambda x: x.split(', ')[1])
df['job_town'] = df['Location'].apply(lambda x: x.split(', ')[0])

print(df.job_state.value_counts())
print(df.job_town.value_counts())

# Age of company

df['age'] = df['Founded'].apply(lambda x: x if x < 1 else 2020 - x)

# Job Description: parse for references of applicable skills

# SQL
df['sql_yn'] = df['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)

# Python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)

# Excel
df['excel_yn'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)

# machine learning
df['ML_yn'] = df['Job Description'].apply(lambda x: 1 if 'machine learning' in x.lower() else 0)

# tableau
df['tableau_yn'] = df['Job Description'].apply(lambda x: 1 if 'tableau' in x.lower() else 0)

# Tensorflow
df['tensorflow_yn'] = df['Job Description'].apply(lambda x: 1 if 'tensorflow' in x.lower() else 0)

# AWS
df['aws_yn'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

# Spark
df['spark_yn'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

# NLP
df['nlp_yn'] = df['Job Description'].apply(lambda x: 1 if 'nlp' in x.lower() else 0)

df.sql_yn.value_counts()
df.python_yn.value_counts()
df.excel_yn.value_counts()
df.ML_yn.value_counts()
df.tableau_yn.value_counts()
df.tensorflow_yn.value_counts()
df.aws_yn.value_counts()
df.spark_yn.value_counts()
df.nlp_yn.value_counts()

# remember to update output to match your input
df.to_csv(r"G:\coding\salary_proj\cleaned_data_7-30.csv", index = False)

# Industry: need to separate into individual strings?

# Sector

# Revenue: relevant to salary? --> cant use, too many unknown values
