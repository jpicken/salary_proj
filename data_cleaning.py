
import pandas as pd 

df = pd.read_csv('data_7-29.csv')

"""
Columns: ['Job Title', 'Salary Estimate', 'Job Description', 'Rating',
          'Company Name', 'Location', 'Headquarters', 'Size', 'Founded',
          'Type of ownership', 'Industry', 'Sector', 'Revenue', 'Competitors']

"""

# Job Title is Clean

# Remove null Salary Estimate values
df = df [ df['Salary Estimate'] != '-1']

df['employer provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided' in x.lower() else 0)
trimmed = df['Salary Estimate'].apply(lambda x: x.replace('Employer Provided Salary:',''))

# Salary Estimate needs cleaning
# format: "$##K - $##K"
salary  = trimmed.apply(lambda x: x.split('(')[0])
salary  = salary.apply(lambda x: x.strip())
cleaned = salary.apply(lambda x: x.replace('K','').replace('$','').replace('\\n',''))

df['min_salary'] = cleaned.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = cleaned.apply(lambda x: int(x.split('-')[1]))
df['average_salary'] = (df.min_salary+df.max_salary)/2

df.to_csv(r"G:\coding\salary_proj\test.csv", index = False)

# Job Description: parse for references of applicable skills

# Company Name, text only

df['company_text'] = df.apply(lambda x: x['Company Name'] if x['Rating'] == -1 else x['Company Name'][:-3], axis = 1)

# Location: parse city and state

# Size: parse into int range

# Industry

# Sector

# Revenue

# Competitors