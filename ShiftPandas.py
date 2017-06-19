import pandas as pd

print('before shift')

date = pd.date_range('1/1/2011', periods=5, freq='H')

df = pd.DataFrame({'cat': ['A', 'A', 'A', 'B', 'B']}, index=date)
print(df)

print('after shift')

df['shifted'] = df.index.shift(-1, freq='H')

print(df)



