import pandas_profiling as pdp
import pandas as pd

df = pd.read_csv('data(bind_9000).csv')
profile = pdp.ProfileReport(df)
profile.to_file(output_file='data(bind_9000).html')
