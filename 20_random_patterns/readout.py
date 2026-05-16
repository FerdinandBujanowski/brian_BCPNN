import pandas as pd

tests = ['30_1', '15_1', '15_2', '15_3', '15_4', '10_1', '10_2', '10_3', '5_1', '5_2']

for batch in ['A', 'B']:
    for test in tests:
        path = f'20_random_patterns/tests_{batch}/stats_{test}.csv'
        try:
            df = pd.read_csv(path)
            print(f'File {path} currently has {len(df)} lines.')
        except:
            print(f'File {path} does not exist')