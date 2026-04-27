import pandas as pd
import re

df = pd.read_csv('/tmp/sweep.csv')

def parse_name(name):
    m = re.search(r'bs(\d+)_d(\d+)_l(\d+)_h(\d+)_m(\d+)', name)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(5))
    return None, None, None

df[['bs', 'd', 'm']] = pd.DataFrame(df['name'].apply(parse_name).tolist(), index=df.index)

df = df[['bs', 'd', 'm', 'achieved_tflops', 'achieved_gbps', 'arithmetic_intensity', 'seconds_per_step', 'steps_per_second']]
df = df.sort_values(by=['bs', 'd'])

print(df.to_string(index=False))
