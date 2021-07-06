import streamlit as st
import numpy as np
import pandas as pd

import json
from pathlib import Path

p = Path().absolute()
p_log=p/'logs'

flns = list(p_log.glob('*json'))

dfs = []
for file in flns:
    with open(file) as f:
        json_data = pd.json_normalize(json.loads(f.read()))
    dfs.append(json_data)
    
df = pd.concat(dfs, sort=False) # or sort=True depending on your needs
st.write("Here's an overview of training model metadata:")
st.write(df)
