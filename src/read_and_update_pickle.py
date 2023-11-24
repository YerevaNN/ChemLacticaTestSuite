import pandas as pd
import pickle

with open("pubchem_stats.pkl", 'rb') as pubchem_stats_file:
    pubchem_stats = pickle.load(pubchem_stats_file)
tanim = pd.read_csv("tanimoto_sim.csv")

tan_pd = []
interval = pd.IntervalIndex.from_arrays(tanim.TanimotoSimilarity-0.005, right=tanim.TanimotoSimilarity+0.005, name="SIMILARITY")

tan_series = pd.Series(data=tanim["count"].values, index=interval, name="count")
pubchem_stats["SIMILARITY"] = tan_series

with open('pubchem_stats_new.pkl', 'wb') as handle:
    pickle.dump(pubchem_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
