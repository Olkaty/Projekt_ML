import pandas as pd, numpy as np

df = pd.read_excel("observations.xlsx", engine="openpyxl")
df.columns = df.columns.str.strip()

df.head()
df.info()
df.describe()

def to_num(c):
    return pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False),
                         errors="coerce")

df["_body"]  = to_num("body mass")
df["_metab"] = to_num("metabolic rate")
df["_brain"] = to_num("brain size")

print("n rows:", len(df), "| species:", df["species"].nunique(),
      "| classes:", df["class"].nunique())

for c, lab in [("_body","body mass"), ("_metab","metabolic rate"), ("_brain","brain size")]:
    s = df[c].dropna(); s = s[s > 0]
    ls = np.log10(s)
    print(f"{lab:16s} n={len(s):4d}  median={s.median():.3g}  "
          
          f"min={s.min():.3g}  max={s.max():.3g}")



#---------------------------------------------------------------------- 2



# jeden wiersz = gatunek; tylko cechy numeryczne 
g = df.groupby(["class", "species"])

species_stats = pd.DataFrame({
    "n_rows":      g.size(),
    "n_body":      g["_body"].apply(lambda s: s.notna().sum()),
    "n_metab":     g["_metab"].apply(lambda s: s.notna().sum()),
    "n_brain":     g["_brain"].apply(lambda s: s.notna().sum()),
    "body_median": g["_body"].median(),
    "metab_median":g["_metab"].median(),
    "brain_median":g["_brain"].median(),
}).reset_index()

# log10 wersje — pod ML bezpieczniejsze
for c in ["body_median", "metab_median", "brain_median"]:
    species_stats["log10_" + c] = np.log10(species_stats[c].where(species_stats[c] > 0))

# ile gatunków ma wszystkie 3 cechy
complete = species_stats.dropna(subset=["body_median","metab_median","brain_median"])
print("species total:", len(species_stats))
print("species with all 3 core traits:", len(complete))
print("species with body+metab:", species_stats[["body_median","metab_median"]].dropna().shape[0])
print("species with body+brain:", species_stats[["body_median","brain_median"]].dropna().shape[0])




#----------------------------------------------------------------- 3



print(species_stats.groupby("class")["n_rows"].agg(["count","sum"])
      .sort_values("sum", ascending=False).head(10).to_string())



#-------------------------------------------------------------ile próbek (wierszy) na rodzinę

fam = (df.groupby(["class", "family"])
         .size()
         .reset_index(name="n_rows"))

# ile rodzin ma dana klasa + jak rozłożone są próbki w tych rodzinach
fam_per_class = (fam.groupby("class")
                    .agg(n_families=("family", "nunique"),
                         n_rows=("n_rows", "sum"),
                         fam_median=("n_rows", "median"),
                         fam_min=("n_rows", "min"),
                         fam_max=("n_rows", "max"))
                    .sort_values("n_rows", ascending=False))
print(fam_per_class.head(15).to_string())

print("\n--- 10 rodzin z największą liczbą próbek ---")
print(fam.sort_values("n_rows", ascending=False).head(10).to_string(index=False))



#------------------------------------------------------------mean vr median

import matplotlib.pyplot as plt, seaborn as sns, numpy as np

body = df["_body"].dropna(); body = body[body > 0]
log_body = np.log10(body)

mean   = log_body.mean()
median = log_body.median()
mode   = log_body.mode().iloc[0]

plt.figure(figsize=(7, 4))
sns.histplot(log_body, bins=40, kde=True)
plt.axvline(mean,   color="red",   label=f"Mean  = {mean:.2f}")
plt.axvline(median, color="green", label=f"Median= {median:.2f}")
plt.axvline(mode,   color="blue",  label=f"Mode  = {mode:.2f}")
plt.legend()
plt.title("log10(body mass) — Mean vs Median vs Mode")
plt.xlabel("log10(body mass [kg])")
plt.show()




#-------------------------------------------------------------- bill vr mass

d = df[["class", "_body", "_metab"]].dropna()
d = d[(d["_body"] > 0) & (d["_metab"] > 0)].copy()
d["log_body"]  = np.log10(d["_body"])
d["log_metab"] = np.log10(d["_metab"])

# top 6 klas, resztę zwijamy 
top = d["class"].value_counts().head(6).index
d["class_top"] = np.where(d["class"].isin(top), d["class"], "Other")

plt.figure(figsize=(7, 5))
sns.scatterplot(data=d, x="log_body", y="log_metab",
                hue="class_top", alpha=0.5, s=15)
plt.title("Body mass vs metabolic rate (log–log), kolor = klasa")
plt.xlabel("log10(body mass [kg])")
plt.ylabel("log10(metabolic rate)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()
