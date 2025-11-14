import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------
# Data (use your 10 models as before)
# -----------------------
data = {
    "Model": [
        "ZeroShot_Plain", "ZeroShot_CiteSource", "ZeroShot_FewShot",
        "RAG_k1", "RAG_k3", "RAG_k5",
        "FineTuned_BioBERT_lr5e-6_bs4", "FineTuned_BioBERT_lr5e-6_bs8",
        "FineTuned_FLANT5_lr5e-5_bs4", "FineTuned_FLANT5_lr5e-5_bs8",
    ],
    "QA_EM": [0.393, 0.393, 0.393, 0.842, 0.777, 0.777, 0.956, 0.956, 0.9383, 0.9383],
    "QA_F1": [0.197, 0.197, 0.197, 0.569, 0.491, 0.491, 0.9853, 0.9851, 0.968, 0.9682],
    "ROUGE-L": [0.230, 0.227, 0.200, 0.241, 0.241, 0.238, 0.241, 0.243, 0.2618, 0.2557],
    "BERT-F1": [0.217, 0.211, 0.190, 0.876, 0.874, 0.872, 0.876, 0.873, 0.8818, 0.8782],
    "FK_Grade": [16.217, 16.045, 17.160, 16.19, 16.34, 16.40, 15.96, 15.97, 15.96, 15.97],
}

df = pd.DataFrame(data).set_index("Model")

# Normalize each metric for color scale (except FK, which is inverted)
df_norm = df.copy()
for col in df.columns:
    if col != "FK_Grade":
        df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    else:
        df_norm[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# -----------------------
# Create annotation with real values (like seaborn docs)
# -----------------------
annot = df.copy().round(3).astype(str)
for c in annot.columns:
    annot[c] = df_norm[c].round(2).astype(str) + "\n(" + annot[c] + ")"

# -----------------------
# Plot heatmap
# -----------------------
plt.figure(figsize=(10, 6))
sns.heatmap(
    df_norm,
    annot=annot,
    fmt="",
    cmap="YlOrRd",        # similar to the seaborn docs (orange-red)
    linewidths=0.5,
    linecolor="white",
    cbar=True,
    square=False,         # rectangular like your example
    annot_kws={"size": 7}
)
plt.title("Model and Ablation Performance (Normalized Heatmap)", fontsize=11)
plt.ylabel("")
plt.xlabel("")
plt.tight_layout()
plt.show()