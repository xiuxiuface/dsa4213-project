import pandas as pd
import matplotlib.pyplot as plt

# Load your ablation CSV (update filename)
df = pd.read_csv("results_qa/qa_ablation_study.csv") 

# Example: top-k vs ROUGE-L (for summarization)
summ = df.groupby("Top-k").agg({"ROUGE-L":"mean","Avg_FK_Grade":"mean"}).reset_index()
plt.figure(figsize=(6,3))
plt.plot(summ["Top-k"], summ["ROUGE-L"], marker='o')
plt.title("Top-k vs ROUGE-L")
plt.xlabel("Top-k")
plt.ylabel("ROUGE-L")
plt.grid(True)
plt.tight_layout()
plt.show()

# Example: Top-k vs FK
plt.figure(figsize=(6,3))
plt.plot(summ["Top-k"], summ["Avg_FK_Grade"], marker='o')
plt.title("Top-k vs Avg Flesch-Kincaid Grade")
plt.xlabel("Top-k")
plt.ylabel("Avg FK Grade (higher = harder)")
plt.grid(True)
plt.tight_layout()
plt.show()
