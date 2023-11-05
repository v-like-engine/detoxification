import seaborn as sns
import pandas as pd
import zipfile
import io
import matplotlib.pyplot as plt

from src.data.make_dataset import unzip_tsv, bring_toxic_to_one_col

FILEPATH = '../../data/raw/filtered_paranmt.zip'


df = unzip_tsv()
print(df.info())
sns.violinplot(data=df, x='ref_tox')
plt.savefig('violin_plot_ref_unprocessed.png')
plt.clf()
sns.violinplot(data=df, x='trn_tox', title='')
plt.savefig('violin_plot_trn_unprocessed.png')
plt.clf()
df = bring_toxic_to_one_col(df)
print(df.info())
sns.violinplot(data=df, x='toxic_tox')
plt.savefig('../../reports/figures/violin_plot_toxic.png')
plt.clf()
sns.violinplot(data=df, x='neutral_tox')
plt.savefig('../../reports/figures/violin_plot_neutral.png')
plt.clf()
# Scatter plot
sns.scatterplot(data=df, x='similarity', y='lenght_diff')
plt.savefig('../../reports/figures/scatter_plot.png')
plt.clf()

sns.boxplot(x="toxic_tox", y="neutral_tox", data=df.iloc[:20000], palette='Set1')
plt.savefig('../../reports/figures/boxplot_of_toxicity.png')
plt.clf()
sns.pairplot(df.iloc[:10000])
plt.savefig('../../reports/figures/pairplot.png')
