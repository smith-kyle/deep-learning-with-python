import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='darkgrid')
df = sns.load_dataset('iris')

fig = sns.kdeplot(df['sepal_width'], shade=True, color="r")
fig = sns.kdeplot(df['sepal_length'], shade=True, color="b")

plt.show()