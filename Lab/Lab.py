import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

class cardiovascular_disease:        

    def bp_categories(self, X, y):
        if y < 80 and X < 120:
            return "Healthy"
        elif y < 80 and 120 <= X < 129:
            return "Elevated"
        elif 80 <= y <= 89 or 130 <= X < 139:
            return "Hypertoni grad 1"
        elif 90 <= y <= 119 or 140 <= X < 179:
            return "Hypertoni grad 2"
        else:
            return "Hypertoni kris"
        
    def categorize_bp(self, df):
        df["bp_category"] = df.apply(lambda row: self.bp_categories(row["ap_hi"], row["ap_lo"]), axis=1)
        return df
    """ Ovan funktion är tagen från GPT. Jag printade in min cell från 
    jupiter och frågade varför jag fick det felmeddelande jag fick. Jag försökte 
    trycka in "ap_hi och "ap_lo" direkt i bp_categories """

    def subplot(self, df):
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 3, 1)
        sns.countplot(data=df, x="bmi kategori", hue="cardio")
        plt.title("BMI över hjärt-kärlsjukdomar på orörd data")

        plt.subplot(2, 3, 2)
        bmi_cardio = df.groupby("bmi kategori")["cardio"].mean().reset_index()
        sns.barplot(data=bmi_cardio, x="bmi kategori", y="cardio")
        plt.title("Genomsnittlig hjärt-kärlsjukdomar efter BMI")
        plt.ylabel("Hjärt-kärlsjukdomar")
        plt.xlabel("BMI")

        plt.subplot(2, 3, 3)
        df_filtered = df[df['active'] == 0]
        feature_plot = df_filtered.groupby(["alco", "smoke", "gender", "active"])["cardio"].mean().reset_index()
        sns.barplot(data=feature_plot, x="gender", y="cardio")
        plt.title("Genomsnittlig hjärt-kärlsjukdomar på kategorisk grupp")
        plt.ylabel("Hjärt-kärlsjukdomar")
        # Visar genomsnittet av hjärt-kärlsjukdomar på sammansatt data. Man kan tydligt se att det är jämnt fördelat mellan män och kvinnor. 

        plt.subplot(2, 3, 4)
        gender = df.groupby(["cholesterol"])["cardio"].mean().reset_index()
        sns.barplot(data=gender, x="cholesterol", y="cardio")
        plt.title("Genomsnittliga hjärt-kärlsjukdomar på kolesterålnivåer")

        plt.subplot(2, 3, 5)
        bmi_cardio = df.groupby("bp_category")["cardio"].mean().reset_index()
        sns.barplot(data=bmi_cardio, x="bp_category", y="cardio")
        plt.title("Genomsnittlig hjärt-kärlsjukdomar efter blodtryck")

        plt.ylabel("Hjärt-kärlsjukdomar")
        plt.xlabel("Blodtryck")

        plt.tight_layout()
        return plt.show()
    
    def colormap(self, df):

        cor = df.corr()
        plt.figure(figsize=(12, 8))
        cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True) 
        sns.heatmap(cor, cmap="Blues", annot=True)

        plt.title('Correlation Heatmap', fontsize=16)
        return plt.show()