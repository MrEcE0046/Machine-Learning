import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

class cardiovascular_disease:        
        
    def __init__(self, df):
        self.df = df
        self.best_models = {}
        self.best_score = {}

    def bp_categories(self, X, y):
        if y < 80 and X < 120:
            return "Frisk"
        elif y < 80 or 120 <= X < 129:
            return "Förhöjt"
        elif 80 <= y <= 89 or 130 <= X < 139:
            return "Hypertoni grad 1"
        elif 90 <= y <= 119 or 140 <= X < 179:
            return "Hypertoni grad 2"
        else:
            return "Hypertoni kris"
        
    def categorize_bp(self):
        self.df["bp_category"] = self.df.apply(lambda row: self.bp_categories(row["ap_hi"], row["ap_lo"]), axis=1)
        return self.df
    """ Ovan funktion är tagen från GPT. Jag printade in min cell från 
    jupiter och frågade varför jag fick det felmeddelande jag fick. Jag försökte 
    trycka in "ap_hi och "ap_lo" direkt i bp_categories """

    def subplot(self):
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 3, 1)
        sns.countplot(data=self.df, x="bmi_kategori", hue="cardio")
        plt.title("BMI över hjärt-kärlsjukdomar på orörd data")

        plt.subplot(2, 3, 2)
        bmi_cardio = self.df.groupby("bmi_kategori")["cardio"].mean().reset_index()
        sns.barplot(data=bmi_cardio, x="bmi_kategori", y="cardio")
        plt.title("Genomsnittlig hjärt-kärlsjukdomar efter BMI")
        plt.ylabel("Hjärt-kärlsjukdomar")
        plt.xlabel("BMI")

        # plt.subplot(2, 3, 3)
        # df_filtered = self.df[self.df['active'] == 0]
        # feature_plot = df_filtered.groupby(["alco", "smoke", "gender", "active"])["cardio"].mean().reset_index()
        # sns.barplot(data=feature_plot, x="gender", y="cardio")
        # plt.title("Genomsnittlig hjärt-kärlsjukdomar på kategorisk grupp")
        # plt.ylabel("Hjärt-kärlsjukdomar")
        # # Visar genomsnittet av hjärt-kärlsjukdomar på sammansatt data. Man kan tydligt se att det är jämnt fördelat mellan män och kvinnor. 

        plt.subplot(2, 3, 3)
        gender = self.df.groupby(["cholesterol"])["cardio"].mean().reset_index()
        sns.barplot(data=gender, x="cholesterol", y="cardio")
        plt.title("Genomsnittliga hjärt-kärlsjukdomar på kolesterålnivåer")

        plt.subplot(2, 3, 4)
        bmi_cardio = self.df.groupby("bp_category")["cardio"].mean().reset_index()
        sns.barplot(data=bmi_cardio, x="bp_category", y="cardio")
        plt.title("Genomsnittlig hjärt-kärlsjukdomar efter blodtryck")

        plt.ylabel("Hjärt-kärlsjukdomar")
        plt.xlabel("Blodtryck")
        plt.tight_layout()

        return plt.show()
    
    def colormap(self, df):
        plt.figure(figsize=(12, 8)) 
        sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
        plt.show()

    def data_one(self, X, y, dataset_name= "data1"):
        if dataset_name not in self.best_models:
            self.best_models[dataset_name] = {}

        X_train, X_test, X_val, y_train, y_test, y_val = self.train_test_split(X, y)
        scaled_X_train, scaled_X_val, scaled_X_test = self.scaling(X_train, X_val, X_test)

        self.hyper_tuning("logistic_regression", scaled_X_train, y_train, scaled_X_val, y_val, dataset_name=dataset_name)
        self.hyper_tuning("RandomForest", scaled_X_train, y_train, scaled_X_val, y_val, dataset_name=dataset_name)
        self.hyper_tuning("KNN", scaled_X_train, y_train, scaled_X_val, y_val, dataset_name=dataset_name)
        # self.hyper_tuning("ElasticNet", scaled_X_train, y_train, scaled_X_val, y_val, dataset_name=dataset_name, cv=5) # funkar inte så bra för klassifiering
        self.hyper_tuning("NaiveBayes", scaled_X_train, y_train, scaled_X_val, y_val, dataset_name=dataset_name, cv=5)
        
        X_train = self.concatenate(scaled_X_train, scaled_X_val) 
        y_train = self.concatenate(y_train, y_val)

        self.evaluate_on_test(scaled_X_test, y_test, dataset_name=dataset_name)

        y_train = y_train.to_numpy().ravel()

        self.voting_classifier(X_train, y_train, scaled_X_test, y_test, dataset_name=dataset_name)
        
        return scaled_X_test, y_test
    def train_test_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

        return X_train, X_test, X_val, y_train, y_test, y_val

    def scaling(self, X_train, X_val, X_test):

        # Standardisering 
        scaler = StandardScaler()
        std_X_train = scaler.fit_transform(X_train) 
        std_X_val = scaler.transform(X_val)
        std_X_test = scaler.transform(X_test)

        # Normalisering
        normal = MinMaxScaler()
        normal_X_train = normal.fit_transform(std_X_train)
        normal_X_val = normal.transform(std_X_val)
        normal_X_test = normal.transform(std_X_test)

        return normal_X_train, normal_X_val, normal_X_test

    def hyper_tuning(self, model_name, X_train, y_train, X_val, y_val, dataset_name="data1", cv=3): # Hyperparameter tuning för att hitta bästa modellen
        """ Jag fick hjälp av Tobbe i klassen med inledningen av denna funktion """
        models = {
            "logistic_regression": (LogisticRegression(), { # Hyperparametrar för Logistic Regression
                "C": [0.001, 0.1, 1, 3],
                "solver": ['liblinear','lbfgs'],
                "max_iter": [1000, 2500, 5000, 10000],
                "class_weight": ["balanced"]
            }),
            "RandomForest": (RandomForestClassifier(), { # Hyperparametrar för Random Forest
                'n_estimators': [100, 300],
                'criterion': ['gini', 'entropy'],
                'max_depth': [2, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }),
            "KNN": (KNeighborsClassifier(), {
                "n_neighbors": range(1, 21, 2),
                "metric": ["euclidean", "manhattan"]
            }), 
            "ElasticNet": (ElasticNet(), {  
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10], 
                "l1_ratio": [0.1, 0.5, 0.7, 0.9, 1.0],  
                "max_iter": [1000, 5000, 10000]
            }),
            "NaiveBayes": (GaussianNB(), {  
                "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            })
        }

        print(f"Hyperparameter tuning for {model_name}...")
        model, param_grid = models[model_name] 
        grid_search = GridSearchCV(estimator= model, param_grid= param_grid, cv=cv, scoring="accuracy", verbose=True, n_jobs=-1) # cv=3 för att dela upp datan i 3 delar. Prövade med cv=5 men det tog alldeles för lång tid.
        grid_search.fit(X_train, y_train) 

        best_model = grid_search.best_estimator_
        self.best_models[dataset_name][model_name] = best_model # Jag har fått hjälp av GPT för att spara de bästa hyperparametrarna 
    
        y_pred = best_model.predict(X_val)
   
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Accuracy for {model_name}: {accuracy:.4f}\n")
        
    def concatenate(self, scaled_X_train, scaled_X_val):
        X_train = pd.DataFrame(scaled_X_train)
        X_val = pd.DataFrame(scaled_X_val)
        return pd.concat([X_train, X_val], axis=0)

    def evaluate_on_test(self, X_test, y_test, dataset_name="data1"):

        for model_name, best_model in self.best_models[dataset_name].items():
            print(f"Evaluating {model_name}...")

            y_pred = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            print(f"\nClassification Report for {model_name}:\n{report}")
            print("Unika prediktioner:", np.unique(y_pred, return_counts=True))

            print("-" * 100) 
            
    def print_best_models(self):
        print("Bästa modeller och deras hyperparametrar:\n")
        print("-" * 100)
        for model_name, model in self.best_models.items():
            print(f"* {model_name}: {model.get_params()}\n")

    def voting_classifier(self, X_train, y_train, X_test, y_test, dataset_name="data1"):

        vote_clf = VotingClassifier(estimators=[
            ('lr', self.best_models[dataset_name]["logistic_regression"]), 
            ('rfc', self.best_models[dataset_name]["RandomForest"]), 
            ('knn', self.best_models[dataset_name]["KNN"]),
            ("nb", self.best_models[dataset_name]["NaiveBayes"])
            ], voting='hard')

        vote_clf.fit(X_train, y_train)
        y_pred = vote_clf.predict(X_test)
    
        self.best_models[dataset_name]["VotingClassifier"] = vote_clf
        report = classification_report(y_test, y_pred)
        print(f"Evaluating VotingClassifier...\n")
        print(report)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Yes", "No"])
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        ax.set_title(f"VotingClassifier")  
        plt.show()

    def print_confusion_matrix(self, X_test, y_test, dataset_name=None, model_name=None):

        best_model =self.best_models[dataset_name][model_name]

        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Yes", "No"])
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        ax.set_title(f"Cunfusion Matrix")  
        plt.show()
        print(report)