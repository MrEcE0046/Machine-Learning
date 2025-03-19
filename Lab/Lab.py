import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
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

        plt.subplot(2, 3, 3)
        df_filtered = self.df[self.df['active'] == 0]
        feature_plot = df_filtered.groupby(["alco", "smoke", "gender", "active"])["cardio"].mean().reset_index()
        sns.barplot(data=feature_plot, x="gender", y="cardio")
        plt.title("Genomsnittlig hjärt-kärlsjukdomar på kategorisk grupp")
        plt.ylabel("Hjärt-kärlsjukdomar")
        # Visar genomsnittet av hjärt-kärlsjukdomar på sammansatt data. Man kan tydligt se att det är jämnt fördelat mellan män och kvinnor. 

        plt.subplot(2, 3, 4)
        gender = self.df.groupby(["cholesterol"])["cardio"].mean().reset_index()
        sns.barplot(data=gender, x="cholesterol", y="cardio")
        plt.title("Genomsnittliga hjärt-kärlsjukdomar på kolesterålnivåer")

        plt.subplot(2, 3, 5)
        bmi_cardio = self.df.groupby("bp_category")["cardio"].mean().reset_index()
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
    
    def train_test_split(self, X, y):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



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

    def hyper_tuning(self, model_name, X_train, y_train, X_val, y_val, cv=3): # Hyperparameter tuning för att hitta bästa modellen
        models = {
            "logistic_regression": (LogisticRegression(), { # Hyperparametrar för Logistic Regression
                'C': [0.01, 0.1, 0.5, 1, 3, 10, 100 ],
                'solver': ['liblinear','lbfgs'],
                'max_iter': [100, 1000, 2500, 5000, 10000]
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
        model, param_grid = models[model_name] # Hämtar modell och parametrar för specifik modell
        grid_search = GridSearchCV(estimator= model, param_grid= param_grid, cv=cv, scoring="accuracy", verbose=True, n_jobs=-1) # cv=3 för att dela upp datan i 3 delar. Prövade med cv=5 men det tog alldeles för lång tid.
        grid_search.fit(X_train, y_train) # Tränar modellen
        best_model = grid_search.best_estimator_
        self.best_models[model_name] = best_model
    
        print(f"Best model for {model_name}: {grid_search.best_params_}\n")

        y_pred = best_model.predict(X_val)
        report = classification_report(y_val, y_pred)
        print(f"\nClassification report for {model_name}:\n")
        print(report)
        
    def concatenate(self, scaled_X_train, scaled_X_val):
        X_train = pd.DataFrame(scaled_X_train)
        X_val = pd.DataFrame(scaled_X_val)
        return pd.concat([X_train, X_val], axis=0)
    
    def voting_classifier(self, X_train, y_train, X_test, y_test):
        # Skapa en voting classifier
        voting_clf = VotingClassifier(estimators=[('lr', self.best_models["logistic_regression"]), ('rfc', self.best_models["RandomForest"]), ('knn', self.best_models["KNN"])], voting='hard')






















    """-----------------------------------------------------------------------------------"""
    def models(self, X_train, y_train):

        lr = LogisticRegression(max_iter=10000)
        rfc = RandomForestClassifier()
        kmean = KMeans()

        kmean.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        rfc.fit(X_train, y_train)


    def model_with_grid_search(self, X_train, y_train, X_test, y_test, model_type):
        """
        Funktion för att träna en modell med GridSearchCV.
        
        Parameters:
        - X_train: Träningsdata
        - y_train: Träningsetiketter
        - X_test: Testdata
        - y_test: Testetiketter
        - model_type: Typ av modell att träna ('logistic_regression', 'knn', 'random_forest')
        
        Returnerar den bästa modellen baserat på GridSearchCV.
        """
        
        # Skapa en pipeline utan skalning om data redan är standardiserade
        pipe = Pipeline([
            # Här läggs bara modellen till utan skalning
        ])
        
        # Definiera modellen och hyperparametrarna
        if model_type == 'logistic_regression':
            model = LogisticRegression(max_iter=10000)
            param_grid = {
                'log_reg__C': [0.01, 0.1, 0.5, 1, 3, 10, 100],
                'log_reg__solver': ['liblinear', 'lbfgs'],
                # 'log_reg__max_iter': [100, 1000, 2500, 5000, 10000]
            }
            pipe.steps.append(('log_reg', model))
            param_grid = {f'log_reg__{key}': value for key, value in param_grid.items()}
            
        elif model_type == 'knn':
            model = KNeighborsClassifier()
            param_grid = {
                'knn__n_neighbors': [3, 5, 7, 10, 15],
                'knn__weights': ['uniform', 'distance'],
                'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'knn__p': [1, 2]
            }
            pipe.steps.append(('knn', model))
            param_grid = {f'knn__{key}': value for key, value in param_grid.items()}
        
        elif model_type == 'random_forest':
            model = RandomForestClassifier()
            param_grid = {
                'rf__n_estimators': [50, 100, 200],
                'rf__max_depth': [None, 10, 20, 30],
                'rf__min_samples_split': [2, 5, 10],
                'rf__min_samples_leaf': [1, 2, 4],
                'rf__bootstrap': [True, False]
            }
            pipe.steps.append(('rf', model))
            param_grid = {f'rf__{key}': value for key, value in param_grid.items()}
        
        else:
            raise ValueError("Invalid model_type. Must be 'logistic_regression', 'knn', or 'random_forest'.")
        
        # Skapa en GridSearchCV-instans
        grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, verbose=True, n_jobs=-1)

        # Träna modellen med grid search på träningsdata
        grid_search.fit(X_train, y_train)

        # Utskrift av bästa parametrar och bästa estimator
        print(f"Best Parameters for {model_type}: ", grid_search.best_params_)
        print(f"Best Estimator for {model_type}: ", grid_search.best_estimator_)

        # Utvärdera den bästa modellen på testdata
        test_score = grid_search.score(X_test, y_test)
        print(f"{model_type.capitalize()} Test set score: {test_score}")

        return grid_search.best_estimator_

    def param_grids(self):
        lr_grids =[
        {
         'C': [0.01, 0.1, 0.5, 1, 3, 10, 100 ],
         'solver': ['liblinear','lbfgs'],
         'max_iter': [100, 1000, 2500, 5000, 10000]}
        ]
        rfc_grids = [
        {
         'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
         'criterion': ['gini', 'entropy'],
         'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
         'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
         'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        ]
        knn_grids = [
            {
                "n_neighbors": range(1, 21, 2),
                "metric": ["euclidean", "manhattan"]}
        ]

#         plt.figure(figsize=(12, 10))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.kdeplot(data=df, x='age', hue='cardio', shade=True)
# plt.title('Age Distribution by Cardiovascular Disease')
# plt.xlabel('Age (years)')
# plt.show()