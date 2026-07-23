import os
import yaml
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sn
import matplotlib.pyplot as plt


class DiseasePrediction:
    def __init__(self, model_name=None):
        try:
            with open('./config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Error reading config.yaml: {e}")

        self.verbose = self.config['verbose']
        self.model_name = model_name
        self.model_save_path = self.config['model_save_path']

        self.train_features, self.train_labels, self.train_df = self._load_train_dataset()
        self.test_features, self.test_labels, self.test_df = self._load_test_dataset()
        self._feature_correlation(data_frame=self.train_df, show_fig=False)

    def _load_train_dataset(self):
        df_train = pd.read_csv(self.config['dataset']['training_data_path'])
        cols = df_train.columns[:-2]
        train_features = df_train[cols]
        train_labels = df_train['prognosis']

        assert (len(train_features.iloc[0]) == 132)
        assert (len(train_labels) == train_features.shape[0])

        if self.verbose:
            print("Length of Training Data: ", df_train.shape)
            print("Training Features: ", train_features.shape)
        return train_features, train_labels, df_train

    def _load_test_dataset(self):
        df_test = pd.read_csv(self.config['dataset']['test_data_path'])
        cols = self.train_features.columns
        test_features = df_test[cols]
        test_labels = df_test['prognosis']

        assert (len(test_features.iloc[0]) == 132)
        assert (len(test_labels) == test_features.shape[0])

        if self.verbose:
            print("Length of Test Data: ", df_test.shape)
            print("Test Features: ", test_features.shape)
        return test_features, test_labels, df_test

    def _feature_correlation(self, data_frame=None, show_fig=False):
        numeric_df = data_frame.select_dtypes(include=['int64', 'float64'])
        corr = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sn.heatmap(corr, square=True, annot=False, cmap="YlGnBu")
        plt.title("Feature Correlation")
        plt.tight_layout()
        if show_fig:
            plt.show()
        plt.savefig('feature_correlation.png')
        plt.close()

    def _train_val_split(self):
        X_train, X_val, y_train, y_val = train_test_split(
            self.train_features, 
            self.train_labels,
            test_size=self.config['dataset']['validation_size'],
            random_state=self.config['random_state']
        )
        return X_train, y_train, X_val, y_val

    def select_model(self):
        if self.model_name == 'mnb':
            self.clf = MultinomialNB()
        elif self.model_name == 'decision_tree':
            self.clf = DecisionTreeClassifier(criterion=self.config['model']['decision_tree']['criterion'])
        elif self.model_name == 'random_forest':
            self.clf = RandomForestClassifier(n_estimators=self.config['model']['random_forest']['n_estimators'])
        elif self.model_name == 'gradient_boost':
            self.clf = GradientBoostingClassifier(
                n_estimators=self.config['model']['gradient_boost']['n_estimators'],
                criterion=self.config['model']['gradient_boost']['criterion']
            )
        else:
            raise ValueError(f"Invalid model_name: {self.model_name}")
        return self.clf

    def train_model(self):
        X_train, y_train, X_val, y_val = self._train_val_split()
        classifier = self.select_model()
        classifier = classifier.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_val, classifier.predict(X_val))
        
        if self.verbose:
            print('\nValidation Accuracy: ', accuracy)

        os.makedirs(self.model_save_path, exist_ok=True)
        dump(classifier, os.path.join(self.model_save_path, f"{self.model_name}.joblib"))

    def make_prediction(self, saved_model_name=None, test_data=None):
        model_path = os.path.join(self.model_save_path, f"{saved_model_name}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        clf = load(model_path)
        if test_data is not None:
            return clf.predict(test_data)
        
        result = clf.predict(self.test_features)
        accuracy = accuracy_score(self.test_labels, result)
        clf_report = classification_report(self.test_labels, result)
        return accuracy, clf_report


if __name__ == "__main__":
    current_model_name = 'decision_tree'
    dp = DiseasePrediction(model_name=current_model_name)
    dp.train_model()
    test_accuracy, classification_report = dp.make_prediction(saved_model_name=current_model_name)
    print("Model Test Accuracy: ", test_accuracy)