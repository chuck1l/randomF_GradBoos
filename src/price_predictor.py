import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

class price_predictor(object):
    def __init__(self, name, data):
        self.name = name
        self.df = data

    # Creating X, y arrays
    def X_y(self):
        self.y_high = self.df['trw_high']
        self.y_low = self.df['trw_low']
        self.X = self.df.drop(['trw_high', 'trw_low', 'datetime', 'date'], axis=1) 
        return None

    # Create data sets for training/testing and holdout
    def train_holdout(self):
        self.X_h_t, self.X_h_ho, self.y_h_t, self.y_h_ho = train_test_split(self.X, self.y_high, test_size=0.25, random_state=42)
        self.X_l_t, self.X_l_ho, self.y_l_t, self.y_l_ho = train_test_split(self.X, self.y_low, test_size=0.25, random_state=42)
        return None

    # Create X, y train and test sets 
    def train_test(self):
        self.X_h_train, self.X_h_test, self.y_h_train, self.y_h_test = train_test_split(self.X_h_t, self.y_h_t, test_size=.30, random_state=42)
        self.X_l_train, self.X_l_test, self.y_l_train, self.y_l_test = train_test_split(self.X_l_t, self.y_l_t, test_size=.30, random_state=42)
        return None

    def shapes(self):
        print('Train and Holdout:')
        print('Training and Holdout Shapes High of Day')
        print(f'X, y high train: {self.X_h_t.shape}, {self.y_h_t.shape}, \nX, y high holdout: {self.X_h_ho.shape}, {self.y_h_ho.shape}')
        print('\nTraining and Holdout Shapes Low of Day')
        print(f'X, y low train: {self.X_l_t.shape}, {self.y_l_t.shape}, \nX, y low holdout: {self.X_l_ho.shape}, {self.y_l_ho.shape}')
        print('\nTrain and Test:')
        print('Training and Test Shapes High of Day')
        print(f'X, y high train: {self.X_h_train.shape}, {self.y_h_train.shape}, \nX, y high test: {self.X_h_test.shape}, {self.y_h_test.shape}')
        print('\nTraining and Test Shapes Low of Day')
        print(f'X, y low train: {self.X_l_train.shape}, {self.y_l_train.shape}, \nX, y low test: {self.X_l_test.shape}, {self.y_l_test.shape}')
        return None

    def feature_importance_selecting(self):
        labels = pd.Series(self.X_h_train.columns, name='features')
        graph_labels = list(self.X_h_train.columns)
        # Create models with default inputs
        boosting_model_high = GradientBoostingRegressor()
        boosting_model_high.fit(self.X_h_train, self.y_h_train)
        boosting_model_low = GradientBoostingRegressor()
        boosting_model_low.fit(self.X_l_train, self.y_l_train)

        randomforest_model_high = RandomForestRegressor()
        randomforest_model_high.fit(self.X_h_train, self.y_h_train)
        randomforest_model_low = RandomForestRegressor()
        randomforest_model_low.fit(self.X_l_train, self.y_l_train)

        # Get feature importance for both models
        importance_high_boosting = pd.Series(boosting_model_high.feature_importances_, name='boosting_high')
        importance_high_randomforest = pd.Series(randomforest_model_high.feature_importances_, name='randomforest_high')
        importance_low_boosting = pd.Series(boosting_model_low.feature_importances_, name='boosting_low')
        importance_low_randomforest = pd.Series(randomforest_model_low.feature_importances_, name='randomforest_low')
        features_df = pd.concat([labels, importance_high_boosting, 
                                        importance_high_randomforest,
                                        importance_low_boosting,
                                        importance_low_randomforest], 
                                        axis=1)
        features_df = features_df.sort_values(by='boosting_high', ascending=False)

        x = np.arange(features_df.shape[0]) # Label locations
        width = 0.35 # the width of the bars for each label
        # Plot Boosting Model Importance
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.bar(x - width/2, features_df['boosting_high'], width, label='High')
        ax.bar(x + width/2, features_df['boosting_low'], width, label='Low')
        ax.set_ylabel('Importance Score')
        ax.set_title('Feature Importance for Boosting Model')
        ax.set_xticks(x)
        ax.set_xticklabels(graph_labels, rotation='vertical')
        ax.legend()
        fig.tight_layout()
        # plt.savefig('../img/boosting_feature_importance.png')
        # plt.show()
        feature_mask = (features_df['boosting_high'] >= .01) | (features_df['boosting_low'] >= .01) | (features_df['randomforest_high'] >= .01) | (features_df['randomforest_low'] >= .01)
        self.important_cols = list(features_df['features'][feature_mask])
        return None
    
    def reassign_X_features(self):
        self.X_h_train, self.X_h_test = self.X_h_train[self.important_cols], self.X_h_test[self.important_cols]
        self.X_l_train, self.X_l_test = self.X_l_train[self.important_cols], self.X_l_test[self.important_cols]
        self.X_h_ho, self.X_l_ho = self.X_h_ho[self.important_cols], self.X_l_ho[self.important_cols]

    # Randomized Search for best parameters
    def gradient_boost_reg_param_select(self, X_train, y_train):
        X_train2 = X_train.values
        y_train2 = y_train.values
        regressor_gbr = GradientBoostingRegressor()
        # Parameters to investigate
        num_estimators = [600, 800, 900, 1000, 1100]
        learn_rates = [0.007, .009, 0.01, 0.015]
        max_depths = [3, 8, 16, 20]
        min_samples_split = [2, 3, 4]
        min_samples_leaf = [1, 2, 3]
        # dictionary containing the parameters
        param_grid = {'n_estimators': num_estimators,
                    'learning_rate': learn_rates,
                    'max_depth': max_depths,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}
        desired_iterations = 100 
        random_search = RandomizedSearchCV(regressor_gbr,
                                            param_grid,
                                            scoring='neg_root_mean_squared_error',
                                            cv=3,
                                            n_iter=desired_iterations,
                                            verbose=1,
                                            return_train_score=True,
                                            n_jobs=-1)
        random_search.fit(X_train2, y_train2)
        boosting_randomsearch_bestparams = random_search.best_params_
        print('Boosting Params: ', boosting_randomsearch_bestparams)
        self.boosting_randomsearch_bestscore = -1 * random_search.best_score_
        print(f'Boosting Score: {self.boosting_randomsearch_bestscore:0.3f}')
        self.gbr_n_estimators = boosting_randomsearch_bestparams['n_estimators']
        self.gbr_min_samples_split = boosting_randomsearch_bestparams['min_samples_split']
        self.gbr_min_saplies_leaf = boosting_randomsearch_bestparams['min_samples_leaf']
        self.gbr_max_depth = boosting_randomsearch_bestparams['max_depth']
        self.gbr_learning_rate = boosting_randomsearch_bestparams['learning_rate']
        return None

    def random_forest_reg_param_select(self, X_train, y_train):
        X_train2 = X_train.values
        y_train2 = y_train.values
        regressor = RandomForestRegressor()
        # Parameters to investigate
        num_estimators = [600, 800, 900, 1000]
        max_features = [4, 6, 8]
        max_depth = [10, 15, 20, 25]
        min_samples_split = [2, 3, 4]
        min_samples_leaf = [1, 2, 3]
        # Dictionary containing the parameters
        param_grid = {'n_estimators': num_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}
        desired_iterations = 100
        random_search = RandomizedSearchCV(regressor, 
                                   param_grid, 
                                   scoring='neg_root_mean_squared_error',
                                   cv=3,
                                   n_iter=desired_iterations,
                                   verbose=1,
                                   return_train_score=True,
                                   n_jobs=-1)
        random_search.fit(X_train2, y_train2)
        randomforest_randomsearch_bestparams = random_search.best_params_
        print('Random Forest Params', random_search.best_params_)  
        self.randomforest_randomsearch_bestscore = -1 * random_search.best_score_
        print(f'Random Forest Score: {self.randomforest_randomsearch_bestscore:0.3f}')  # negative root mean square error

        self.rfr_n_estimators = randomforest_randomsearch_bestparams['n_estimators']
        self.rfr_min_samples_split = randomforest_randomsearch_bestparams['min_samples_split']
        self.rfr_min_saplies_leaf = randomforest_randomsearch_bestparams['min_samples_leaf']
        self.rfr_max_features = randomforest_randomsearch_bestparams['max_features']
        self.rfr_max_depth = randomforest_randomsearch_bestparams['max_depth']
        return None

    # Creating the model and predictions for the High of Day values
    def run_model_high(self, X_train, y_train, X_test, y_test, X_holdout, y_holdout):
        X_train2, y_train2 = X_train.values, y_train.values
        X_test2, y_test2 = X_test.values, y_test.values
        X_holdout2, y_holdout2 = X_holdout.values, y_holdout.values
        # Gradient Boosting model
        boosting_model = GradientBoostingRegressor(learning_rate=self.gbr_learning_rate,
                                                    max_depth=self.gbr_max_depth,
                                                    min_samples_leaf=self.gbr_min_saplies_leaf,
                                                    min_samples_split=self.gbr_min_samples_split,
                                                    n_estimators=self.gbr_n_estimators)
        boosting_model.fit(X_train2, y_train2)
        y_hat_boosting = boosting_model.predict(X_test2)
        # Random Forest Model
        randomforest_model = RandomForestRegressor(max_features=self.rfr_max_features,
                                                    max_depth=self.rfr_max_depth,
                                                    min_samples_leaf=self.rfr_min_saplies_leaf,
                                                    min_samples_split=self.rfr_min_samples_split,
                                                    n_estimators=self.rfr_n_estimators)
        randomforest_model.fit(X_train2, y_train2)
        y_hat_radomforest = randomforest_model.predict(X_test2)
            
        self.rmse_boosting_high = np.sqrt(mean_squared_error(y_hat_boosting, y_test2))
        self.rmse_randomforest_high = np.sqrt(mean_squared_error(y_hat_radomforest, y_test2))

        print(f'Boosting RMSE High of Day: {self.rmse_boosting_high:0.3f}')
        print(f'Random Forest RMSE High of Day: {self.rmse_randomforest_high:0.3f}')

        if self.rmse_boosting_high < self.rmse_randomforest_high:
            self.y_hat_high = boosting_model.predict(X_holdout2)
           
        else:
            self.y_hat_high = randomforest_model.predict(X_holdout2)

        rmse_high_holdout = np.sqrt(mean_squared_error(self.y_hat_high, y_holdout2))
        print(f'High of Holdout RMSE: {rmse_high_holdout:0.3f}')
        return None

    # Creating the model and predictions for the Low of Day values
    def run_model_low(self, X_train, y_train, X_test, y_test, X_holdout, y_holdout):
        X_train2, y_train2 = X_train.values, y_train.values
        X_test2, y_test2 = X_test.values, y_test.values
        X_holdout2, y_holdout2 = X_holdout.values, y_holdout.values
        # Gradient Boosting model
        boosting_model = GradientBoostingRegressor(learning_rate=self.gbr_learning_rate,
                                                    max_depth=self.gbr_max_depth,
                                                    min_samples_leaf=self.gbr_min_saplies_leaf,
                                                    min_samples_split=self.gbr_min_samples_split,
                                                    n_estimators=self.gbr_n_estimators)
        boosting_model.fit(X_train2, y_train2)
        y_hat_boosting = boosting_model.predict(X_test2)
        # Random Forest Model
        randomforest_model = RandomForestRegressor(max_features=self.rfr_max_features,
                                                    max_depth=self.rfr_max_depth,
                                                    min_samples_leaf=self.rfr_min_saplies_leaf,
                                                    min_samples_split=self.rfr_min_samples_split,
                                                    n_estimators=self.rfr_n_estimators)
        randomforest_model.fit(X_train2, y_train2)
        y_hat_radomforest = randomforest_model.predict(X_test2)

        self.rmse_boosting_low = np.sqrt(mean_squared_error(y_hat_boosting, y_test2))
        self.rmse_randomforest_low = np.sqrt(mean_squared_error(y_hat_radomforest, y_test2))

        print(f'Boosting RMSE Low of Day: {self.rmse_boosting_low:0.3f}')
        print(f'Random Forest RMSE Low of Day: {self.rmse_randomforest_low:0.3f}')

        if self.rmse_boosting_low < self.rmse_randomforest_low:
            self.y_hat_low = boosting_model.predict(X_holdout2)
           
        else:
            self.y_hat_low = randomforest_model.predict(X_holdout2)

        rmse_low_holdout = np.sqrt(mean_squared_error(self.y_hat_low, y_holdout2))
        print(f'Low of Day Holdout RMSE: {rmse_low_holdout:0.3f}')
        return None

    # Creating a DataFrame to Analyze the results of the models vis-a-vis holdout    
    def result_analysis(self):
        temp_df1 = pd.DataFrame(self.y_hat_low, index=self.y_l_ho.index, columns=['y_hat_low'])
        temp_df2 = pd.DataFrame(self.y_hat_high, index=self.y_h_ho.index, columns=['y_hat_high'])
        self.result_df = pd.concat([self.y_h_ho, self.y_l_ho, temp_df2, temp_df1], axis=1)
        self.result_df.rename(columns={'trw_high': 'true_high', 'trw_low': 'true_low'}, inplace=True)
        self.result_df.sort_index(inplace=True)
        self.result_df.to_csv('../data/spy_results.csv')
        return None

if __name__ == '__main__':
    pass