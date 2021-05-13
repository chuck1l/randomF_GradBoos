import pandas as pd  
import numpy as np
from data_cleaner import stock_clean
from graphing import graphing
from price_predictor import price_predictor

#Cleaning the Data
df = pd.read_csv('../data/spy_1d_all_indc.csv')
stock_daily = stock_clean('stock_daily', df)
stock_daily.drop_na_col()
col_lst = ['EMA Divergence', 'Plot.11', 'Plot.12', 'Plot.13', 'Plot.14']
stock_daily.drop_cols(col_lst)
stock_daily.date_col()
stock_daily.new_cols()
stock_daily.five_day_mean()
stock_daily.drop_na_rows()
stock_daily.format_cols()

# Graphing, VIF, and Correlations
toggle = 0
stock_daily_graphs = graphing('stock_graphs', stock_daily.df)
if toggle == 1:
    stock_daily_graphs.scatters()
    stock_daily_graphs.correlations()
    stock_daily_graphs.calc_vif()

# Create training/test and holdout data sets
data = stock_daily.df.copy()
stock_estimator = price_predictor('stock_estimator', data)
stock_estimator.X_y()
stock_estimator.train_holdout()
stock_estimator.train_test()
stock_estimator.feature_importance_selecting()
stock_estimator.reassign_X_features()
#stock_estimator.shapes()

# Start building models GradientBoostingRegression, first
stock_estimator.gradient_boost_reg_param_select(stock_estimator.X_h_train, 
                                                stock_estimator.y_h_train)
stock_estimator.random_forest_reg_param_select(stock_estimator.X_h_train, 
                                                stock_estimator.y_h_train)

# Creating the model and predictions for the High of Day values                                              
stock_estimator.run_model_high(stock_estimator.X_h_train, stock_estimator.y_h_train, 
                        stock_estimator.X_h_test, stock_estimator.y_h_test,
                        stock_estimator.X_h_ho, stock_estimator.y_h_ho)

# Creating the model and predictions for the Low of Day values
stock_estimator.run_model_low(stock_estimator.X_l_train, stock_estimator.y_l_train, 
                        stock_estimator.X_l_test, stock_estimator.y_l_test,
                        stock_estimator.X_l_ho, stock_estimator.y_l_ho)

# Creating the results DataFrame
stock_estimator.result_analysis()
print(stock_estimator.result_df.head())