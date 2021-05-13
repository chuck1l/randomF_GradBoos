import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt  
from datetime import datetime
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})





if __name__ == '__main__':
    df = pd.read_csv('../data/results.csv')
    
    df.rename(columns={'trw_high': 'true_high', 'trw_low': 'true_low'}, inplace=True)
    df.set_index('datetime', inplace=True)
    
    ax = df[['y_hat_high', 'true_high']].plot(figsize=(12, 10))
    ax.set_ylabel('High of Day Stock Price')
    ax.set_xlabel('Datetime')
    ax.set_title('Comparing Predicted Price to True Price, High of Day')
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.savefig('../res_imgs/high_of_day.png')

    ax1 = df[['y_hat_low', 'true_low']].plot(figsize=(12, 10))
    ax1.set_ylabel('Low of Day Stock Price')
    ax1.set_xlabel('Datetime')
    ax1.set_title('Comparing Predicted Price to True Price, Low of Day')
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.savefig('../res_imgs/low_of_day.png')
    #plt.show()
    shares = 1000
    stop_loss = .10
    trailing = .05
    profit_point1 = .6
    profit_mark = (df['y_hat_high'] - df['y_hat_low']) * profit_point1 + df['y_hat_low']

    condition1 = (df['y_hat_low'] < df['true_low'])
    choice1 = 0

    condition2 = (df['y_hat_low'] > df['true_high'])
    choice2 = 0

    condition3 = (df['y_hat_low'] >= df['true_low']) & (df['y_hat_high'] <= df['true_high'])
    choice3 = (df['y_hat_high'] - df['y_hat_low']) * shares * (1-profit_point1) + (df['y_hat_high'] - df['y_hat_low']) * .5 * profit_point1*shares 

    condition4 = (df['y_hat_low'] >= df['true_low']) & (df['y_hat_high'] > df['true_high']) & (df['true_high'] >= profit_mark) & (profit_mark >= df['y_hat_low']+trailing)
    choice4 = (df['y_hat_high'] - df['y_hat_low']) * .5 * shares * profit_point1 + trailing * shares*(1-profit_point1)

    condition5 = (df['y_hat_low'] >= df['true_low']) & (df['y_hat_high'] > df['true_high']) & (df['true_high'] < profit_mark) & (df['true_high'] >= df['y_hat_low'] + trailing)
    choice5 = trailing * shares

    condition6 = (df['y_hat_low'] >= df['true_low']) & (df['y_hat_high'] > df['true_high']) & (df['true_high'] < df['y_hat_low'] + trailing)
    choice6 = -1 * trailing * shares

    conditions = [condition1, condition2, condition3, condition4, condition5, condition6]
    choices = [choice1, choice2, choice3, choice4, choice5, choice6]
    df['profit'] = np.select(conditions, choices, default=np.nan)
    print(np.sum(df['profit']))
    print(np.sum(df['profit'])/df.shape[0])
    print(df.shape[0])