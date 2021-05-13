# Load Libraries
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
plt.style.use('ggplot')
import subprocess
from statsmodels.stats.outliers_influence import variance_inflation_factor

class graphing():
    def __init__(self, name, data):
        self.name = name
        self.df = data

    def scatters(self):
        high = self.df['trw_high']
        low = self.df['trw_low']
        graph_df = self.df.copy().drop(['trw_high', 'trw_low'], axis=1)
        columns = graph_df.columns
        
        for col in columns:
            plt.scatter(high, graph_df[col], alpha=0.5, label=col)
            plt.ylabel(col)
            plt.xlabel("Tomorrow's High of Day")
            plt.legend(loc='best')
            location = f'../img/{col}_v_hofd.png'
            #plt.savefig(location)
    def correlations(self):
        high = self.df['trw_high']
        lo = self.df['trw_low']
        corr_df = self.df.copy().drop(['trw_high', 'trw_low', 'datetime'], axis=1)
        columns = corr_df.columns
        df1 = pd.DataFrame(corr_df.columns, columns=['features'])
        hi = []
        low = []
        for col in columns:
            hi.append(high.corr(self.df[col], method='spearman'))
            low.append(lo.corr(self.df[col], method='spearman'))
        df2 = pd.DataFrame(hi, columns=['tomorrow_high_corr'])
        df3 = pd.DataFrame(low, columns=['tmorrow_low_corr'])
        res_df = pd.concat([df1, df2, df3], axis=1)
        
        # res_df.to_html('../spy_correlation_table.html')
        # subprocess.call(
        #     'wkhtmltoimage -f png --width 0 table2.html table2.png', shell=True)
        print(res_df)
    def calc_vif(self):
        '''
        VIF starts at 1 and has no upper limit
        VIF = 1, no correlation between the independent variable and the other variables
        VIF exceeding 5 or 10 indicates high multicollinearity between this independent variable and the others
        '''
        # Calc VIF
        vif = pd.DataFrame()
        df1 = self.df.copy().drop(['datetime', 'date', 'trw_high', 'trw_low'], axis=1).round(2)
        vif['variables'] = df1.columns
        vif['VIF'] = [variance_inflation_factor(df1.values, i) for i in range(df1.shape[1])]

        vif.to_html('../spy_vif_table.html')
        subprocess.call('wkhtmltoimage -f png --width 0 table2.html table2.png', shell=True)
        print(vif)
        



if __name__ == '__main__':
    pass