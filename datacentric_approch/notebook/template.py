import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as pl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from imblearn.over_sampling import RandomOverSampler


class data_val():
    
    def __init__(self,df : input):
        self.df = df


    def missing_v(self, column_types):
        missing = {}
        for column in self.df.columns:
            if column_types[column] == 'numerical':
                missing[column] = np.count_nonzero((self.df[column]!=self.df[column])==1)
        print(missing)

    def get_column_type(self):
        column_types = {}
        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                if self.df[column].nunique() <= 10 :
                    column_types[column] = 'categorical'
                else :
                    column_types[column] = 'numerical'
            elif pd.api.types.is_string_dtype(self.df[column]) or pd.api.types.is_categorical_dtype(self.df[column]):
                column_types[column] = 'categorical'
            else:
                column_types[column] = 'other'
        return column_types

    def missing_count(self, column_types):
        missing = {}
        for column in self.df.columns:
            if column_types[column] == 'numerical':
                missing[column] = self.df[column].isna().sum()
        print(missing)

    def zero_count(self,):
        zeros = {}
        for column in self.df.columns:
            zeros[column] = np.count_nonzero(self.df[column] == 0)
        return pd.DataFrame(list(zeros.items()), columns=["Column_name","Zeros"])

    #def zero_count1(self):
       # numeric = self.df.select_dtypes(include=[np.number])
       # print(numeric)
       # zeros = numeric[numeric == 0].sum().astype(int)
       # print(zeros)
       # print("hi")
       # return zeros.to_dict
        """zeros = {}
        if pd.api.types.is_numeric_dtype(self.df.columns): 
            zeros[self.df.columns] = np.count_nonzero(self.df[self.df.columns]==0)
        return zeros"""

    def remove_outliers(self,):
        numeric = self.df.select_dtypes(include = [np.number])
        for col in numeric.columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3-q1
            lowerBound = q1 - 1.5*iqr
            upperBound = q3 + 1.5*iqr
            self.df = self.df[(self.df[col]>=lowerBound) & (self.df[col]<=upperBound)]
            
        return self.df


    def scale_dataset(self,Oversample = False,scaler1 = StandardScaler):
        col = self.df.columns
        x1 = self.df[self.df.columns[:-1]].values
        y1 = self.df[self.df.columns[-1]].values
        scalar = scaler1()
        x1 = scalar.fit_transform(x1)
        if Oversample:
            ros = RandomOverSampler(random_state=0)
            x1, y1 = ros.fit_resample(x1,y1)
        self.df = np.hstack((x1, np.reshape(y1,(-1,1))))
        self.df =  pd.DataFrame(self.df, columns=col)
        return self.df
    
    def unique_count(self):
        catergoical = self.df.select_dtypes(exclude = [np.number])
        unique = {}
        for column in catergoical.columns:
            unique[column] = pd.unique(self.df.columns[column])
        return unique
    
    def histplot_columns(self,length = 1, Width = 1 , figuresize = (10,10)):
        fig, axs =plt.subplots(length, Width, figsize=figuresize)
        sns.diverging_palette(15,260)
        x = [col for col in self.df.columns[:-1]]
        axes = axs.flatten()
        for ax , cols in zip(axes,x):
            for cat in pd.unique(self.df[self.df.columns[-1]]):
                sns.histplot(data=self.df[self.df[self.df.columns[-1]]==cat][cols].to_frame(name=cols),x=cols,ax=ax,label = cat,alpha=0.7)
            ax.set(xlabel = cols,ylabel ="count")
            ax.legend()
    
    """ def barplot_columns(self, length=1, width=1, figuresize=(10, 10)):
        fig, axs = plt.subplots(length, width, figsize=figuresize)
        x = [col for col in self.df.columns[:-1]]
        axes = axs.flatten()
        for ax, cols in zip(axes, x):
            for cat in pd.unique(self.df[self.df.columns[-1]]):
                cat_data = self.df[self.df[self.df.columns[-1]] == cat]
                sns.barplot(x=cols, y=cat_data[cols].count(), ax=ax, label=cat, alpha=0.7)
            ax.set(xlabel=cols, ylabel="count")
            ax.legend(title=self.df.columns[-1])
        plt.tight_layout()
        plt.show()"""
    
    def barplot_columns(self,length=1, width=1, figuresize=(10, 10)):
        fig, axs = plt.subplots(length, width, figsize=figuresize)
        sns.set_palette("crest")
        
        # Assuming the target variable is the last column in the dataframe
        target = self.df.columns[-1]
        features = self.df.columns[:-1]
        
        axes = axs.flatten()
        
        for ax, col in zip(axes, features):
            sns.barplot(data=self.df, x=target, y=col, ax=ax, ci=None, alpha=0.7)
            ax.set(xlabel=target, ylabel=col)
        
        plt.tight_layout()
        plt.show()

    def histplot_columns1(self, length=1, width=1, figuresize=(10, 10)):
        fig, axs = plt.subplots(length, width, figsize=figuresize)
        sns.set_palette("crest")
        
        # Select numerical columns
        numerical_cols = self.df.columns
        
        axes = axs.flatten()
        
        for ax, col in zip(axes, numerical_cols):
            sns.histplot(data=self.df, x=col, ax=ax, kde=True, bins=30, alpha=0.7)
            ax.set(xlabel=col, ylabel='count')
        
        plt.tight_layout()
        plt.show()

    def scatterplot(self,row = 1, column = 1 , figuresize = (10,10),column_type = None):
        fig, axs = plt.subplots(row, column, figsize=figuresize)
        sns.set_palette("crest")
        axes = axs.flatten()

        for col ,axs , in zip(self.df.columns[:-1],axes):
            if column_type[col] == 'numerical':
                sns.scatterplot(data=self.df, x=col, y=self.df[self.df.columns[-1]],ax=axs)
                axs.set(xlabel=col, ylabel=self.df.columns[-1])
            else:
                fig.delaxes(axs)
        
       
        
        plt.tight_layout()
        plt.show()

    def scatterplot1(self, row=1, column=1, figuresize=(10, 10), column_type=None):
        numerical_columns = [col for col in self.df.columns[:-1] if column_type and column_type[col] == 'numerical']
        n_plots = len(numerical_columns)
        
        # Calculate the grid size based on the number of numerical columns
        total_plots = min(row * column, n_plots)
        fig, axs = plt.subplots(row, column, figsize=figuresize)
        sns.set_palette("crest")
        
        # Flatten the axes array for easy iteration
        axes = axs.flatten()
        
        # Plot only on the needed axes
        for ax, col in zip(axes[:total_plots], numerical_columns):
            sns.scatterplot(data=self.df, x=col, y=self.df[self.df.columns[-1]], ax=ax)
            ax.set(xlabel=col, ylabel=self.df.columns[-1])
        
        # Remove extra axes from the figure
        for ax in axes[total_plots:]:
            fig.delaxes(ax)
        
        plt.tight_layout()
        plt.show()

    def schema(self):
        schema = {}
        for col in self.df.columns:
            schema[col]=self.df[col].dtype
        return pd.DataFrame(schema.items(),columns=["columns","dtype"])
