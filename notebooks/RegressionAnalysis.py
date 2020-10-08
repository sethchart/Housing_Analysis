#!/usr/bin/env python
# coding: utf-8

# # Regression Analysis Development
# In this notebook we develop helper functions for inclusion in a regression analysis class.

# ## Import libraries

# In[ ]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('talk')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats
from statsmodels import api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


# ## Set path to cleaned data

# In[ ]:


cleaned_data = os.path.abspath('data/cleaned.csv')


# ## Test data

# In[ ]:


np.random.seed(42)
df = pd.DataFrame(np.random.randint(0,100,size=(10, 4)), columns=list('ABCD'))
X = df.drop('A', axis=1)
y = df['A']
linreg = LinearRegression(fit_intercept=True)
model = linreg.fit(X, y)
y_hat = model.predict(X)
residuals = y-y_hat


# ## Function to load cleaned data

# In[ ]:


def load_data():
    """Load the cleaned data as a data frame."""
    df = pd.read_csv(cleaned_data)
    return df

def test_load_data():
    """Test the load_data function."""
    test_passes = True
    try:
        df = load_data()
    except:
        test_passes = False
    return test_passes

test_load_data()


# ## Function to make pair plot of dataframe

# In[ ]:


def make_pair_plot(df):
    """Given a dataframe make a pairplot of all variables."""
    return sns.pairplot(data=df)

def test_make_distribution_plot():
    """Test the make_pair_plot function."""
    test_passes = True 
    try:
        make_pair_plot(df);
    except:
        test_passes = False
    return test_passes

test_make_distribution_plot()


# ## Function to make correlation of dataframe

# In[ ]:


def make_correlation_plot(df):
    """Given a dataframe make heatmap of pairwise correlation."""
    corr = df.corr()
    return sns.heatmap(corr, annot=True, cmap='Blues')

def test_make_correlation_plot():
    """Test the make_corelation_plot function."""
    test_passes = True
    try:
        make_correlation_plot(df);
    except:
        test_passes = False
    return test_passes

test_make_correlation_plot()


# ## Function to fit model to traning data

# In[ ]:


def fit_model(X_train, y_train, fit_intercept=True):
    """Fit a least squares linear model to the training inputs X_train and training outputs y_train."""
    linreg = LinearRegression(fit_intercept=fit_intercept)
    model = linreg.fit(X_train, y_train)
    return model

def test_fit_model():
    """Test fit_model function."""
    try:
        model = fit_model(X, y)
        coef_compair = model.coef_ == np.array([-0.02388839656181407, 0.4214208340984754, 0.6505954052083907])
        coef_match = coef_compair.min()
        intercept_match = model.intercept_ == -20.611944473223005
        test_passes = coef_match and intercept_match
    except:
        test_passes = False/j
    return test_passes

test_fit_model()


# ## Function to predict target from input variabes

# In[ ]:


def predict_target(X, model):
    """Use the provided model and data X to produce predicted values of the target."""
    y_hat = model.predict(X)
    return y_hat

def test_predict_target():
    """Test predict_target function."""
    try:
        y_hat = predict_target(X, model) 
        prediction_match = y_hat.round(2) == np.array(
            [29.28, 69.42, 78.69, 22.02, 13.6, 15.76, 15.28, 53.9, 53.48, 33.56]
        )
        test_passes = prediction_match.min()
    except:
        test_passes = False
    return test_passes

test_predict_target()


# ## Function to compute residuals

# In[ ]:


def compute_residuals(y, y_hat):
    """Compute residuals from a series of observed values y and predicted values y_hat"""
    return y-y_hat

def test_compute_residuals():
    try:
        residuals = compute_residuals(y, y_hat)
        residuals_match = residuals.round(2) == np.array(
            [21.72, -9.42, -4.69, 0.98, -12.6, -14.76, 16.72, 34.1, -12.48, -19.56]
        )
        test_passes = residuals_match.min()
    except:
        test_passes = False
    return test_passes
test_compute_residuals()    


# ## Function to plot residuals against all input variables

# In[ ]:


def plot_residuals_against_inputs(X, residuals):
    """Plot residuals against input variables."""
    number_of_plots = len(X.columns)
    nrows = int(np.sqrt(number_of_plots))
    ncols = int(number_of_plots/nrows)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_figheight(6*nrows)
    fig.set_figwidth(8*ncols)
    for ax, col in zip(axs.flat, X.columns):
        sns.scatterplot(x=X[col], y=residuals, ax=ax)
        sns.lineplot(x=[X[col].min(), X[col].max()], y=[0, 0], ax=ax, color='red')
        ax.set_xlabel(col.title())
        ax.set_ylabel('Residuals')
        ax.set_title(f'Residuals vs {col.title()}')

def test_plot_residuals_against_inputs():
    """Test the plot_residuals function."""
    try:
        plot_residuals_against_inputs(X, residuals)
        test_passes = True
    except:
        test_passes = False
    return test_passes

test_plot_residuals_against_inputs()


# ## Function to plot residuals against predicted values

# In[ ]:


def plot_residuals_against_prediction(y, residuals):
    """Plot residuals against predicted values."""
    ax = sns.scatterplot(
        x=y, 
        y=residuals
    )
    sns.lineplot(x=[y.min(), y.max()], y=[0, 0], color='red', ax=ax)
    ax.set(
        title='Residuals vs Predicted Values',
        xlabel='Predicted Values',
        ylabel='Residuals'
    )

def test_plot_residuals_against_prediction():
    """Test the plot_residuals function."""
    try:
        plot_residuals_against_prediction(y, residuals)
        test_passes = True
    except:
        test_phistogramasses = False
    return test_passes

test_plot_residuals_against_prediction()


# In[ ]:


## Function to plot distribution of residuals


# In[ ]:


def plot_residuals_distribution(residuals):
    """Plot the distribution of normalized residuals."""
    normalized_residuals = residuals/residuals.std()
    ax = sns.distplot(normalized_residuals)
    ax.set(
        title='Distribution of Normalized Residuals.',
        xlabel='Residuals',
        ylabel='Probability Density'
    )
    
def test_plot_residuals_ditribution():
    """Test plot_residuals_, title='Test'distribution function"""
    try:
        plot_residuals_distribution(residuals)
        test_passes = True
    except:
        test_passes = False
    return test_passes

test_plot_residuals_ditribution()


# ## Function to make a QQ plot of residuals and conduct a Kolmogorov Smirnov test for normality 

# In[ ]:


def plot_residuals_normal_qq(residuals):
    """Make a QQ ploat of normalized residuals against a standard normal distribution."""
    plt.style.use('ggplot')
    normalized_residuals = residuals/residuals.std()
    sm.graphics.qqplot(normalized_residuals, dist=stats.norm, line='45', fit=True)
    kstest_result = stats.kstest(normalized_residuals, 'norm')
    if kstest_result.pvalue <= 0.05:
        print(f'We reject the null hypothesis that our residuals are normally distributed at the $\alpha =0.05$ level.')
    else:
        print(f'We fail to reject the null hypothesis that our residuals are normally distributed at the alpha = 0.05 level.')


def test_plot_residuals_ditribution():
    """Test plot_residuals_normal_qq function"""
    try:
        plot_residuals_normal_qq(residuals)
        test_passes = True
    except:
        test_passes = False
    return test_passes

test_plot_residuals_ditribution()


# ## Function to compute variance inflation factors for all input variables

# In[ ]:


def compute_vif(X):
    """Compute variance inflation factors for all input variables."""
    vifs = dict()
    for ind, col in enumerate(X.columns):
        vif_score = vif(np.matrix(X), ind)
        vifs[col] = vif_score.round(2)
    return vifs

def test_compute_vif():
    """Test the compute_vif function."""
    ref_values = {'B': 3.3, 'C': 5.0, 'D': 5.41}
    vifs = compute_vif(X)
    tests = [vifs[col] == ref_values[col] for col in X.columns]
    test_passes = np.min(tests)
    return test_passes

test_compute_vif()

