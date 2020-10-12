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


class RegressionAnalysis(object):


    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.X = self.data.drop(target, axis=1)
        self.y = self.data[target]


    def make_pair_plot(self):
        """Given a dataframe make a pairplot of all variables."""
        return sns.pairplot(data=self.data)


    def make_correlation_plot(self):
        """Given a dataframe make heatmap of pairwise correlation."""
        corr = self.data.corr()
        return sns.heatmap(corr, annot=True, cmap='Blues')

    def make_test_train_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


    def fit_model(self, X_train, y_train, fit_intercept=True):
        """Fit a least squares linear model to the training inputs X_train and training outputs y_train."""
        linreg = LinearRegression(fit_intercept=fit_intercept)
        model = linreg.fit(X_train, y_train)
        return model


    def predict_target(self, X, model):
        """Use the provided model and data X to produce predicted values of the target."""
        y_hat = model.predict(X)
        return y_hat


    def compute_residuals(self, y, y_hat):
        """Compute residuals from a series of observed values y and predicted values y_hat"""
        return y-y_hat


    def plot_residuals_against_inputs(self, X, residuals):
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


    def plot_residuals_against_prediction(self, y_hat, residuals):
        """Plot residuals against predicted values."""
        ax = sns.scatterplot(
            x=y_hat, 
            y=residuals
        )
        sns.lineplot(x=[y_hat.min(), y_hat.max()], y=[0, 0], color='red', ax=ax)
        ax.set(
            title='Residuals vs Predicted Values',
            xlabel='Predicted Values',
            ylabel='Residuals'
        )



    def plot_residuals_distribution(self, residuals):
        """Plot the distribution of normalized residuals."""
        normalized_residuals = residuals/residuals.std()
        ax = sns.distplot(normalized_residuals)
        ax.set(
            title='Distribution of Normalized Residuals.',
            xlabel='Residuals',
            ylabel='Probability Density'
        )


    def plot_residuals_normal_qq(self, residuals):
        """Make a QQ ploat of normalized residuals against a standard normal distribution."""
        plt.style.use('ggplot')
        normalized_residuals = residuals/residuals.std()
        sm.graphics.qqplot(normalized_residuals, dist=stats.norm, line='45', fit=True)
        kstest_result = stats.kstest(normalized_residuals, 'norm')
        if kstest_result.pvalue <= 0.05:
            print(f'We reject the null hypothesis that our residuals are normally distributed at the alpha =0.05 level.')
        else:
            print(f'We fail to reject the null hypothesis that our residuals are normally distributed at the alpha = 0.05 level.')


    def compute_vif(self, X):
        """Compute variance inflation factors for all input variables."""
        vifs = dict()
        for ind, col in enumerate(X.columns):
            vif_score = vif(np.matrix(X), ind)
            vifs[col] = vif_score.round(2)
        return vifs


class _TestRegressionAnalysis(RegressionAnalysis):


    def __init__(self):
        np.random.seed(42)
        self.test_data = pd.DataFrame(np.random.randint(0,100,size=(10, 4)), columns=list('ABCD'))
        self.test_target = 'A'
        super().__init__(self.test_data , self.test_target)
#        self.linreg = LinearRegression(fit_intercept=True)
#        self.model = self.linreg.fit(self.X, self.y)
#        self.y_hat = self.model.predict(self.X)
#        self.residuals = self.y-self.y_hat


    def test_make_pair_plot(self):
        """Test the make_pair_plot function."""
        try:
            self.make_pair_plot();
            test_passes = True
        except:
            test_passes = False
        return test_passes


    def test_make_correlation_plot(self):
        """Test the make_corelation_plot function."""
        try:
            self.make_correlation_plot()
            test_passes = True
        except:
            test_passes = False
        return test_passes


    def test_fit_model(self):
        """Test fit_model function."""
        try:
            model = self.fit_model(self.X, self.y)
            coef_compair = model.coef_ == np.array([-0.02388839656181407, 0.4214208340984754, 0.6505954052083907])
            coef_match = coef_compair.min()
            intercept_match = model.intercept_ == -20.611944473223005
            test_passes = coef_match and intercept_match
        except:
            test_passes = False
        return test_passes


    def test_predict_target(self):
        """Test predict_target function."""
        try:
            y_hat = self.predict_target(self.X, self.model) 
            prediction_match = y_hat.round(2) == np.array(
                [29.28, 69.42, 78.69, 22.02, 13.6, 15.76, 15.28, 53.9, 53.48, 33.56]
            )
            test_passes = prediction_match.min()
        except:
            test_passes = False
        return test_passes


    def test_compute_residuals(self):
        try:
            residuals = self.compute_residuals(self.y, self.y_hat)
            residuals_match = residuals.round(2) == np.array(
                [21.72, -9.42, -4.69, 0.98, -12.6, -14.76, 16.72, 34.1, -12.48, -19.56]
            )
            test_passes = residuals_match.min()
        except:
            test_passes = False
        return test_passes


    def test_plot_residuals_against_inputs(self):
        """Test the plot_residuals function."""
        try:
            self.plot_residuals_against_inputs(self.X, self.residuals)
            test_passes = True
        except:
            test_passes = False
        return test_passes


    def test_plot_residuals_against_prediction(self):
        """Test the plot_residuals function."""
        try:
            self.plot_residuals_against_prediction(self.y, self.residuals)
            test_passes = True
        except:
            test_passes = False
        return test_passes


    def test_plot_residuals_ditribution(self):
        """Test plot_residuals_, title='Test'distribution function"""
        try:
            self.plot_residuals_distribution(self.residuals)
            test_passes = True
        except:
            test_passes = False
        return test_passes


    def test_plot_residuals_normal_qq(self):
        """Test plot_residuals_normal_qq function"""
        try:
            self.plot_residuals_normal_qq(self.residuals)
            test_passes = True
        except:
            test_passes = False
        return test_passes


    def test_compute_vif(self):
        """Test the compute_vif function."""
        ref_values = {'B': 3.3, 'C': 5.0, 'D': 5.41}
        vifs = self.compute_vif(self.X)
        tests = [vifs[col] == ref_values[col] for col in self.X.columns]
        test_passes = np.min(tests)
        return test_passes
