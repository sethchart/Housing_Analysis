# Housing Market Analysis

In this repository we analyze housing data from King County in Seattle. We produce a regression model for house price and investigate the factors that influence price.

## Business Problem 

For this project we identify our hypothetical stakeholder as King County Assessor [John Wilson](https://www.kingcounty.gov/depts/assessor/About-Us/Assessors-Bio.aspx). We are asked to use data from house sales in King County occurring between May 2, 2014 and May 27, 2015 to produce a model for house prices. This model will be used by the assessor to validate the existing property taxation model. 

## Key Insights
In this section we briefly summarise our findings as they relate to the business problem.
### Model

We produced a linear model for the base 10 logarithm of price. That is, if the price of a house is <img src="https://render.githubusercontent.com/render/math?math=P">, then the target for our linear model is <img src="https://render.githubusercontent.com/render/math?math=y = \log_{10}(P)">. Our model incorporates four input variables, which we will denote as <img src="https://render.githubusercontent.com/render/math?math=x_{k}"> for <img src="https://render.githubusercontent.com/render/math?math=k = 1, 2, 3, 4."> Each variable is described below.
 1. The variable <img src="https://render.githubusercontent.com/render/math?math=x_{1}"> is an indicator variable that is 1 if the house is in one of zip codes that we have identified as having low median price and 0 otherwise.
 2. The variable <img src="https://render.githubusercontent.com/render/math?math=x_{2}"> is an indicator variable that is 1 if the house is in one of zip codes that we have identified as having intermediate median price and 0 otherwise. Note that when both indicator variables are zero, the house is in one of the zip codes that we have identified as having high median price.
 3. The variable <img src="https://render.githubusercontent.com/render/math?math=x_{3}"> is a continuous variable that describes the total livable square footage of the houses 15 nearest neighbors. This variable has been Box-Cox transformed with parameter <img src="https://render.githubusercontent.com/render/math?math=\lambda = -0.21">.
 4. The variable <img src="https://render.githubusercontent.com/render/math?math=x_{4}"> is a continuous variable that describes the number of bathrooms per bedroom in the house. This variable has been Box-Cox transformed with parameter <img src="https://render.githubusercontent.com/render/math?math=\lambda = 0.21">.

Our linear model takes the form:

<img src="https://render.githubusercontent.com/render/math?math='y=0.37-0.29x_{1}-0.12x_{2}+1.44x_{3}\+0.08x_{4}'">

### Spatial Distribution of Price

### Classification of Zip Codes by Price

### Bathroom to Bedroom Ratio

## Data Validation

## Data Cleaning 

## Feature Engineering

## Model Selection

## Model Validation 

### Linearity

### Normality of Residuals

### Uniform Variance of Residuals

### Linear Independence of Inputs

### Significance of Parameters

### Measures of Model Quality

## Points of Interest

### Test Driven Design

### Object Oriented Programming
