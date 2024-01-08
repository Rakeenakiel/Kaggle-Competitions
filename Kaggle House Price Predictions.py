#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, KFold,RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import joblib

from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.decomposition import PCA 
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet as EN
import catboost as cb 


from skopt import BayesSearchCV, space, plots
from copy import deepcopy

pd.set_option('display.max_columns', None)
pd.set_option('display.max_columns', None)


# In[2]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# # Data Exploration

# In[3]:


df_train.isnull().sum(axis = 0).sort_values(ascending = False).head(19)


# We have 19 columns with NaN values that we need to deal with, for most of these columns NaN has a categorical meaning. So, some preprocessing is required.

# Additionally, by looking through the Meta Data, it is clear that some variables ordinal. Therefore, it is important to encode them appropriately.

# In[4]:


sns.histplot(df_train['SalePrice'])


# In[5]:


#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


# # Create Transformers and Pipelines For Data PreProcessing

# In order to prevent any leakage to have neat and tidy code, for each step that I did in my preprocessing, I created transformers for each step. I can pass this into a scikit learn pipeline and then just call it if I need it.
# 
# A pipeline is a lot like a workflow in R and these custom transformers are just your steps in between (i.e your `step_dummy`, `step_zv`, and etc)

# In[6]:


from sklearn.base import BaseEstimator, TransformerMixin

class GarageFix_1(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        A = X.copy()
        #Now we have to deal with Garage variables because these are less straightforward
        A.GarageFinish = A.GarageFinish.replace(np.nan, 'No Garage')
        A.GarageQual = A.GarageQual.replace(np.nan, 'No Garage')
        A.GarageType = A.GarageType.replace(np.nan, 'No Garage')
        A.GarageCond = A.GarageCond.replace(np.nan, 'No Garage')

        #Garage Year is tricky 
        A['HasGarage'] = A['GarageYrBlt'].apply(lambda x: 0 if pd.isna(x) else 1)
        A = A.drop(columns = ['GarageYrBlt']) #Just going to drop this for simplicity, ideally would figure out a way to do this
        return A 
    
class FixRest_2(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        """
        This function is a preprocessing step. The goal of this function is to fill any NAs in the
        data set that has meaning (derived from the meta data).
        """
        A = X.copy()
        A.PoolQC = A.PoolQC.replace(np.nan, 'No Pool').astype(str)
        A.MiscFeature = A.MiscFeature.replace(np.nan, 'No').astype(str)
        A.Alley = A.Alley.replace(np.nan, 'No Alley').astype(str)
        A.Fence = A.Fence.replace(np.nan, 'No Fence').astype(str)
        A.FireplaceQu = A.FireplaceQu.replace(np.nan, 'No Fireplace').astype(str)
        A.LotFrontage = A.LotFrontage.replace(np.nan, 0)
        A.BsmtFinType2 = A.BsmtFinType2.replace(np.nan, 'No Basement').astype(str)
        A.BsmtExposure = A.BsmtExposure.replace(np.nan, 'No Basement').astype(str)
        A.BsmtQual = A.BsmtQual.replace(np.nan, 'No Basement').astype(str)
        A.BsmtCond = A.BsmtCond.replace(np.nan, 'No Basement').astype(str)
        A.BsmtFinType1 = A.BsmtFinType1.replace(np.nan, 'No Basement').astype(str)
        A.MasVnrArea = A.MasVnrArea.replace(np.nan, 0)
        A.MasVnrType = A.MasVnrType.replace(np.nan, 'None').astype(str)
        #We still have one row that has a null value for electrical, since it is impossible to find what this means
        #I will just drop that row in the end 
        A = A.drop(columns = 'Electrical')
        A.Utilities = A.Utilities.replace(np.nan, 'AllPub').astype(str)
        A.KitchenQual = A.KitchenQual.replace(np.nan, 'Po').astype(str)
        A.Functional = A.Functional.replace(np.nan, 'Sal').astype(str)
        A.MSZoning = A.MSZoning.replace('C (all)', 'C').astype(str)
        return A   
    
    
class OrdEnc_3(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    def transform(self,X):
        """
        This function is to transform categorical columns into ordinal. I provided custom ordering based on 
        The meta data provided.
        """
        A = X.copy()
        A['MSSubClass'] = A['MSSubClass'].astype(str)
        A['LotShape'] = OrdinalEncoder(categories = [['Reg','IR1','IR2','IR3']]).fit_transform(A[['LotShape']])
        A['LandContour'] = OrdinalEncoder(categories = [['Lvl','Bnk','HLS','Low']]).fit_transform(A[['LandContour']])
        A['Utilities'] = OrdinalEncoder(categories = [['AllPub','NoSwer','NoSeWa','ELO']]).fit_transform(A[['Utilities']])
        A['LandSlope'] = OrdinalEncoder(categories = [['Gtl','Mod','Sev']]).fit_transform(A[['LandSlope']])
        A['BldgType'] = A['BldgType'].replace('Duplex','Duplx')
        A['BldgType'] = A['BldgType'].replace('2fmCon','2FmCon')
        A['BldgType'] = OrdinalEncoder(categories = [['1Fam','2FmCon', 'Duplx', 'TwnhsE', 'Twnhs']]).fit_transform(A[['BldgType']])
        A['ExterQual'] = OrdinalEncoder(categories = [['Ex','Gd','TA', 'Fa', 'Po']]).fit_transform(A[['ExterQual']])
        A['ExterCond'] = OrdinalEncoder(categories = [['Ex','Gd','TA', 'Fa', 'Po']]).fit_transform(A[['ExterCond']])
        A['BsmtQual'] = OrdinalEncoder(categories = [['Ex','Gd','TA', 'Fa', 'Po', 'No Basement']]).fit_transform(A[['BsmtQual']])
        A['BsmtCond'] = OrdinalEncoder(categories = [['Ex','Gd','TA', 'Fa', 'Po', 'No Basement']]).fit_transform(A[['BsmtCond']])
        A['BsmtExposure'] = OrdinalEncoder(categories = [['Gd','Av','Mn','No','No Basement']]).fit_transform(A[['BsmtExposure']])
        A['BsmtFinType1'] = OrdinalEncoder(categories = [['GLQ','ALQ','BLQ','Rec','LwQ','Unf','No Basement']]).fit_transform(A[['BsmtFinType1']])
        A['BsmtFinType2'] = OrdinalEncoder(categories = [['GLQ','ALQ','BLQ','Rec','LwQ','Unf','No Basement']]).fit_transform(A[['BsmtFinType2']])
        A['HeatingQC'] = OrdinalEncoder(categories = [['Ex','Gd','TA','Fa','Po']]).fit_transform(A[['HeatingQC']])
        A['KitchenQual'] = OrdinalEncoder(categories = [['Ex','Gd','TA','Fa','Po']]).fit_transform(A[['KitchenQual']])
        A['Functional'] = OrdinalEncoder(categories = [['Typ','Min1','Min2','Mod','Maj1','Maj2','Sev','Sal']]).fit_transform(A[['Functional']])
        A['FireplaceQu'] = OrdinalEncoder(categories = [['Ex','Gd','TA', 'Fa', 'Po', 'No Fireplace']]).fit_transform(A[['FireplaceQu']])
        A['GarageFinish'] = OrdinalEncoder(categories = [['Fin','RFn','Unf', 'No Garage']]).fit_transform(A[['GarageFinish']])
        A['GarageQual'] = OrdinalEncoder(categories = [['Ex','Gd','TA', 'Fa', 'Po', 'No Garage']]).fit_transform(A[['GarageQual']])
        A['GarageCond']= OrdinalEncoder(categories = [['Ex','Gd','TA', 'Fa', 'Po', 'No Garage']]).fit_transform(A[['GarageCond']])
        A['PavedDrive'] = OrdinalEncoder(categories = [['Y','P','N']]).fit_transform(A[['PavedDrive']])
        A['PoolQC'] = OrdinalEncoder(categories = [['Ex','Gd','TA', 'Fa', 'No Pool']]).fit_transform(A[['PoolQC']])
        A['Fence'] = OrdinalEncoder(categories = [['GdPrv','MnPrv','GdWo', 'MnWw', 'No Fence']]).fit_transform(A[['Fence']])
        return A 

class NAimp_4(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        """
        This function is a preprocessing step. If there are still missing values, the missing values are imputed by the mean
        of the column for integer types and the mode for object types.

        Parameters:
        X (DataFrame): The input DataFrame with potential missing values.

        Returns:
        DataFrame: The DataFrame with missing values imputed.
        """
        A = X.copy()
        for column in A.columns:
            if np.issubdtype(A[column].dtype, np.number):
                A[column].fillna(A[column].mean(), inplace=True)
            elif A[column].dtype == 'object':
                # Imputing with the first mode value (most frequent value)
                A[column].fillna(A[column].mode()[0], inplace=True)
        return A
    
class Scaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    def transform(self,X):
        scaler = StandardScaler()
        A = X.copy()
        scaler.fit(A)
        A = scaler.transform(A)
        return A 
    

class OneHot(BaseEstimator, TransformerMixin):
    def __init__(self, type_):
        self.type_ = type_
        if type_ not in ['parametric', 'non-p']:
            raise ValueError("type_ must be 'parametric' or 'non-p'")
    def fit(self, X, y = None):
        return self
    def transform(self,X):
        """
        Function to One hot Encode. 
        """
        #Create copy of original Input
        A = X.copy()
        original_cat_columns  = list(A.columns[A.dtypes == 'object'])
        # I will change all the categories in each categorical variable to a format
        # of {column_name}_category, such that it is easier to differentiate in the final data frame
        # and avoid any errors with columns that have the same name because of having identical category names.
        for i in original_cat_columns:
            A[i] = i + '_' + A[i].astype(str)
       
        #Define Encoder
        encoder = OneHotEncoder(handle_unknown = 'ignore')

        #Save Id column, I will concat this in the end.
        id_ = pd.DataFrame()
        id_['id'] = A.index
        #Get relevant categorical columns
        cat_columns  = list(A.columns[A.dtypes == 'object'])
        #Fit Encoder on categorical columnsof A 
        encoder.fit(A[cat_columns])
        #Transform categorical columns of A and change it to array representation
        temp = encoder.transform(A[cat_columns]).toarray()
        #Get the feature labels to use as column names in final DataFrame
        feature_labels = encoder.categories_
        #Get the labels into one dimensional array/ list to use as column names
        flattened_labels = np.concatenate(feature_labels)
        #Create the dataframe of categorical columns
        temp2 = pd.DataFrame(temp, columns = flattened_labels)
        col_to_drop = []
        if self.type_ == 'parametric':
            """
            if this is fitted to a parametric (linear regression) model,
            then we want to drop one category column per original category column to prevent perfect
            colinearity.
            """
            original_cat_columns  = list(A.columns[A.dtypes == 'object'])
            temp2_col_names = list(temp2.columns)
            for i in original_cat_columns:
                temp_3_col_names = [k for k in temp2_col_names if k.startswith(i + '_')]
                for k in temp2_col_names: 
                    if i in k:
                        col_to_drop.append(k)
                        break
        elif self.type_ == 'non-p':
            pass
        A = A.drop(columns = cat_columns).reset_index()
        temp2 = temp2.drop(columns = col_to_drop).reset_index()
        temp3 = pd.concat([A,temp2], axis = 1)
        temp4 = pd.concat([id_,temp3], axis = 1)
        temp4 = temp4.set_index('id')
        if 'index' in temp4.columns:
            temp4 = temp4.drop(columns = 'index')
        return temp4
    
class DropCat(BaseEstimator, TransformerMixin):
    def fit(self,X, y = None):
        return self
    def transform(self,X, y = None):
        A = X.copy()
        cat_columns = list(A.columns[A.dtypes == 'object'])
        A = A.drop(columns = cat_columns)
        return A 


# ## Load Data

# In[7]:


#Combine data 
df_train['df_train'] = 1
df_test['df_test'] = 1

df_temp = pd.concat([df_train,df_test])
df_temp['df_train'] = df_temp['df_train'].replace(np.nan, 0)
df_temp['df_test'] = df_temp['df_test'].replace(np.nan, 0)

df_temp = df_temp.drop(columns = 'Id')


# # Do transformations

# In[8]:


pipe_np = Pipeline([
    ('Step 1', GarageFix_1()),
    ('Step 2', FixRest_2()),
    ('Step 3', OrdEnc_3()),
    ('Step 4', NAimp_4()),
    ('Step 5', OneHot(type_ = 'non-p'))
                ])


pipe_p = Pipeline([
    ('Step 1', GarageFix_1()),
    ('Step 2', FixRest_2()),
    ('Step 3', OrdEnc_3()),
    ('Step 4', NAimp_4()),
    ('Step 5', OneHot(type_ = 'parametric'))
                ])

pipe_nc = Pipeline([
    ('Step 1', GarageFix_1()),
    ('Step 2', FixRest_2()),
    ('Step 3', OrdEnc_3()),
    ('Step 4', NAimp_4()),
    ('Step 5', DropCat())
                ])

pipe_cat = Pipeline([
    ('Step 1', GarageFix_1()),
    ('Step 2', FixRest_2()),
    ('Step 3', OrdEnc_3()),
    ('Step 4', NAimp_4())
                ])
pipe_np.fit(df_temp)
df_np = pipe_np.transform(df_temp) 


pipe_p.fit(df_temp)
df_p = pipe_p.transform(df_temp) 


pipe_nc.fit(df_temp)
df_nc = pipe_nc.transform(df_temp)


pipe_cat.fit(df_temp)
df_cat = pipe_cat.transform(df_temp)


# In[9]:


df_train_np = df_np[df_np.df_train == 1].drop(columns = ['df_train','df_test'])
X_train_np = df_train_np.drop(columns = 'SalePrice')
y_train = df_train_np['SalePrice']

df_train_p = df_p[df_p.df_train == 1].drop(columns = ['df_train','df_test'])
X_train_p = df_train_p.drop(columns = 'SalePrice')
y_train = df_train_p['SalePrice']

df_train_nc = df_nc[df_nc.df_train == 1].drop(columns = ['df_train','df_test'])
X_train_nc = df_train_nc.drop(columns = 'SalePrice')
y_train = df_train_nc['SalePrice']

df_train_cat = df_cat[df_cat.df_train == 1].drop(columns = ['df_train','df_test'])
X_train_cat = df_train_cat.drop(columns = 'SalePrice')
y_train = df_train_cat['SalePrice']

X_train_p


# ## Create Analysis and Assesment split

# In[10]:


(X_an_np,
X_as_np,
y_an_np,
y_as_np) = train_test_split(X_train_np, y_train, test_size = 0.3, random_state = 0)

(X_an_p,
X_as_p,
y_an_p,
y_as_p) = train_test_split(X_train_nc, y_train, test_size = 0.3, random_state = 0)

(X_an_nc,
X_as_nc,
y_an_nc,
y_as_nc) = train_test_split(X_train_nc, y_train, test_size = 0.3, random_state = 0)

(X_an_cat,
X_as_cat,
y_an_cat,
y_as_cat) = train_test_split(X_train_cat, y_train, test_size = 0.3, random_state = 0)


# # KNN

# a KNN model creates predictions based on the nearest 'neighbor' or observation from the training set. The hyper parameter `n_neighbors` determines how many nearest neighbours are being considered for the prediction of a new observation. It is a distance based model, so scaling is required

# In[11]:


get_ipython().run_cell_magic('time', '', "p_knn = Pipeline([\n        ('Scaling', Scaler()),\n        ('KNR', KNR())\n                ])\nparams_knn = {}\nparams_knn['KNR__n_neighbors'] = list(np.arange(1,100,2))\n\n\nkf = KFold(n_splits = 5, shuffle = True, random_state = 0)\ngrid_knn = RandomizedSearchCV(p_knn, params_knn, cv= kf, n_iter = 50, scoring= 'neg_root_mean_squared_error', verbose =1, error_score ='raise')\ngrid_knn.fit(X_an_np, y_an_np)\njoblib.dump(grid_knn, 'knn_grid.pkl')\ngrid_knn = joblib.load('knn_grid.pkl')\n\ndata_knn ={\n    'k': list(map(lambda x: x['KNR__n_neighbors'],grid_knn.cv_results_['params'])),\n    'score' :list(-1*grid_knn.cv_results_['mean_test_score']),\n    'std_error': list(grid_knn.cv_results_['std_test_score'])\n}\ntuned_knn_grid = pd.DataFrame(data = data_knn)\ntuned_knn_grid.head()\n")


# In[12]:


sns.lineplot(x = 'k', y = 'score', data = tuned_knn_grid)


# We see that for KNR, the best model lies somehwere in between 0 and 10, We will now apply one standard error rule to reduce the risk of overfitting.

# In[13]:


tuned_knn_grid.sort_values('score').head(10) #score =39708.003784 k = 5


# Apply 1SE Rule

# In[14]:


threshold_knn = min(tuned_knn_grid.score) + tuned_knn_grid.loc[tuned_knn_grid['score'].idxmax()].std_error
simpler_models_knn = tuned_knn_grid[tuned_knn_grid.k >= 5].sort_values('score')
simpler_models_knn = simpler_models_knn[simpler_models_knn.score <= threshold_knn]

simpler_models_knn #N = 31 is decided


# ## Finalise KNN

# In[15]:


p_knn_fin = Pipeline([
        ('Scaling', Scaler()),
        ('KNR', KNR(n_neighbors = 31))
                ])
p_knn_fin.fit(X_an_np,y_an_np)


# # 2. Lasso, Ridge, EN

# Lasso, Ridge, and Elastic Net are models that apply regularisation to the typical OLS. Lasso utilises an L1-Norm as the penalty to the objective function, whereas Ridge uses L2-Norm (for both `alpha`). Elastic Net on the other hand attempts to use both and give weights (`l1_ratio`) to both penalties. 
# 
# The main benefit of these models is the ability of feature shrinkage (EN, Lasso, & Ridge) and Feature Selection (Lasso only). 
# 
# Scaling is also essential for these models

# ## 2a. Lasso

# In[16]:


get_ipython().run_cell_magic('time', '', "#step to interact all the terms in the data set, we can do this with \n#regularised regression because it will automatically select for us which \n#features (including its interactions) to include.\ninteraction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)\n\np_lasso = Pipeline([\n    ('interaction',interaction),\n    ('scaler', Scaler()),\n    ('lasso', Lasso(tol = 0.0001))\n])\nparams_lasso = {}\nparams_lasso['interaction__degree'] = [2]\n#Important we define a wide enough penalty region \nparams_lasso['lasso__alpha'] = np.logspace(3,4.5,50) \n\nkf = KFold(n_splits = 5, shuffle = True, random_state = 0)\ngrid_lasso = GridSearchCV(p_lasso, params_lasso, cv= kf, scoring= 'neg_root_mean_squared_error', verbose =1, n_jobs = -1)\ngrid_lasso.fit(X_an_p, y_an_p)\n\ndata_lasso ={\n    'lambda': list(map(lambda x: x['lasso__alpha'],grid_lasso.cv_results_['params'])),\n    'score' :list(-1*grid_lasso.cv_results_['mean_test_score']),\n    'std_error': list(grid_lasso.cv_results_['std_test_score'])\n}\ntuned_lasso_grid = pd.DataFrame(data = data_lasso)\ntuned_lasso_grid.head()\n")


# In[17]:


sns.scatterplot(x = 'lambda', y = 'score', data = tuned_lasso_grid)


# In[18]:


tuned_lasso_grid.sort_values('score').head(6) #Best value lambda = 2500.110383


# In[19]:


threshold_lasso = min(tuned_lasso_grid.score) + tuned_lasso_grid.loc[tuned_lasso_grid['score'].idxmin()].std_error
simpler_models_lasso = tuned_lasso_grid[tuned_lasso_grid['lambda'] >= tuned_lasso_grid.loc[tuned_lasso_grid['score'].idxmin()]['lambda']]
simpler_models_lasso = simpler_models_lasso[simpler_models_lasso.score <= threshold_lasso]

simpler_models_lasso.sort_values('lambda') #lambda = 15627.069765


# ## Lasso finalise Model

# In[20]:


p_lasso_fin = Pipeline([
    ('scaler', Scaler()),
    ('interaction',interaction),
    ('lasso', Lasso(alpha= 15627.069765	,tol = 0.001))
])
p_lasso_fin.fit(X_an_p, y_an_p)


# Ridge and Elastic net follow the same steps, I will not do them

# # 3. Random Forest

# A Random Forest is an ensemble method that is built on top of Decision Trees. The idea of a random forest is that at each split a random and diffent subset of features are considered. The number of features considered is controlled by `max_features`, which we will tune over.

# ## 3a. With 1 HE

# Typically 1HE should be avoided when doing RF because this increases the dimensionality of the data, which is not ideal for tree based models. However, I will still try and fit a model with 1HE.
# 
# 1HE is especially bad when we have high cardinality in the dataset.

# In[21]:


get_ipython().run_cell_magic('time', '', "p_rf = Pipeline([('RF', RF(n_estimators = 250))])\n\nparams_rf = {} \nparams_rf['RF__max_features'] = np.arange(1,1022,2)\nparams_rf['RF__min_samples_leaf'] = [5,7]\n\nkf = KFold(n_splits = 5, shuffle = True, random_state = 0)\ngrid_rf = RandomizedSearchCV(p_rf, params_rf,n_iter =50,cv= kf, scoring= 'neg_root_mean_squared_error', verbose =3, n_jobs = -1)\ngrid_rf.fit(X_an_p, y_an_p)\n\njoblib.dump(grid_rf, 'rf_grid2.pkl')\ngrid_rf= joblib.load('rf_grid2.pkl')\n\ndata_rf ={\n    'max_feature': list(map(lambda x: x['RF__max_features'],grid_rf.cv_results_['params'])),\n    'min_n': list(map(lambda x: x['RF__min_samples_leaf'], grid_rf.cv_results_['params'])),\n    'score' :list(-1*grid_rf.cv_results_['mean_test_score']),\n    'std_error': list(grid_rf.cv_results_['std_test_score'])\n}\n\ntuned_rf_grid = pd.DataFrame(data = data_rf)\ntuned_rf_grid.head()\n")


# In[22]:


sns.lineplot(x = 'max_feature', y = 'score', hue = 'min_n', data = tuned_rf_grid)


# not every max_feature is tested for all min_n because we are using randomised Cv. Which randomly chooses the hyperparameter combination to test, however this is not entirely random. The function will test more in the space where there is good performance

# In[23]:


tuned_rf_grid.sort_values('score').head() #The best model  has a max_feature of 15 and min 5


# I will retune using only min_ < 9 because above this point it seems that the performance of the model degrades too much

# In[24]:


threshold_rf = min(tuned_rf_grid.score) + tuned_rf_grid.loc[tuned_rf_grid['score'].idxmin()].std_error
simpler_models_rf = tuned_rf_grid[
    (tuned_rf_grid.max_feature <= tuned_rf_grid.loc[tuned_rf_grid['score'].idxmin()].max_feature)
    & (tuned_rf_grid.min_n >= tuned_rf_grid.loc[tuned_rf_grid['score'].idxmin()].min_n)
]
simpler_models_rf = simpler_models_rf[simpler_models_rf.score <= threshold_rf]

simpler_models_rf.sort_values('score')


# This really does not make sense

# ### 3a. Finalise Fit

# In[25]:


p_rfhe_fin = Pipeline([('RF', RF(
    max_features = 15,
    min_samples_leaf = 5,
    n_estimators = 250))])
p_rfhe_fin.fit(X_an_np, y_an_np)


# ## 3b. Random Forest: Without Categorical variables

# In[26]:


get_ipython().run_cell_magic('time', '', "p_rfnc = Pipeline([('RF', RF(n_estimators = 250))])\n\nparams_rfnc = {} \nparams_rfnc['RF__max_features'] = np.arange(1,57,1)\nparams_rfnc['RF__min_samples_leaf'] = np.arange(3,10, 2)\n\nkf = KFold(n_splits = 5, shuffle = True, random_state = 0)\ngrid_rfnc = RandomizedSearchCV(p_rfnc, params_rfnc,n_iter =50,cv= kf, scoring= 'neg_root_mean_squared_error', verbose =3, n_jobs = -1)\ngrid_rfnc.fit(X_an_nc, y_an_nc)\n\njoblib.dump(grid_rfnc, 'rfnc_grid.pkl')\ngrid_rfnc= joblib.load('rfnc_grid.pkl')\n\ndata_rfnc ={\n    'max_feature': list(map(lambda x: x['RF__max_features'],grid_rfnc.cv_results_['params'])),\n    'min_n': list(map(lambda x: x['RF__min_samples_leaf'], grid_rfnc.cv_results_['params'])),\n    'score' :list(-1*grid_rfnc.cv_results_['mean_test_score']),\n    'std_error': list(grid_rfnc.cv_results_['std_test_score'])\n}\n\ntuned_rfnc_grid = pd.DataFrame(data = data_rfnc)\ntuned_rfnc_grid.head()\n")


# In[27]:


sns.scatterplot(x = 'max_feature', y = 'score', hue = 'min_n', data = tuned_rfnc_grid)


# In[28]:


tuned_rfnc_grid.sort_values('score').head() #Best model max_feature = 24 and min_n 3


# In[29]:


threshold_rfnc = min(tuned_rfnc_grid.score) + tuned_rfnc_grid.loc[tuned_rfnc_grid['score'].idxmin()].std_error
simpler_models_rfnc = tuned_rfnc_grid[
    (tuned_rfnc_grid.max_feature <= tuned_rfnc_grid.loc[tuned_rfnc_grid['score'].idxmin()].max_feature)
    & (tuned_rfnc_grid.min_n >= tuned_rfnc_grid.loc[tuned_rfnc_grid['score'].idxmin()].min_n)
]
simpler_models_rfnc = simpler_models_rfnc[simpler_models_rfnc.score <= threshold_rfnc]

simpler_models_rfnc.sort_values('score')
# Best model within 1SE is max_feature 3 and min_n 3.


# ## Fit 

# In[30]:


p_rfnc_fin = Pipeline([('RF', RF(
    max_features = 3,
    min_samples_leaf = 3,
    n_estimators = 250))])
p_rfnc_fin.fit(X_an_nc, y_an_nc)


# # 4. Boosting

# Boosting is also an ensemble method that combines weak learners (typically decision trees) to create a better model.
# 
# First, a null model is fitted to the data and the subsequent residuals of that model is obtained. Second, the boosting algorithm will use these residuals to train an additional model that attempts to correct the mistakes (residuals) of the previous model. This is done sequentially until it reaches the stopping criterion. Boosting adopts the concept of slow learning when correcting these residuals. 

# ## 4a. Without Categorical variables

# The Boosting Regressor in scikit learn has decision trees as its base learner, therefore, categorical variable with high cordinality is typically not desired. So, in this iteration we will not include any categorical variables.

# In[36]:


get_ipython().run_cell_magic('time', '', "p_gb_nc = Pipeline([('GBR', GBR())])\n\nparams_gb_nc = {} \nparams_gb_nc['GBR__learning_rate'] = [0.1,0.05,0.01,0.005,0.001]\nparams_gb_nc['GBR__n_estimators'] = np.arange(250,4000, 250)\nparams_gb_nc['GBR__max_depth'] = [1,3,5,7,9]\n\nkf = KFold(n_splits = 5, shuffle = True, random_state = 0)\ngrid_gb_nc = RandomizedSearchCV(p_gb_nc, params_gb_nc,cv= kf, n_iter = 100, scoring= 'neg_root_mean_squared_error', verbose =3, n_jobs = -1)\ngrid_gb_nc.fit(X_an_nc, y_an_nc)\n\ndata_gb_nc ={\n    'learning_rate': list(map(lambda x: x['GBR__learning_rate'],grid_gb_nc.cv_results_['params'])),\n    'n_estimators': list(map(lambda x: x['GBR__n_estimators'],grid_gb_nc.cv_results_['params'])),\n    'max_depth': list(map(lambda x: x['GBR__max_depth'],grid_gb_nc.cv_results_['params'])),\n    'score' :list(-1*grid_gb_nc.cv_results_['mean_test_score']),\n    'std_error': list(grid_gb_nc.cv_results_['std_test_score'])\n}\n\ntuned_gb_grid = pd.DataFrame(data = data_gb_nc)\ntuned_gb_grid\n")


# In[37]:


joblib.dump(grid_gb_nc, 'gb_nc_grid1.pkl')


# In[31]:


grid_gb_nc =  joblib.load('gb_nc_grid1.pkl')
data_gb_nc ={
    'learning_rate': list(map(lambda x: x['GBR__learning_rate'],grid_gb_nc.cv_results_['params'])),
    'n_estimators': list(map(lambda x: x['GBR__n_estimators'],grid_gb_nc.cv_results_['params'])),
    'max_depth': list(map(lambda x: x['GBR__max_depth'],grid_gb_nc.cv_results_['params'])),
    'score' :list(-1*grid_gb_nc.cv_results_['mean_test_score']),
    'std_error': list(grid_gb_nc.cv_results_['std_test_score'])
}
tuned_gb_grid = pd.DataFrame(data = data_gb_nc)


# In[32]:


g = sns.FacetGrid(tuned_gb_grid, col="learning_rate", hue="max_depth", palette='viridis', col_wrap=4)

# Draw a line plot on each Axes
g = g.map(plt.scatter, "n_estimators", "score").add_legend()

g.set_titles("Learning Rate: {col_name}")  # Set title for each facet
g.set_axis_labels("Trees", "RMSE")  # Set x and y axis labels

plt.show()


# In[33]:


# LearningRate 0.005 looks really good


# In[34]:


tuned_gb_grid.sort_values('score')


# In[35]:


threshold_gb = min(tuned_gb_grid.score) + tuned_gb_grid.loc[tuned_gb_grid['score'].idxmin()].std_error
simpler_models_gb = tuned_gb_grid[
    (tuned_gb_grid.n_estimators <= tuned_gb_grid.loc[tuned_gb_grid['score'].idxmin()].n_estimators)
    & (tuned_gb_grid.max_depth <= tuned_gb_grid.loc[tuned_gb_grid['score'].idxmin()].max_depth)
]
simpler_models_gb = simpler_models_gb[simpler_models_gb.score <= threshold_gb]

simpler_models_gb.sort_values('score') 

#I will choose learning_rate =0.01 and n_estimators= 1000 with max_depth = 1
#Seems to balance both score and std_error


# In[36]:


tuned_gb_grid[tuned_gb_grid.learning_rate == 0.01].sort_values('n_estimators')


# ### 4a. Finalise Fit 

# In[37]:


p_gb_nc_fin = Pipeline([('GBR', GBR(
                        learning_rate = 0.01,
                        n_estimators = 1000,
                        max_depth = 1))])
p_gb_nc_fin.fit(X_an_nc, y_an_nc)


# # 4b. CatBoost 

# In[38]:


cat_var = list(X_as_cat.dtypes[X_as_cat.dtypes == object].index)


# In[48]:


get_ipython().run_cell_magic('time', '', "p_catboost = Pipeline([('CatBoost', cb.CatBoostRegressor(cat_features = cat_var,loss_function = 'RMSE', random_state = 0))])\n\n\nparams_cb = {} \nparams_cb['CatBoost__learning_rate'] = [0.1,0.05,0.01,0.005,0.001]\nparams_cb['CatBoost__iterations'] = np.arange(250,4000,500)\nparams_cb['CatBoost__depth'] = [4,6,8]\n\nkf = KFold(n_splits = 5, shuffle = True, random_state = 0)\ngrid_cb = RandomizedSearchCV(p_catboost, params_cb,cv= kf, n_iter= 50,\n                        scoring= 'neg_root_mean_squared_error', verbose = 1, n_jobs = -1)\nnp.int = int\ngrid_cb.fit(X_an_cat, y_an_cat)\n")


# In[49]:


joblib.dump(grid_cb, 'cb_grid.pkl') #save model


# In[39]:


grid_cb = joblib.load('cb_grid.pkl')
data_cb ={
    'learning_rate': list(map(lambda x: x['CatBoost__learning_rate'],grid_cb.cv_results_['params'])),
    'n_estimators': list(map(lambda x: x['CatBoost__iterations'],grid_cb.cv_results_['params'])),
    'max_depth': list(map(lambda x: x['CatBoost__depth'],grid_cb.cv_results_['params'])),
    'score' :list(-1*grid_cb.cv_results_['mean_test_score']),
    'std_error': list(grid_cb.cv_results_['std_test_score'])
}

tuned_cb_grid = pd.DataFrame(data = data_cb)
tuned_cb_grid.head()


# In[40]:


g = sns.FacetGrid(tuned_cb_grid, col="learning_rate", hue="max_depth", palette=["#D55E00", "#0072B2", "#009E73"], col_wrap=4)

# Draw a line plot on each Axes
g = g.map(plt.scatter, "n_estimators", "score").add_legend()

g.set_titles("Learning Rate: {col_name}")  # Set title for each facet
g.set_axis_labels("Trees", "RMSE")  # Set x and y axis labels

plt.show()


# Looks like learning rate of 0.01, 0.05, and 0.1 is good 

# In[54]:


get_ipython().run_cell_magic('time', '', "p_catboost = Pipeline([('CatBoost', cb.CatBoostRegressor(cat_features = cat_var,loss_function = 'RMSE', random_state = 0))])\n\n\nparams_cb = {} \nparams_cb['CatBoost__learning_rate'] = [0.1,0.05,0.01]\nparams_cb['CatBoost__iterations'] = np.arange(250,4000,500)\nparams_cb['CatBoost__depth'] = [4,6,8]\n\nkf = KFold(n_splits = 5, shuffle = True, random_state = 0)\ngrid_cb = RandomizedSearchCV(p_catboost, params_cb,cv= kf, n_iter= 50,\n                        scoring= 'neg_root_mean_squared_error', verbose = 1, n_jobs = -1)\nnp.int = int\ngrid_cb.fit(X_an_cat, y_an_cat)\n")


# In[55]:


joblib.dump(grid_cb, 'cb_grid_r2.pkl') #save model


# In[41]:


grid_cb2 = joblib.load('cb_grid_r2.pkl')


# In[42]:


data_cb2 ={
    'learning_rate': list(map(lambda x: x['CatBoost__learning_rate'],grid_cb2.cv_results_['params'])),
    'n_estimators': list(map(lambda x: x['CatBoost__iterations'],grid_cb2.cv_results_['params'])),
    'max_depth': list(map(lambda x: x['CatBoost__depth'],grid_cb2.cv_results_['params'])),
    'score' :list(-1*grid_cb2.cv_results_['mean_test_score']),
    'std_error': list(grid_cb2.cv_results_['std_test_score'])
}

tuned_cb_grid2 = pd.DataFrame(data = data_cb2)
tuned_cb_grid2.head()


# In[43]:


g = sns.FacetGrid(tuned_cb_grid2, col="learning_rate", hue="max_depth", palette=["#D55E00", "#0072B2", "#009E73"], col_wrap=4)
g = g.map(plt.scatter, "n_estimators", "score").add_legend()  # Changed to plt.scatter for dot plot

g.set_titles("Learning Rate: {col_name}")  # Set title for each facet
g.set_axis_labels("Trees", "RMSE")  # Set x and y axis labels

plt.show()


# In[44]:


threshold_cb = min(tuned_cb_grid2.score) + tuned_cb_grid2.loc[tuned_cb_grid2['score'].idxmin()].std_error
simpler_models_cb = tuned_cb_grid2[
    (tuned_cb_grid2.n_estimators <= tuned_cb_grid2.loc[tuned_cb_grid2['score'].idxmin()].n_estimators)
    & (tuned_cb_grid2.max_depth <= tuned_cb_grid2.loc[tuned_cb_grid2['score'].idxmin()].max_depth)
]
simpler_models_cb = simpler_models_cb[simpler_models_cb.score <= threshold_cb]

simpler_models_cb.sort_values('score') 
#I will just choose learning_rate = 0.01, n_estimators = 1250, and max_depth = 6


# In[45]:


p_catboost_fin = Pipeline([('CatBoost',
                            cb.CatBoostRegressor(
                                learning_rate = 0.01,
                                depth = 6,
                                n_estimators = 1250,
                                cat_features = cat_var,loss_function = 'RMSE', random_state = 0))])

p_catboost_fin.fit(X_an_cat, y_an_cat)


# # Compare Model

# In[46]:


rfnc_rmse = np.sqrt(mean_squared_error(y_as_nc,p_rfnc_fin.predict(X_as_nc)))
gbnc_rmse = np.sqrt(mean_squared_error(y_as_nc,p_gb_nc_fin.predict(X_as_nc)))
rfhe_rmse = np.sqrt(mean_squared_error(y_as_nc,p_rfhe_fin.predict(X_as_np)))
lasso_rmse = np.sqrt(mean_squared_error(y_as_nc,p_lasso_fin.predict(X_as_p)))
knn_rmse = np.sqrt(mean_squared_error(y_as_nc,p_knn_fin.predict(X_as_np)))
cb_rmse = np.sqrt(mean_squared_error(y_as_nc,p_catboost_fin.predict(X_as_cat))) 


# In[47]:


rmse_values = {
    'Model': ['Random Forest NC', 'Gradient Boosting NC', 'CatBoost','Random Forest HE', 'Lasso', 'KNN'],
    'RMSE': [rfnc_rmse, gbnc_rmse,cb_rmse, rfhe_rmse, lasso_rmse, knn_rmse]
}

# Converting the dictionary into a DataFrame
rmse_df = pd.DataFrame(rmse_values)


# In[49]:


rmse_df_sorted = rmse_df.sort_values('RMSE')

# Creating a bar plot
plt.figure(figsize=(10, 6))
plt.barh(rmse_df_sorted['Model'], rmse_df_sorted['RMSE'], color='skyblue')
plt.xlabel('RMSE')
plt.title('Root Mean Squared Error (RMSE) Comparison')
plt.gca().invert_yaxis()  # Invert the y-axis to display the highest RMSE at the top

# Saving the plot as a PNG file
plt.savefig('rmse_comparison.png', bbox_inches='tight')

# Display the plot (optional)
plt.show()


# We choose Cat Boost

# In[50]:


p_catboost_fin.fit(X_train_cat,y_train)


# # Generate Final Predictions

# In[55]:


X_test = pipe_cat.transform(df_temp[df_temp.df_test == 1].drop(columns = 'SalePrice'))


# In[56]:


fin_pred = p_catboost_fin.predict(X_test)


# In[57]:


index_x_test = np.arange(1461,2920,1)


# In[58]:


index_x_test


# In[59]:


df_fin = pd.DataFrame(data = {
    'Id':index_x_test,
    'SalePrice': fin_pred }).set_index('Id')


# In[60]:


df_fin.to_csv('final_sub2.csv')

