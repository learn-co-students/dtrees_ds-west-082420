
# Decision Trees

# Agenda

1. FSM and Metric Discussion
2. Decision Trees at a High Level
3. ASM (Attribute Selection Methods): Entropy/Information Gain and Gini
4. Issues with Decision Trees: Overfitting, sensitivity to training data, greediness
5. Feature Importances
6. Grid Search

# Discussion Question

What effect does L1 lasso penalty have on model coefficients?  
What effect does L2 ridge penalty have on model coefficients?


```python
from src.student_caller import one_random_student, three_random_students
from src.student_list import student_first_names
one_random_student(student_first_names)
```


```python
#__SOLUTION__
"L1 Lasso zeros out the coefficients. L2 ridge shrinks the coefficients, but does not zero them out."
```


```python
import pandas as pd
import numpy as np

# This is always a good idea
%load_ext autoreload
%autoreload 2

import warnings
warnings.filterwarnings('ignore')

import os
import sys
module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)
```


```python
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

```

# 1. FSM and Metric Discussion


```python
# import requests
# import json

# response = requests.get('https://data.austintexas.gov/resource/9t4d-g238.json')
# animal_shelter = pd.DataFrame(response.json())
```


```python
# Let's return to our Austin Animal shelter data.

animal_shelter = pd.read_csv('data/austin.csv')
animal_shelter.set_index('Unnamed: 0', inplace=True)
```


```python
# We will convert the original dataframe shown below.
animal_shelter.head()
```


```python
from src.shelter_preprocess import preprocess_df

auo_na = animal_shelter.age_upon_outcome.isna().sum()
# We will convert age_upon_outcome to an integer
# and need to drop N/A's before applying a preprocessor.

animal_shelter.dropna(subset=['age_upon_outcome'], inplace=True )
print(f'{auo_na} age_upon_outome N/A\'s dropped')
```


```python
df = preprocess_df(animal_shelter)
df.head()
```

# New sklearn module: tree
https://scikit-learn.org/stable/modules/tree.html


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
# Per usual, let's set a test set off to the side.  
# We will build our model exclusively on the training set.

from sklearn.model_selection import train_test_split

X = df.drop(['adoption'], axis=1)
y = df.adoption

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size = .2)
X_train.head()

```


```python
# Let's do a secondary train/val split.  
# I'll set test_size to .25 to match my crossval 4 fold used later.

X_t, X_val, y_t, y_val = train_test_split(X_train,y_train, random_state=42, test_size = .25)

print(f"X_t   {X_t.shape[0]} and y_t   {y_t.shape[0]} have the same number of records\n")
print(f"X_val {X_val.shape[0]} and y_val {y_val.shape[0]} have the same number of records\n")

```

The methods associated with fitting decision tree classifier are similar to those attached to linear, logistic or knn.  

With your knowledge built up thus far, build an FSM: fit a dtree and score it on the validation set.


```python
one_random_student(student_first_names)
```


```python
# Instantiate the classifier object
dt = None

# Fit on training set

# Store predictions in a variable
y_hat_val = None

# Score on both training and validation sets

```


```python
#__SOLUTION__
dt = DecisionTreeClassifier()
dt.fit(X_t, y_t)
y_hat_val = dt.predict(X_val)
print(f'Training Score: {dt.score(X_t, y_t)}')
print(f'Val      Score: {dt.score(X_val, y_val)}')
```


```python
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

fig, ax = plt.subplots()
plot_confusion_matrix(dt, X_val, y_val, ax=ax, cmap='bwr')

ax.set_yticklabels(['no adoption', 'adoption'])
ax.set_xticklabels(['no adoption', 'adoption']);

```


```python
print(f'Accuracy Score: {accuracy_score(y_val, y_hat_val)}')
print('-------------------')
print(f'Precision score: {precision_score(y_val, y_hat_val)}')
print('-------------------')
print(f'Recall Score: {recall_score(y_val, y_hat_val)}')
print('-------------------')
print(f'f1_score: {f1_score(y_val, y_hat_val)}')
print('-------------------')
print(f'roc_auc_score{roc_auc_score(y_val, y_hat_val)}')

```

If you were building a model for the animal shelter, which metric would you focus on?

# 2. Decision Trees at a High Level

The **CART (Classification and Regression Trees) algorithm** is structured as a sequence of questions to try and split up the different observations into their own groups. The result of these questions is a tree like structure where the ends are **terminal nodes** (leaves) at which point there are no more questions.  A simple example of a decision tree is as follows:

<img src='./img/titanic_tree.png' width=600/>

A decision tree is a machine learning model that works by partitioning our sample space in a hierarchical way.

How do we partition the space? The key idea is that some attributes provide **more information** than others when trying to make a decision.

## High level of how the Decision Tree algorithm works

* Select the best attribute using Attribute Selection Measures (Gini/Entropy) to split the records.
* Make that attribute a decision node and break the dataset into smaller subsets.
* Starts tree building by repeating this process recursively for each child until one of these conditions will match:
    * You have reached a pure split: the leaf has only 1 class
    * There are no more remaining attributes to split on.
    * There are no more instances.

### Important Terminology related to Decision Trees
Let’s look at the basic terminologies used with Decision trees:

- **Root Node:** It represents entire population or sample and this further gets divided into two or more homogeneous sets.
- **Decision Node:** When a sub-node splits into further sub-nodes, then it is called decision node.
- **Leaf/ Terminal Node:** Nodes with no children (no further split) is called Leaf or Terminal node.
- **Pruning:** When we reduce the size of decision trees by removing nodes (opposite of Splitting), the process is called pruning.
- **Branch / Sub-Tree:** A sub section of decision tree is called branch or sub-tree.
- **Parent and Child Node:** A node, which is divided into sub-nodes is called parent node of sub-nodes where as sub-nodes are the child of parent node.


<img src='./img/decision_leaf.webp' width=600 />


```python
example_sample = df.sample(10, random_state=7)
example_sample.drop('age_in_days',axis=1, inplace=True)
example_sample.reset_index(inplace=True, drop=True)
example_sample.head(10)
```

## Partitioning

I partition my data by asking a question about the independent variables. The goal is to ask the right questions in the right order so that the resultant groups are "pure" with respect to the dependent variable. More on this below!

Suppose, for example, that I choose:

### Is the animal a dog?

This would divide my data into two groups:

- Group 1 (Dog = 0):

data points: 1,6,7

- Group 2 (Dog = 1):

data points:  0,2,3,4,5,8,9


#### Key Question: How are the values of the target distributed in this group?

In Group 1, I have: adoption, no adoption, no adoption

In Group 2, I have, in order: adoption, adoption, adoption, no adoption, no adoption, no adoption, adoption

This seems like an ok split:   
  - The first group is majority no adoptions (2/3), so it suggests cats are unlikely to get adopted.   
  - The second group has captured most of the adoptions, but there is only a slight majority of adoptions over no adoptions (4/7)

Would a different question split our data more effectively? Let's try:

### Is the animal female?

In pairs, examine whether this would be a better or worse split than the is_dog split.

> Take notes here


```python
#__SOLUTION__

'''
Let's look at our splits:

- Group 1 (Female = 0):

data points: 5,7,8

- Group 2 (Female = 1):


data points: 0,1,2,3,4,6,9


In Group 1, I have: no adoption, no adoption, no adoption  
In Group 2, I have: adoption, adoption, adoption, adoption, no adoption, no adoption, adoption

That seems better:  
  - Group 1 is a pure group, no adoption when sex is male
  - Group 2 has 5/7 adoptions.
  
This seems like a much better split.

So a (very simple!) model based on the above sample predicts:  
(i) A female animal will be adopted  
(ii) A male animal will not be adopted  

would perform fairly well.  

'''
```

But how would my partition be *best* split? How do I really know one split is better than the other? Can I do better than intuition here?  

# 3. Entropy/Information Gain and Gini

The goal is to have our ultimate classes be fully "ordered" (for a binary dependent variable, we'd have the 1's in one group and the 0's in the other). So one way to assess the value of a split is to measure how *disordered* our groups are, and there is a notion of *entropy* that measures precisely this.


```python
dt = DecisionTreeClassifier(criterion='entropy')
X = example_sample.drop('adoption', axis=1)
y = example_sample.adoption
dtree = dt.fit(X,y)
```


```python
from sklearn.tree import plot_tree
plot_tree(dt, feature_names=X.columns, filled=True);
```

The entropy of the whole dataset is given by:

$\large E = -\Sigma^n_i p_i\log_2(p_i)$,

where $p_i$ is the probability of belonging to the $i$th group, where $n$ is the number of groups (i.e. target values).

**Entropy will always be between 0 and 1. The closer to 1, the more disordered your group.**

<img src='./img/Entropy_mapped.png' width=600/>

To repeat, in the present case we have only two groups of interest: adoption and no adoption.

5 out of 10 were adopted and 5 out of 10 were not adopted, so **these are the relevant probabilities** for our calculation of entropy.

So our entropy for the sample above is:

$-0.5*\log_2(0.5) -0.5*\log_2(0.5)$.

Let's use the ``numpy's`` `log2()` function to calculate this:


```python
#__SOLUTION__
(-.5) * np.log2(.5) - (.5) * np.log2(.5)
```

Now, let's calculate the entropy of the training set.


```python
values, counts = None, None


p_0 = None
p_1 = None

def entropy(p_0,p_1):
    
    "entropy of a dataset with binary outcomes"
    
    pass
```


```python
#__SOLUTION__
values, counts = np.unique(y_t, return_counts=True)

print(values)
print(counts)

p_0 = counts[0]/sum(counts)
p_1 = counts[1]/sum(counts)

print(p_0, p_1)

def entropy(p_0,p_1):
    
    "entropy of a dataset with binary outcomes"
    
    entropy = (-p_0)*np.log2(p_0) - p_1*np.log2(p_1)
    return entropy

entropy(p_0, p_1)
```

To calculate the entropy of a *split*, we're going to want to calculate the entropy of each of the groups made by the split, and then calculate a weighted average of those groups' entropies––weighted, that is, by the size of the groups. Let's calculate the entropy of the split produced by our "is our animal a dog" question:

Group 1 (not a dog): adoption, no adoption, no adoption



$E_{g1} = - 2/3 * \log_2(2/3) - 1/3 * \log_2(1/3)$. 

Group 2: adoption, adoption, adoption, no adoption, no adoption, no adoption, adoption  
$E_{g2} = -\frac{3}{7} * \log_2\left(\frac{3}{7}\right) - \frac{4}{7} * \log_2\left(\frac{4}{7}\right)$.





```python
Eg1 = (-2/3)*np.log2(2/3) - (1/3)*np.log2(1/3)
print(Eg1)

Eg2 =(-3/7)*np.log2(3/7) - (4/7)*np.log2(4/7)
print(Eg2)
```

Now weight those by the probability of each group, and sum them, to find the entropy of the split:


```python
pg1 = (3/10) * Eg1
pg2 = (7/10) * Eg2

E_split_d = pg1 + pg2
E_split_d
```

Compare that to the male/female question:


```python
# In Group 1: no adoption, no adoption, no adoption  

Egm =  - (3/3)*np.log2(3/3) # This is a pure split, entropy is 0
print(Egm)

# In Group 2, I have: adoption, adoption, adoption, adoption, no adoption, no adoption, adoption

Egf =(-2/7)*np.log2(2/7) - (5/7)*np.log2(5/7) 
print(Egf)


```

Weighted sum


```python
pgm = Egm * 3/10
pgf = Egf * 7/10

E_split_f = pgm + pgf
E_split_f
```

For a given split, the **information gain** is simply the entropy of the parent group less the entropy of the split.


```python
total_entropy_sample = (-5/10)*np.log2(5/10) - (5/10) * np.log2(5/10)


# Information gain, dog or cat

ig_d = total_entropy_sample - E_split_d
print(f"Information gain dog/cat: {ig_d}")

ig_f = total_entropy_sample - E_split_f
print(f"Information gain male/female: {ig_f}")


```

For a given parent, then, we maximize our model's performance by *minimizing* the split's entropy.

What we'd like to do then is:

1. to look at the entropies of all possible splits, and
2. to choose the split with the lowest entropy.

In practice there are far too many splits for it to be practical for a person to calculate all these different entropies ...

... but we can make computers do these calculations for us!

## Gini Impurity

An alternative metric to entropy comes from the work of Corrado Gini. The Gini Impurity is defined as:

$\large G = 1 - \Sigma_ip_i^2$, or, equivalently, $\large G = \Sigma_ip_i(1-p_i)$.

where, again, $p_i$ is the probability of belonging to the $i$th group.

**Gini Impurity will always be between 0 and 0.5. The closer to 0.5, the more disordered your group.**

# Pairs (7 minutes)

Below, you will find a new Decision Tree Classifier with the hyperparameter *'criterion'* set to *'gini'*.

The decision tree is printed out just below.


```python
dt_gini = DecisionTreeClassifier(criterion='gini')

X = example_sample.drop('adoption', axis=1)
y = example_sample.adoption
dt_gini = dt_gini.fit(X,y)

plot_tree(dt_gini, feature_names=X.columns, filled=True);
```


```python
# In pairs, calculate the gini score for each split: is_female, is_dog. 

gini_is_female = None
gini_is_male = None
gini_total_malefemale = None

gini_is_dog = None
gini_is_cat = None
gini_total_cat_dog = None


```


```python
#__SOLUTION__

gini_is_female = 3/3*(1-3/3) + 0/3*(1-0/3)
gini_is_male = 2/7 * (1-2/7) + 5/7 * (1-5/7)
gini_total_malefemale = 3/10 * gini_is_female + 7/10* gini_is_male

print(gini_is_male)
print(gini_total_malefemale)

gini_is_dog = 1/2 * (1-1/2) + 1/2 * (1-1/2)
gini_is_cat = 1/5 * (1-1/5) + 4/5 * (1-4/5)
gini_total_cat_dog = 2/7*gini_is_dog + 5/7*gini_is_cat

print(gini_is_dog)
print(gini_total_cat_dog)
```

# Caveat


As found in *Introduction to Data Mining* by Tan et. al:

`Studies have shown that the choice of impurity measure has little effect on the performance of decision tree induction algorithms. This is because many impurity measures are quite consistent with each other [...]. Indeed, the strategy used to prune the tree has a greater impact on the final tree than the choice of impurity measure.`

# 4. Issues with Decision Trees
  
  - They are prone to overfitting.
  - They are highly dependent on the training data: small differences in training sets produce completely different trees.
  - They are susceptible to class imbalance.
  - They are greedy.
     



### Decision trees are prone to overfitting

Let's add back in age_in_days and refit the tree



```python
animal_shelter.dropna(subset=['age_upon_outcome'], inplace=True )
df = preprocess_df(animal_shelter)
df.shape
```


```python

X = df.drop(['adoption'], axis=1)
y = df.adoption
X.head()
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.2)
X_t, X_val, y_t, y_val = train_test_split(X_train, y_train, random_state=42, test_size=.25)
```


```python
dt = DecisionTreeClassifier()

dt.fit(X_t,y_t)

```


```python
plot_tree(dt, feature_names=X_t.columns, filled=True);
```

That is a good visual to go represent an overfit tree.  Let's look at the accuracy.


```python
# That is massive
dt.score(X_t,y_t)
```

That super high accuracy score is a telltale sign of an overfit model.


```python
# That's a bit dropoff
dt.score(X_val, y_val)
```

### Bias-Variance with Decision Trees

The CART algorithm will repeatedly partition data into smaller and smaller subsets until those final subsets are homogeneous in terms of the outcome variable. In practice this often means that the final subsets (known as the leaves of the tree) each consist of only one or a few data points. 

This results in low-bias, high variance trees.


## Stopping Criterion - Pruning Parameters


The recursive binary splitting procedure described above needs to know when to stop splitting as it works its way down the tree with the training data.

**min_samples_leaf:**  The most common stopping procedure is to use a minimum count on the number of training instances assigned to each leaf node. If the count is less than some minimum then the split is not accepted and the node is taken as a final leaf node.

**max_leaf_nodes:** 
Reduce the number of leaf nodes

**max_depth:**
Reduce the depth of the tree to build a generalized tree
Set the depth of the tree to 3, 5, 10 depending after verification on test data

**min_impurity_decrease :**
A node will split if its impurity is above the threshold, otherwise it is a leaf.


Let's try limiting the depth:


```python
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_t, y_t)
print(dt.score(X_t, y_t))
print(dt.score(X_val, y_val))
    
```


```python
fig, ax = plt.subplots(figsize=(10,10))
plot_tree(dt, feature_names=X_t.columns, filled=True, ax=ax)
plt.savefig('img/dt_md5.svg');
```

# Zoom in on this one

![dt_md5](img/dt_md5.svg)

Let's try limiting minimum samples per leaf:


```python
dt = DecisionTreeClassifier(min_samples_leaf=10)
dt.fit(X_t, y_t)
print(dt.score(X_t, y_t))
print(dt.score(X_val, y_val))
```


```python
fig, ax = plt.subplots(figsize=(25,25))
plot_tree(dt, feature_names=X_t.columns, filled=True, ax=ax)
plt.savefig('img/dt_msl10.svg')

fig, ax = plt.subplots(figsize=(25,25))
plot_tree(dt, feature_names=X_t.columns, filled=True, ax=ax, max_depth=3)
ax.set_title('Max Samples Leaf 10:\n Look at How the Left Most Leaf Stops Splitting', fontsize=30);
```


```python
dt = DecisionTreeClassifier(max_leaf_nodes=10)
dt.fit(X_t, y_t)
print(dt.score(X_t, y_t))
print(dt.score(X_val, y_val))
```


```python
fig, ax = plt.subplots(figsize=(25,25))
plot_tree(dt, feature_names=X_t.columns, filled=True, ax=ax)
ax.set_title('Max Leaf Nodes 10:', fontsize=30);
```

# Pairs (7 minutes)

Search for the optimal combination of hyperparamters w.r.t the accuracy of the validation set. 

Report back best parameters as well as validation score.





```python
# Your code here
```

# Grid Search

That is a lot of guess work.  Luckily, we have a tool called Grid Search which will make our lives a lot easier.  

Grid search takes a dictionary of paramater key words and values as an argument, iterates through all the possible combinations of those parameters and cross validates each model, then returns the parameter combination which performs best on the validation set.

While grid search is very convenient, it requires knowledge of what parameters each type of model requires as well as what values those paramters accept.  

It can also take a really long time.


```python
param_grid = [
  {'criterion': ['gini', 'entropy'], 
   'max_leaf_nodes': list(range(2,20)),
   'max_depth': list(range(2,10)),
   'min_impurity_decrease': np.linspace(0,1,10)} 
             ]
```


```python
from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=4, 
                  return_train_score=True, refit=True)
gs.fit(X_train, y_train)
```


```python
gs.__dict__.keys()
```


```python
gs.best_score_
```


```python
gs.best_params_
```


```python
gs.best_estimator_.score(X_test, y_test)
```

## Greediness

Decision trees will always split on the features with the most advantageous split. 

This could obscure valuable information held in those variables.  

For example, just for fun, let's dummy all of the colors of the animals.




```python
animal_shelter = pd.read_csv('data/austin.csv')
animal_shelter.set_index('Unnamed: 0', inplace=True)
animal_shelter.dropna(subset=['age_upon_outcome'], inplace=True )
color = animal_shelter.color
df = preprocess_df(animal_shelter)

# Add color back in to the dataset
df['color'] = color
df.color.value_counts()
```

The decision tree very rarely uses color as a split. Most often, age in days is chosen as the optimal split.  


```python
# Let's consolodate into Black/White/Other

def color_selector(color):
    if color == 'Black':
        return 'Black'
    elif color == 'White':
        return 'White'
    else:
        return 'other'
    
df['color'] = df['color'].apply(color_selector)
df.color.value_counts()
```


```python
X = df.drop('adoption', axis=1)
y = df.adoption

X = X.join(pd.get_dummies(X.color))
X.drop('color', axis=1, inplace=True)
X.head()
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.2)

param_grid = [
  {'criterion': ['gini', 'entropy'], 
   'max_leaf_nodes': list(range(2,20)),
   'max_depth': list(range(2,10)),} 
             ]

from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=4, 
                  return_train_score=True, refit=True)
gs.fit(X_train, y_train)
```


```python
gs.best_score_
```


```python
fig, ax = plt.subplots(figsize=(25,25))
plot_tree(gs.best_estimator_, feature_names=X_t.columns, filled=True, ax=ax)
ax.set_title('Max Leaf Nodes 10:', fontsize=30);
```


## Feature Importances

The fitted tree has an attribute called `ct.feature_importances_`. What does this mean? Roughly, the importance (or "Gini importance") of a feature is a sort of weighted average of the impurity decrease at internal nodes that make use of the feature. The weighting comes from the number of samples that depend on the relevant nodes.

> The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance. [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.feature_importances_)


```python
for fi, feature in zip(gs.best_estimator_.feature_importances_, X_train.columns):
    print(fi, feature)

```

More on feature importances [here](https://towardsdatascience.com/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3)

# Conclusions

Decision Tree is a white box type of ML algorithm. It shares internal decision-making logic, which is not available in the black box type of algorithms such as Neural Network. Its training time is faster compared to the neural network algorithm. The decision tree is a non-parametric method, which does not depend upon probability distribution assumptions. Decision trees can handle high dimensional data with good accuracy.

#### Pros
- Decision trees are easy to interpret and visualize.
- They can easily capture non-linear patterns.
- They require little data preprocessing from the user, for example, there is no need to normalize columns.
- They can be used for feature engineering such as predicting missing values, suitable for variable selection.
- Decision trees have no assumptions about distribution because of the non-parametric nature of the algorithm.

#### Cons
- Sensitive to noisy data. It can overfit noisy data.
- The small variation(or variance) in data can result in the different decision tree. This can be reduced by bagging and boosting algorithms.
- Decision trees are biased with imbalance dataset, so it is recommended that balance out the dataset before creating the decision tree.
