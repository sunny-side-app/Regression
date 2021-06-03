# https://www.python.org/dev/peps/pep-0263/
# -*- coding: utf-8 -*-

# nonlinear regressioinにおける信頼区間、予測区間の計算
# 基本的な実装は次を参考にするが、一部に誤りがあり、それを修正した一連の関数を実装する
# https://apmonitor.com/che263/index.php/Main/PythonRegressionStatistics
# 修正する際の数式は主に次の論文を参照
# Rasmussen, B. et al. (2004) https://trace.tennessee.edu/utk_graddiss/2379/
# 僅かにだが次も参照
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6027739/

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt    
from scipy import stats
import pandas as pd
import uncertainties.unumpy as unp
import uncertainties as unc


# # pip install uncertainties, if needed
# try:
    
# except:
#     try:
#         from pip import main as pipmain
#     except:
#         from pip._internal import main as pipmain
#     pipmain(['install','uncertainties'])
#     import uncertainties.unumpy as unp
#     import uncertainties as unc

# import data
url = 'https://apmonitor.com/che263/uploads/Main/stats_data.txt'
data = pd.read_csv(url)
x = data['x'].values
y = data['y'].values
n = len(y)

def f(x, a, b, c):
    return a * np.exp(b*x) + c

popt, pcov = curve_fit(f, x, y)

# retrieve parameter values
a = popt[0]
b = popt[1]
c = popt[2]
print('Optimal Values')
print('a: ' + str(a))
print('b: ' + str(b))
print('c: ' + str(c))

# compute r^2
r2 = 1.0-(sum((y-f(x,a,b,c))**2)/((n-1.0)*np.var(y,ddof=1)))
print('R^2: ' + str(r2))

# calculate parameter confidence interval
a,b,c = unc.correlated_values(popt, pcov)
print('Uncertainty')
print('a: ' + str(a))
print('b: ' + str(b))
print('c: ' + str(c))

# plot data
plt.scatter(x, y, s=3, label='Data')

# calculate regression confidence interval
px = np.linspace(14, 24, 100)
py = a*unp.exp(b*px)+c
nom = unp.nominal_values(py)
std = unp.std_devs(py)

def model(x,a,b,c):
    """
    Return model value to input x.
    """
    return a * np.exp(b*x) + c

def calc_rss(param, x, y, model):
    """
    Return RSS(Residual Sum of Squares).
    """
    a, b, c = param
    N = len(x)
    rss = np.sum(np.power(y - model(x,a,b,c), 2))
    mse = rss / N
    return rss

def calc_partial_derivative(param, x):
    """
    Return partial derivative of model (= (partial f(x;theta)) / (partial theta)).
    """
    a, b, c = param
    partial_a = np.exp(b * x)
    partial_b = a * b * np.exp(b * x)
    partial_c = np.ones(len(x))
    # return array of len(x) by len(param)(=3)
    return np.stack([partial_a, partial_b, partial_c], axis=1)

def calc_covariance_matrix(param, x, y, model):
    """
    Return covariance matrix in non-linear regression.
    Retuned value is equivalent to "S" (Chryssolouris, 1996)
    (see: Rasmussen, B. et al. (2004), p81)
    S: len(param)(=3) by len(param) matrix
    """
    a,b,c = param
    dim_param = len(param)
    n = len(x)
    
    # Calcurate noise variance
    rss = calc_rss(param, x, y, model)
    noise_var = rss / (n - dim_param)
    
    # Calculate Jacobi matrix (denoted "F" in Rasmussen,B. et al.(2004))
    # len(x) by len(param)
    Jacobi_mat = calc_partial_derivative(param, x)

    S = noise_var * np.linalg.inv(np.transpose(Jacobi_mat) @ Jacobi_mat)

    print("noise_var---------", noise_var)
    print("partial-----------", Jacobi_mat)
    print("S-----------", S)

    # 逆行列をとりnoise_varを掛ける
    return S

def calc_variance_of_regression(x_pred, param, S):
    """
    x_pred: requested points
    S: covariance matrix of model
    Return Var[f(x;theta)].
    """
    # len(x) by len(param)
    Jacobi_mat_pred = calc_partial_derivative(param, x_pred)
    # len(x) by len(x)
    var_of_reg = Jacobi_mat_pred @ S @ np.transpose(Jacobi_mat_pred)

    return np.diag(var_of_reg)


def predband(x, xd, yd, p, func, conf=0.95):
    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters
    # func = function name
    alpha = 1.0 - conf    # significance
    N = xd.size          # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * \
                 np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb

print("rss", calc_rss(popt,x,y,model))
print("derivative--------")
calc_partial_derivative(popt, x)
print("covariance--------")
SS = calc_covariance_matrix(popt, x, y, model)
print("SS------------", SS)
result = calc_variance_of_regression(px, popt, SS)
print("se-------", np.sqrt(result))
print('std------', std)
lpb, upb = predband(px, x, y, popt, f, conf=0.95)

# plot the regression
plt.plot(px, nom, c='black', label='y=a exp(b x) + c')

# uncertainty lines (95% confidence)
plt.plot(px, nom - 1.96 * std, c='orange',\
         label='95% Confidence Region')
plt.plot(px, nom + 1.96 * std, c='orange')
plt.plot(px, nom - 1.96 * np.sqrt(result), c='green',\
         label='95% Confidence Region (Scratch)')
plt.plot(px, nom + 1.96 * np.sqrt(result), c='green')


# prediction band (95% confidence)
plt.plot(px, lpb, 'k--',label='95% Prediction Band')
plt.plot(px, upb, 'k--')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(loc='best')

# save and show figure
plt.savefig('regression.png')
plt.show()



