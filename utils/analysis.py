import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lifelines
import math
import os

from lifelines.utils import concordance_index

def plt_km(results):
    '''
    Plot Kaplan Meier curve for low and high risk groups
    
    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing 'time_to_event', 'event', and 'lowrisk' columns
    '''
    low_risk = results.loc[results['lowrisk']]
    high_risk = results.loc[~results['lowrisk']]

    kmf = lifelines.KaplanMeierFitter()
    kmf.fit(low_risk['time_to_event'], low_risk['event'], label="low risk")
    ax = kmf.plot_survival_function()

    kmf2 = lifelines.KaplanMeierFitter()
    kmf2.fit(high_risk['time_to_event'], high_risk['event'], label="high risk")
    kmf2.plot_survival_function(ax=ax)
    # add lifelines.plotting.add_at_risk_counts(kmf) to the bottom of your plot
    lifelines.plotting.add_at_risk_counts(kmf,kmf2, ax=ax)

    # add a box that shows the p value and the hazard ratio 
    hr, ci_lower, ci_upper, p = get_hazard(results)
    plt.text(0.5, 0.5, 'HR = {:.3f}\nCI [{:.3f}, {:.3f}]\np = {:.3f}'.format(hr, ci_lower, ci_upper, p), fontsize=10, bbox=dict(facecolor='gray', alpha=0.1))
    plt.ylabel('probability')
    plt.ylim(0, 1)
    plt.title('Model Kaplan Meier curve')

def plt_km_tri(results, risk_column='risk_group'):
    '''
    Plot Kaplan Meier curve for low, medium, and high risk groups
    
    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing 'time_to_event', 'event', and risk group column
    risk_column : str
        Name of the column containing risk group labels ('low', 'medium', 'high')
    '''
    low_risk = results.loc[results[risk_column] == 'low']
    medium_risk = results.loc[results[risk_column] == 'medium']
    high_risk = results.loc[results[risk_column] == 'high']

    kmf_low = lifelines.KaplanMeierFitter()
    kmf_low.fit(low_risk['time_to_event'], low_risk['event'], label="low risk")
    ax = kmf_low.plot_survival_function()

    kmf_medium = lifelines.KaplanMeierFitter()
    kmf_medium.fit(medium_risk['time_to_event'], medium_risk['event'], label="medium risk")
    kmf_medium.plot_survival_function(ax=ax)

    kmf_high = lifelines.KaplanMeierFitter()
    kmf_high.fit(high_risk['time_to_event'], high_risk['event'], label="high risk")
    kmf_high.plot_survival_function(ax=ax)
    
    lifelines.plotting.add_at_risk_counts(kmf_low, kmf_medium, kmf_high, ax=ax)

    plt.ylabel('probability')
    plt.ylim(0, 1)
    plt.title('Model Kaplan Meier curve (Three Risk Groups)')

def get_hazard(results):
    '''
    Calculate hazard ratio, confidence intervals, and p-value between low and high risk groups
    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing 'time_to_event', 'event', and 'lowrisk' columns
    Returns
    -------
    hr : float
        Hazard ratio
    ci_lower : float
        Lower bound of the confidence interval
    ci_upper : float
        Upper bound of the confidence interval
    p : float
        p-value
    '''
    cph = lifelines.CoxPHFitter()
    cph.fit(results[['lowrisk', 'time_to_event', 'event']], duration_col='time_to_event', event_col='event')
    hr = float(cph.hazard_ratios_[0])
    ci_lower = math.exp(float(cph.confidence_intervals_.iloc[0][0]))
    ci_upper = math.exp(float(cph.confidence_intervals_.iloc[0][1]))
    p = cph.log_likelihood_ratio_test().p_value

    return hr, ci_lower, ci_upper, p


def baseline_model(k, splits_dir, data, lowrisk_percentile):
    data['slide_id'] = data['slide_id'].astype(str)
    results = []

    for k in range(k):
        split = pd.read_csv(os.path.join(splits_dir, 'splits_{}.csv'.format(k)))
        train = pd.DataFrame()
        train['slide_id'] = split['train'].astype(str)
        train = train.merge(data, on='slide_id')[['event', 'time_to_event', 'rop', 'rpp']]
        test = pd.DataFrame()
        test['slide_id'] = split['test'].astype(str)
        test = test.merge(data, on='slide_id')[['event', 'time_to_event', 'rop', 'rpp']]
        
        train.dropna(inplace=True)
        test.dropna(inplace=True)
        print('train: {}, test: {}'.format(train.shape, test.shape))

        results.append(fit_model(train, test, lowrisk_percentile))
    
    return pd.concat(results)

def fit_model(train, test, lowrisk_percentile):

    # fit cox model
    cph = lifelines.CoxPHFitter(penalizer=0.1)
    cph.fit(train, duration_col='time_to_event', event_col='event')
    predictions = cph.predict_partial_hazard(train)
    thresh_perc = np.percentile(predictions, lowrisk_percentile)

    # run inference and split into high / low risk groups
    predicted_hazard = cph.predict_partial_hazard(test)
    test['predicted_hazard'] = predicted_hazard
    test['lowrisk'] = test['predicted_hazard'] < thresh_perc

    return test

def c_index_lifelines(risks, events, times):
    risks = risks.reshape(-1)
    times = times.reshape(-1)
    label = []
    risk = []
    surv_time = []
    for i in range(len(risks)):
        if not np.isnan(risks[i]):
            label.append(events[i])
            risk.append(risks[i])
            surv_time.append(times[i])
    new_label = np.asarray(label)
    new_risk = np.asarray(risk)
    new_surv = np.asarray(surv_time)
    return concordance_index(new_surv, -new_risk, new_label)