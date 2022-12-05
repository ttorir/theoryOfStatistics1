import pandas as pd
from semopy import Model, gather_statistics
import numpy as np
import itertools as itertools
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


houses = pd.read_pickle('homeDataCSV.pkl')[['condition','sqft_living','waterfront','grade','price']]
houses.columns = ['x1','x2','x3','eta1','eta2']

# parameter values

x1_e1 = 1.2967
x2_e1 = 0.0015
mod_x2_e1 = 0.05 
x2_e2 = 244.1761
x3_e2 = 8.463*(10**5)
e1_e2 = 4627.9743

# mods
sample_sizes = [50, 100, 150, 200, 250, 500] # directly from sharma methods
x2_e1_mods = [x2_e1+(z*mod_x2_e1) for z in range(-2,3,1)] # + -2, -1, 0, 1, 2 mod
# mimic the ave cases evaluated by Sharma, 2019 (not using the r library that they used, as well as a boolean variable with x3)


"""
These models are generated from Figure 1 of Sharma, 2019
"""
models_dict= {
        # incorrect model
        'model_1':'''eta1 ~ x1 + x3
          eta2 ~ x1 + x2 + eta1''',
        # parsimonious model
        'model_2':'''eta1 ~ x1
          eta2 ~ x2 + x3 + eta1''',
        # incorrect model
        'model_3': '''eta1 ~ x1 + x2 + x3
          eta2 ~ x1 + eta1''',
        # incorrect model
        'model_4': '''eta1 ~ x3
          eta2 ~ x1 + x2 + eta1''',
        # data generation model
        'model_5': f'''eta1 ~ {str(x1_e1)}*x1 +  {str(x2_e1)}*x2 
                eta2 ~ {str(x2_e2)}*x2 + {str(x3_e2)}*x3 + {str(e1_e2)}*eta1''',
        # incorrect model
        'model_6': '''eta1 ~ x1 + x2 + x3
          eta2 ~ eta1''',
        # saturated model
        'model_7': '''eta1 ~ x1 + x2 + x3
          eta2 ~ x1 + x2 + x3 + eta1'''
          }

def get_aic(model_stats):
    """Compute best fit by AIC"""
    aic_stats = dict([(k,v.aic) for k,v in model_stats.items()])
    return min(aic_stats, key=aic_stats.get)

def get_aicc(y_pred, model_stats):
    """Compute best fit by AICc"""
    k = 4
    n = len(list(y_pred.values())[0])
    aicc_stats = dict([(key,(v.aic + ((2*(k**2)+ 2*k)/(n-k-1)))) for key,v in model_stats.items()])
    return min(aicc_stats, key=aicc_stats.get)

def get_bic(model_stats):
    """Compute best fit by BIC"""
    bic_stats = dict([(k,v.bic) for k,v in model_stats.items()])
    return min(bic_stats, key=bic_stats.get)

def get_gfi(model_stats):
    """Compute best fit by GFI"""
    gfi_stats = dict([(k,v.gfi) for k,v in model_stats.items()])
    return max(gfi_stats, key=gfi_stats.get)

def calc_r2(y, y_test):
        model = LinearRegression()
        model.fit(y_test, y)
        return model.score(y_test, y)

def get_r2(y, y_pred):
    """Compute best fit by R^2"""
    r2_stats = dict([(k, calc_r2(y, v)) for k,v in y_pred.items()])
    return max(r2_stats, key=r2_stats.get)

def get_r2_adj(y, y_pred):
    """Compute best fit by Adjusted R^2"""
    r2_stats = dict([(k, calc_r2(y, v)) for k,v in y_pred.items()])
    r2_adj_stats = dict([(k, 1-(1-v)*(len(y)-1)/(len(y)-4-1)) for k,v in r2_stats.items()])
    return max(r2_adj_stats, key=r2_adj_stats.get)

np.random.seed(0)

"""
These models are generated from Figure 1 of Sharma, 2019
"""
def generate_n_simulations(n):
    # initialize dataframe to store results in
    stats = {'AIC': get_aic,
                'AICc': get_aicc,
                'BIC': get_bic,
                'gfi': get_gfi,
                'R^2': get_r2,
                'Adjusted R^2':get_r2_adj
                }

    case_I_results_df = pd.DataFrame(0, index=stats.keys(), columns=models_dict.keys())
    case_II_results_df = pd.DataFrame(0, index=stats.keys(), columns=['model_1', 'model_2', 'model_3', 'model_4', 'model_6', 'model_7'])


    seed_count = 0
    for i in tqdm(range(n)):
        if (i%25 == 0):
          print(f'Current Status: {i}')
        for x2_e1_mod in x2_e1_mods:

            model_eqn = f'''eta1 ~ {str(x1_e1)}*x1 +  {str(x2_e1_mod)}*x2 
                    eta2 ~ {str(x2_e2)}*x2 + {str(x3_e2)}*x3 + {str(e1_e2)}*eta1'''

            for sample_size in sample_sizes:
                for n_waterfront in [5, 10, 25, 45]:
                    test_values = houses[['x1','x2','x3']].sample(n=50)
                    
                    not_waterfront_data = houses[houses.x3 == 0].sample(n=2*(sample_size-n_waterfront))
                    waterfront_data = houses[houses.x3 == 1].sample(n=(2*n_waterfront))
                    sample_data = pd.concat([not_waterfront_data, waterfront_data]).sort_values('x3')

                    model = Model(model_eqn)
                    model.fit(sample_data.iloc[0::2]) # to train the model for synthetic data
                    synthetic_data = model.predict(sample_data.iloc[1::2][['x1','x2','x3']])
                    
                    y = model.predict(test_values)[['eta2']].values #list(itertools.chain(*model.predict(test_values)[['eta1','eta2']].values))

                    all_model_stats = {}
                    all_model_y_pred = {}
                    for model_num, model_def in models_dict.items():
                        mod = Model(model_def)
                        mod.fit(synthetic_data)
                        all_model_y_pred[model_num] = mod.predict(test_values)[['eta2']].values #list(itertools.chain(*mod.predict(test_values)[['eta1','eta2']].values))
                        all_model_stats[model_num] = gather_statistics(mod)

                    def update_df(df, all_model_stats, all_model_y_pred):
                        df.loc['AIC',stats['AIC'](all_model_stats)] += 1
                        df.loc['AICc',stats['AICc'](all_model_y_pred, all_model_stats)] += 1
                        df.loc['BIC',stats['BIC'](all_model_stats)] += 1
                        df.loc['gfi',stats['gfi'](all_model_stats)] += 1
                        df.loc['R^2',stats['R^2'](y, all_model_y_pred)] += 1
                        df.loc['Adjusted R^2',stats['Adjusted R^2'](y, all_model_y_pred)] += 1

                    # case I
                    update_df(case_I_results_df, all_model_stats, all_model_y_pred)

                    # case II
                    all_model_stats.pop('model_5')
                    all_model_y_pred.pop('model_5')
                    update_df(case_II_results_df, all_model_stats, all_model_y_pred)

                    # make sure seed is changing for sample variation (but in a recreatable way)
                    seed_count += 1
                    np.random.seed(seed_count)

    return case_I_results_df, case_II_results_df


case_I_results_df, case_II_results_df = generate_n_simulations(100)

case_I_results_df.to_pickle('case_I.pkl')
case_II_results_df.to_pickle('case_II.pkl')