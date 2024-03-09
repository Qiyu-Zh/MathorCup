import numpy as np
import xgboost as xgb
from tqdm import tqdm
from tpot.builtins import StackingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# 0.5721835946336085
class Model():
    def __init__(self, w = None):
        # # 0.5512405417992985
        # self.XGBandRFR = make_pipeline(
        #     StackingEstimator(estimator=xgb.XGBRegressor(learning_rate=0.001,
        #                                                     max_depth=1,
        #                                                     min_child_weight=2,
        #                                                     n_estimators=100,
        #                                                     objective='reg:squarederror',
        #                                                     subsample=0.6500000000000001)
        #                                                 ),
        #     RandomForestRegressor(bootstrap=False,
        #                             max_features=0.5,
        #                             min_samples_leaf=2,
        #                             min_samples_split=2
        #                             )
        # )
        
        # # 0.5484298843279193
        # self.RFRandRFR = make_pipeline(
        #     StackingEstimator(estimator=RandomForestRegressor(
        #         bootstrap=True,
        #         max_features=0.5,
        #         min_samples_leaf=1,
        #         min_samples_split=11,
        #         n_estimators=100)),
        #     RandomForestRegressor(bootstrap=False,
        #                         max_features=0.35000000000000003,
        #                         min_samples_leaf=8,
        #                         min_samples_split=17,
        #                         n_estimators=100
        #                         )
        # )

        # 0.5565362135449956
        self.RFR = RandomForestRegressor(bootstrap=False,
                                        max_features=0.5,
                                        min_samples_leaf=2,
                                        min_samples_split=2,
                                        n_estimators=100
                                        )
        setattr(self.RFR, 'random_state', 42)
        
        # 0.5282568542645305
        self.XGB = xgb.XGBRegressor(eta=0.1,
                                    eval_metric='mae',
                                    gamma=0.1,
                                    max_depth=13,
                                    n_estimators=190
                                    )
        
        # 0.5273454131783123
        self.GBR = GradientBoostingRegressor(alpha=0.99,
                                            learning_rate=0.1,
                                            loss='lad',
                                            max_depth=9,
                                            max_features=0.6,
                                            min_samples_leaf=14,
                                            min_samples_split=10,
                                            n_estimators=100,
                                            subsample=1.0
                                            )
        
        # 0.5413237528920519
        self.ETRandRFR = make_pipeline(
            StackingEstimator(estimator=ExtraTreesRegressor(
                bootstrap=False,
                max_features=0.4,
                min_samples_leaf=1,
                min_samples_split=10,
                n_estimators=100)
                ),
            RandomForestRegressor(bootstrap=False,
                                max_features=0.55,
                                min_samples_leaf=12,
                                min_samples_split=16,
                                n_estimators=100
                                )
        )
        
        # 0.5548067700477394
        self.GBR_enhanced = GradientBoostingRegressor(alpha=0.99,
                                                    learning_rate=0.1,
                                                    loss="huber",
                                                    max_depth=9,
                                                    max_features=0.6000000000000001,
                                                    min_samples_leaf=8,
                                                    min_samples_split=10,
                                                    n_estimators=100,
                                                    subsample=1.0
                                                    )
        
        # 0.5332544073820505
        self.XGB_enhanced = xgb.XGBRegressor(eta=0.1,
                                            gamma=0,
                                            max_depth=12,
                                            n_estimators=230,
                                            objective=self.mape
                                            )
        
        # # 0.550494357903349
        # self.RFR_enhanced = RandomForestRegressor(bootstrap=False,
        #                                     max_features=0.5,
        #                                     min_samples_leaf=3,
        #                                     min_samples_split=5,
        #                                     n_estimators=100)
        
        # 0.5457399012228348
        self.RFRandETRandENCV = make_pipeline(
            StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.8, tol=0.001)),
            StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.55, min_samples_leaf=2, min_samples_split=17, n_estimators=100)),
            RandomForestRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=8, min_samples_split=14, n_estimators=100)
        )
        
        
        

        if w == None:
            self.w = np.array([0.02565457, 0.03410536, 0.07077384, 0.19386866, 0.39652538, 0.0982651, 0.18080708999999995])
        else:
            self.w = w
        self.models = [self.RFR, self.XGB, self.GBR, self.ETRandRFR, self.GBR_enhanced, self.XGB_enhanced, self.RFRandETRandENCV]
        self.trained = False
            
    def fit(self, X, y):
        print('Fitting Models:')
        for model in tqdm(self.models):
            model.fit(X, y)
        self.trained = True
        return 
        
    def predict(self, X):
        print('Model Predicting:')
        if self.trained:
            result = []
            for model in self.models:
                result.append(model.predict(X))
            final_pred = 0
            for i in tqdm(range(len(self.models))):
                final_pred += self.w[i] * result[i]
            return final_pred
        else:
            print("Models are not trained.")
            
    def mape(self, dtrain, preds):
        d = preds - dtrain
        h = 1
        scale = 1 + (d / h) ** 2
        scale_sqrt = np.sqrt(scale)
        grad = d / scale_sqrt
        hess = 1 / scale / scale_sqrt
        return grad, hess