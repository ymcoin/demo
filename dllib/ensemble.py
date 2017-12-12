import multiprocessing
import random
from functools import partial
from sklearn import metrics

from sklearn.metrics import confusion_matrix

from dllib.util import summary_result, load_data_from_path, get_tpr_tnr, combine_log
from dllib import settings
from dllib.classifier import csdnn_classifier
from dllib.log import get_logger
import numpy as np

class Ensemble(object):
    def __init__(self,data_list,param,type):
        self.data_list = data_list
        self.param = param
        self.log_dir = settings.log_dir
        self.type = type
        self.predict_probs = None
        self.eval_result = None
        self.log_files = []
        self.actual_y = None
    def create_experiments(self):
        exp_list = []

        if (self.type =='csdnn'):
            for i,(train,test) in enumerate(self.data_list):
                exp = self.param.copy()
                exp['train_path'] = train
                exp['valid_path'] = train
                exp['test_path'] = test
                exp['exp_name']='learner_'+str(i+1)
                exp['cost_vec'] = [1,1]
                exp_list.append(exp)
        elif(self.type =='csdnn_dynamic_cost'):
            # generate randomly cost vec
            min, max, step = self.param['cost_vec']
            l = 1+np.arange(min, max, step)
            for i, (train, test) in enumerate(self.data_list):
                exp = self.param.copy()
                exp['train_path'] = train
                exp['valid_path'] = train
                exp['test_path'] = test
                exp['exp_name'] = 'learner_' + str(i + 1)
                a = l[np.random.randint(0, len(l))]
                b = l[np.random.randint(0, len(l))]
                if(i%2==0):

                    while(a<b):
                        a = l[np.random.randint(0, len(l))]
                        b = l[np.random.randint(0, len(l))]
                else:
                    while (a >= b):
                        a = l[np.random.randint(0, len(l))]
                        b = l[np.random.randint(0, len(l))]
                exp['cost_vec']=[a,b]

                # exp['cost_vec'] = [np.random.rand(1)[0]+1,np.random.rand(1)[0]+1]

                exp_list.append(exp)
        return exp_list
    def run(self):
        e_list = self.create_experiments()
        return self.run_experiment(e_list)
    def run_experiment(self,exp_list):
        p = multiprocessing.Pool()
        for e in exp_list:
            nn = self.log_dir + '/' + e['exp_name'] + '_' + str(random.random()) + '.txt'
            self.log_files.append(nn)
            e['nn'] = nn

        result = (p.map(self.csdnn_helper,exp_list))
        p.close()
        # result = [self.csdnn_helper(ps) for ps in exp_list]
        eval_result = []
        predict_probs_train = []
        predict_probs_test = []
        #y_pred_positive =[]
       # (train_result, test_result, p_dict),y_pred_score_test
        for r in result:
            eval_result.append(r[0])
            predict_probs_test.append(r[1])
            #y_pred_positive.append(r[1][:,1])

        sum_probabilities = np.sum(predict_probs_test, axis=0)
        y_pred = np.argmax(sum_probabilities, axis=1)
       # y_pred_max_on_positive = np.max(y_pred_positive,axis=0)


        test_path  = exp_list[0]['test_path']
        _,y = load_data_from_path(test_path)
        y = np.argmax(y.eval(), axis=1)
        TPR, TNR = get_tpr_tnr(y, y_pred)
        best_confusion_matrix = confusion_matrix(y, y_pred, labels=[0, 1])
        #AUC_of_max_prob = metrics.roc_auc_score(y, y_pred_max_on_positive)
        combine_log(path=self.log_dir + '/combine_logs.txt',file_list=self.log_files)
        return eval_result,[TPR,TNR],best_confusion_matrix

    def csdnn_helper(self,p_dict):
        # (train_result, test_result,model_name,y_pred_score_test)
        logger = get_logger(name=p_dict['nn'], file_name=p_dict['nn'], verbose=True)
        train_result, test_result,model_path, y_pred_score_test = csdnn_classifier(log_dir=self.log_dir,logger=logger,p_dict=p_dict)
        return (train_result, test_result, p_dict),y_pred_score_test

    def summary_ensemble(self):
        if self.predict_probs != None:
            sum_probabilities = np.sum(self.predict_probs,axis=0)
            y_predict = np.argmax(sum_probabilities,axis=1)
            return y_predict,self.eval_result


