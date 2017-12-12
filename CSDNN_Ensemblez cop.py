from __future__ import print_function
import timeit
import _pickle as cPickle
import multiprocessing
import numpy
import numpy as np
import sys
import os
import itertools
print(np.__version__)
import shutil

from dllib import settings
from dllib.SdA_Ensemble import SdA_Ensemble
from dllib.classifier import csdnn_classifier
from dllib.ensemble import Ensemble

sys.path.append('dllib')
# for p in sys.path:
#     print(p)
from sklearn.metrics import confusion_matrix

from dllib.SdA import SdA
from dllib.util import load_data, load_data_unblanced, load_data_with_valid, get_tpr_tnr, load_data_without_valid, \
    load_data_lift, random_undersampling, extract_feature, load_model, summary_result, prepare_data_ensemble_learning, \
    summary_fold, combine_log, save_file, load_data_from_path, print_param
import matplotlib.pyplot as plt
from sklearn import metrics
from dllib.log import get_logger
from dllib.util_lift import evaluate_decile, get_cost_vector
import random
import time
# theano.config.compute_test_value = 'warn'
import theano

#theano.config.exception_verbosity = 'high'
data_dir = 'data/taiwan'

def test_SdA(p_dict,ens_param, logger):
    train_result = []
    test_result = []
    vote_result = []
    num_of_fold = 10
    num_subset = p_dict['num_subset']
    p_dict_backup = dict(p_dict)

    for mm in range(num_of_fold):
        p_dict['train_path'] = data_dir + '/train' + str(mm + 1) + '.csv'
        p_dict['valid_path'] = data_dir + '/train' + str(mm + 1) + '.csv'
        p_dict['test_path'] = data_dir + '/test' + str(mm + 1) + '.csv'
        # check if same model existed
        main_model_path = False#exist_model(p_dict_backup)
        if (main_model_path == False):

            logger.info("Trail K=%d" % (mm + 1))
            logger.info('Load data from %s', data_dir)
            # Train main model
            main_train_result, main_test_result, main_model_path, _ = csdnn_classifier(log_dir=log_dir, p_dict=p_dict,
                                                                                      logger=logger)
            save_file('log/'+data_dir+'/last_p',p_dict_backup)
            save_file('log/' + data_dir + '/last_model', load_model(main_model_path))
        else:
            main_train_result,main_test_result = compute_result_from_pretrain_model(main_model_path,p_dict['train_path'],p_dict['test_path'])


        # Prepare for ensemble learning
        data_list = prepare_data_ensemble_learning(train_path=p_dict['train_path'],test_path=p_dict['test_path'],model_path=main_model_path,num_subset=num_subset)
        ens = Ensemble(data_list, ens_param, 'csdnn')
        ens_result,vote,conf_matrix = ens.run()
        train_result.append(np.mean([e[0][0] for e in ens_result],0))
        vote.append(np.mean([e[1][0][2] for e in ens_result],0))
        vote_result.append(vote)

        # Combine log file from weak learner to current logger
        with open(logger.name,'a') as outfile:
            with open(log_dir + '/combine_logs.txt') as infile:
                for line in infile:
                    outfile.write(line)

        print_param(logger,p_dict)
        logger.info('*****Fold %d result*****' % (mm+1))
        logger.info('Training set\nTPR=%s  TNR=%s  AUC=%s' % tuple(main_train_result[0]))
        logger.info('Test set\nTPR=%s  TNR=%s  AUC=%s' % tuple(main_test_result[0]))
        logger.info('Vote result\nTPR=%s  TNR=%s  AUC=%s' %(tuple(vote)))
        logger.info(conf_matrix)
        summary_result(ens_result,logger)

    # Summary fold
    # Average result of vote result
    print('vote')
    print_param(logger,p_dict,'Main Model: ')
    summary_fold(np.hstack([train_result,vote_result]),ens_param,logger)
    # Save only results to other main summary file, for convenience

    print_param(main_summary_logger,p_dict,'Main Model: ')
    summary_fold(np.hstack([train_result,vote_result]),ens_param,main_summary_logger,print_header=False)
    return train_result, test_result

def run_helper(log_dir, p_dict,ens_param=None):  # contain dict
    # If main model existed

    nn = log_dir + '/' +p_dict['exp_name']+'_'+str(random.random()) + '.txt'
    main_file_list.append(nn)
    logger = get_logger(name=nn, file_name= nn, verbose=True)
    train_result, test_result = test_SdA(p_dict=p_dict, logger=logger,ens_param=ens_param)
    return train_result, test_result, p_dict

    # train_result, test_result, test_lift = test_SdA(finetune_lr=p_dict['finetune_lr'], pretraining_epochs=p_dict['pretraining_epochs'],
    #          pretrain_lr=p_dict['pretrain_lr'], training_epochs=p_dict['training_epochs'],
    #          batch_size=p_dict['batch_size'], h=p_dict['h'], cl=p_dict['cl'], cost_vec=p_dict['cost_vec'],beta=p_dict['beta'],logger=logger)
    # return train_result, test_result, test_lift, p_dict


def run_experiment():
    experiment_list = []
    ens_param_list = []


#================


    # e = {'h': [100,100], 'training_epochs': 100, 'pretraining_epochs': 300, 'finetune_lr': 0.01, 'beta': 1,
    #      'cl': [0.1, 0.1, 0.1, .1], 'cost_vec': [1, 3], 'batch_size': 1, 'pretrain_lr': 0.001,
    #      'exp_name': 'main_model',
    #      'pretrain_batchsize': 20, 'reg_coef': None, 'num_subset': 5, 'drop_ps': [.5, .5, .5, .5]}
    #
    #
    # experiment_list.append(dict(e))
    # ens_param_list.append(
    #     {'h': [300], 'training_epochs': 200, 'pretraining_epochs': 50, 'finetune_lr': 0.001, 'beta': 1,
    #      'cl': [0.1, 0.1, 0.1, .1], 'cost_vec': [1, 1], 'batch_size': 20, 'pretrain_lr': 0.001,
    #      'pretrain_batchsize': 10, 'reg_coef': None, 'drop_ps': [.5, .5, .5, .5], 'exp_name': None})

    e = {'h': [150,150], 'training_epochs': 100, 'pretraining_epochs': 300, 'finetune_lr': 0.01, 'beta': 1,
         'cl': [0.1, 0.1, 0.1, .1], 'cost_vec': [1, 2], 'batch_size': 1, 'pretrain_lr': 0.001,
         'exp_name': 'main_model',
         'pretrain_batchsize': 20, 'reg_coef': None, 'num_subset': 5, 'drop_ps': [.5, .5, .5, .5]}


    experiment_list.append(dict(e))
    ens_param_list.append(
        {'h': [150], 'training_epochs': 200, 'pretraining_epochs': 50, 'finetune_lr': 0.005, 'beta': 1,
         'cl': [0.1, 0.1, 0.1, .1], 'cost_vec': [1, 1], 'batch_size': 20, 'pretrain_lr': 0.001,
         'pretrain_batchsize': 10, 'reg_coef': None, 'drop_ps': [.5, .5, .5, .5], 'exp_name': None})

    result = [run_helper(log_dir,ex,ens_param) for ex,ens_param in zip(experiment_list,ens_param_list)]


    # logger = get_logger(name='experiment_settings', file_name=log_dir + '/experiment_setting.txt', verbose=True)
    # for e in experiment_list:
    #     logger.info('experiment_list.append(%s)', e)
    # result = [run_helper(log_dir, ex,ens_param) for ex in experiment_list]




def compute_result_from_pretrain_model(main_model_path,train_path,test_path):
    train_result = []
    test_result = []
    model = load_model(main_model_path)
    train_set_x, train_set_y = load_data_from_path(train_path)
    test_set_x, test_set_y = load_data_from_path(test_path)
    ff_test = model.test_model([train_set_x, train_set_y])
    [y_pred, y_pred_score_train] = ff_test()
    y = np.argmax(train_set_y.eval(), axis=1)
    TPR, TNR = get_tpr_tnr(y, y_pred)
    AUC = metrics.roc_auc_score(y, y_pred_score_train[:, 1])
    train_result.append([TPR, TNR, AUC])

    ff_test = model.test_model([test_set_x, test_set_y])
    [y_pred, y_pred_score_train] = ff_test()
    y = np.argmax(test_set_y.eval(), axis=1)
    TPR, TNR = get_tpr_tnr(y, y_pred)
    AUC = metrics.roc_auc_score(y, y_pred_score_train[:, 1])
    test_result.append([TPR, TNR, AUC])
    return train_result,test_result
def exist_model(p_dict):

    try:
        path = 'log/'+data_dir+'/last_p'
        f = open(path, 'rb')
        d = cPickle.load(f)
        if(p_dict==d):
            print('Load last model')
            return 'log/'+data_dir+'/last_model'
    except Exception:
        print('No last model')
        return False
    return False
if __name__ == '__main__':
    main_file_list = []
    datetim = time.strftime("%d_%m_%Y_%H_%M_%S")
    log_dir = 'log/' + data_dir + '/' + datetim
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_dir + '/tmp_model'):
        os.makedirs(log_dir + '/tmp_model')
    if not os.path.exists(log_dir + '/tmp_data'):
        os.makedirs(log_dir + '/tmp_data')
    settings.init(log_dir)
    start_time = timeit.default_timer()
    nn = log_dir + '/summary_result.txt'
    main_summary_logger = get_logger(name=nn, file_name=nn, verbose=False)
    # run_experiment_parallel()
    run_experiment()

    end_time = (timeit.default_timer() - start_time) / 60
    # logger.info('Time took:%f mn', end_time)
    logger = get_logger(name='summary_result', file_name=log_dir + '/summary_result.txt', verbose=True)
    logger.info('Time took:%f mn', end_time)


    time.sleep(10)
    main_file_list.append(log_dir+'/summary_result.txt')
    combine_log(path=log_dir + '/' + datetim + '.txt', file_list=main_file_list)

