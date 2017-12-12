from __future__ import print_function
import timeit

import multiprocessing
import numpy
import numpy as np
import sys
import os

sys.path.append('dllib')
for p in sys.path:
    print(p)
from sklearn.metrics import confusion_matrix


from dllib.SdA import SdA
from dllib.util import load_data, load_data_unblanced, load_data_with_valid, get_tpr_tnr, load_data_without_valid, \
    load_data_lift
import matplotlib.pyplot as plt
from sklearn import metrics
from dllib.log import get_logger
from dllib.util_lift import evaluate_decile, get_cost_vector
import random
import time
# theano.config.compute_test_value = 'warn'
train_all = []
test_all = []
test_lift_all = []
auc_all = []
data_dir = 'data/DM'

def test_SdA(finetune_lr=0.1, pretraining_epochs=1,
             pretrain_lr=0.001, training_epochs=1,
             batch_size=10, h=[10, 10], cl=[.1, .2], cost_vec=[1, 1.2], beta=30,logger=None,pretrain_batchsize=10):


    logger.info(
        'pre-epoch:%d\ntrain_epoch:%d\npre_lr:%lf\nfine_lr:%lf\nhidden_layer:%s\nCoruption level:%s\nCost_vec:%s\nBeta:%s' % (
            pretraining_epochs, training_epochs, pretrain_lr, finetune_lr, h, cl, cost_vec, beta))
    cost_vec = numpy.array(cost_vec, dtype="float32")
    train_result = []
    test_result = []
    auc_list = []
    test_lift = [[],[]]
    hidden_l_size = h
    num_of_fold = 10
    for mm in range(3):
    #for mm in range(5,6):
        logger.info("Trail K=%d" % (mm + 1))
        logger.info('Load data from %s',data_dir)
        datasets = load_data_lift(mm + 1, data_dir)

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        profit_train_old, profit_test = datasets[3]
        profit_train = get_cost_vector(profit_train_old, beta, cost_vec)
        datasets[3] = (profit_train, profit_test)
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_pretrain_batches = int(n_train_batches / pretrain_batchsize)
        n_train_batches //= batch_size

        # numpy random generator
        # start-snippet-3
        numpy_rng = numpy.random.RandomState(89677)
        logger.info('... building the model')
        # construct the stacked denoising autoencoder class
        sda = SdA(
            numpy_rng=numpy_rng,
            n_ins=train_set_x.eval().shape[1],
            hidden_layers_sizes=hidden_l_size,
            n_outs=2,
            costVec=cost_vec,

        )
        #    corruption_levels = [.1, .2]
        corruption_levels = cl

        #########################
        # PRETRAINING THE MODEL #
        #########################
        logger.info('... getting the pretraining functions')
        pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size)

        logger.info('... pre-training the model')

        ## Pre-train layer-wise

        for i in range(sda.n_layers):
            # go through pretraining epochs
            for epoch in range(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in range(n_pretrain_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                                                corruption=corruption_levels[i],
                                                lr=pretrain_lr))
                logger.info('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c, dtype='float64')))



        # logger.info(('The pretraining dllib for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
        # end-snippet-4
        ########################
        # FINETUNING THE MODEL #
        ########################

        # get the training, validation and testing function for the model

        logger.info('... getting the finetuning functions')
        train_fn, validate_model, test_model = sda.build_finetune_functions(
            datasets=datasets,
            batch_size=batch_size,
            learning_rate=finetune_lr,

        )

        logger.info('... finetunning the model')
        # early-stopping parameters
        patience = 10 * n_train_batches  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
        # found
        improvement_threshold = 0.995  # a relative improvement of this much is


        #improvement_threshold = 1.005
        # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
        print('patience:%s, validation_freq:%s' %(patience,validation_frequency))
        # go through this many
        # minibatche before checking the network
        # on the validation set; in this case we
        # check every epoch

        best_validation_loss = numpy.inf
        test_score = 0.


        done_looping = False
        epoch = 0

        while (epoch < training_epochs) and (not done_looping):
            epoch += 1
            for minibatch_index in range(n_train_batches):
                minibatch_avg_cost = train_fn(minibatch_index)
                # logger.info ("AVG_COST:%f",(minibatch_avg_cost))
                iter = (epoch - 1) * n_train_batches + minibatch_index
                #print('iter:%s' % iter)
                if (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model()
                    # this_validation_loss = numpy.mean(validation_losses, dtype='float64')
                    ff_test = sda.test_model(datasets[1])
                    [y_pred, y_pred_score] = ff_test()
                    # get_any = sda.get_any(datasets[0], profit_train)
                    # a,b=get_any()
                    # get_cost = sda.get_cost(datasets[0], profit_train)
                    # d = get_cost()
                    # s = dbn.getSoftmaxresult(datasets[2])
                    # softm = s()
                    y = np.argmax(datasets[1][1].eval(), axis=1)
                    # tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

                    # logger.info(classification_report(y_true, y_pred, target_names=['no','yes'],labels=[0,1]))
                    #  logger.info(confusion_matrix(y, y_pred, labels=[0, 1]))
                    TPR, TNR = get_tpr_tnr(y, y_pred)
                    # temp = [0, 0]
                    logger.info('TPR=', TPR)
                    # temp[0] = TPR
                    logger.info('TNR=', TNR)
                    # temp[1] = TNR
                    #response_lift, profit_lift = evaluate_decile(prob=y_pred_score, label=y,
                      #                                           actual_profit=profit_train_old, isPlot=True)
                    #this_validation_loss = profit_lift[0]
                    this_validation_loss = numpy.abs((TPR-TNR))
                    # this_validation_loss = numpy.mean(validation_losses, dtype='float64')
                    logger.info('epoch %i, minibatch %i/%i, validation partial error %f %%' %
                                (epoch, minibatch_index + 1, n_train_batches,
                                 this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        # improve patience if loss improvement is good enough
                        if (
                                    this_validation_loss < best_validation_loss *
                                    improvement_threshold
                        ):
                            print('increase patience from:%s' % patience)
                            patience = max(patience, iter * patience_increase)
                            print('to:%s' % patience)
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        # test_losses = test_model()
                        # test_score = numpy.mean(test_losses, dtype='float64')
                        # logger.info(('     epoch %i, minibatch %i/%i, test error of '
                        #              'best model %f %%') %
                        #             (epoch, minibatch_index + 1, n_train_batches,
                        #              test_score * 100.))
                        logger.info("========new best with testset=========")
                        ff_test = sda.test_model(datasets[2])
                        [y_pred, y_pred_score] = ff_test()
                        y = np.argmax(datasets[2][1].eval(), axis=1)

                        response_lift, profit_lift = evaluate_decile(prob=y_pred_score, label=y,
                                                                     actual_profit=profit_test, isPlot=True)
                        logger.info('response_lift:%s', response_lift)

                        logger.info('profit lift:%s', profit_lift)
                        # logger.info(classification_report(y_true, y_pred, target_names=['no','yes'],labels=[0,1]))
                        best_confusion_matrix = confusion_matrix(y, y_pred, labels=[0, 1])
                        logger.info(best_confusion_matrix)
                        best_test = [0, 0, 0]
                        # best_response_lift = []
                        # best_profit_lift = []
                        TPR, TNR = get_tpr_tnr(y, y_pred)
                        best_test[0] = TPR
                        best_test[1] = TNR
                        best_test[2] = metrics.roc_auc_score(y, y_pred_score[:, 1])
                        best_response_lift = (response_lift)
                        best_profit_lift = (profit_lift)
                        logger.info('AUC=%s', best_test[2])
                        logger.info('TPR=%s', TPR)
                        logger.info('TNR=%s', TNR)
                        #                        test_result[mm] = best_test

                        # train
                        ff_test = sda.test_model(datasets[0])
                        [y_pred, y_pred_score] = ff_test()
                        y = np.argmax(datasets[0][1].eval(), axis=1)
                        best_train = [0, 0, 0]
                        TPR, TNR = get_tpr_tnr(y, y_pred)
                        best_train[0] = TPR
                        best_train[1] = TNR
                        best_train[2] = metrics.roc_auc_score(y, y_pred_score[:, 1])

                        #                       train_result[mm] = best_train
                if patience <= iter:
                    done_looping = True
                    break


        # logger.info RESULT

        logger.info("========Test with training set=========")
        # ff = sda.test_model(datasets[0])
        # [y_pred, y_pred_score] = ff()
        # y = np.argmax(datasets[0][1].eval(),axis=1)
        # logger.info(confusion_matrix(y, y_pred, labels=[0, 1]))
        # temp = [0, 0]
        # TPR,TNR = get_tpr_tnr(y,y_pred)
        # temp[0] = TPR
        # temp[1] = TNR
        logger.info('AUC=%s', best_train[2])
        logger.info('TPR=%s', best_train[0])
        logger.info('TNR=%s', best_train[1])

        train_result.append(best_train)
        # evaluate
        logger.info("========Test with test set=========")
        # ff_test = sda.test_model(datasets[2])
        # [y_pred, y_pred_score] = ff_test()
        # y = np.argmax(datasets[2][1].eval(),axis=1)
        # auc = metrics.roc_auc_score(y, y_pred_score[:,1])
        # logger.info('AUC=', auc)
        # auc_list.append(auc)

        # ============


        # logger.info(classification_report(y_true, y_pred, target_names=['no','yes'],labels=[0,1]))
        # logger.info(confusion_matrix(y, y_pred, labels=[0, 1]))
        # TPR,TNR = get_tpr_tnr(y,y_pred)
        logger.info('Response_lift=%s', best_response_lift)
        logger.info('Profit_lift=%s', best_profit_lift)
        logger.info('AUC=%s', best_test[2])
        logger.info('TPR=%s', best_test[0])
        logger.info('TNR=%s', best_test[1])

        test_result.append(best_test)
       # test_lift[0] = np.vstack([test_lift[0], best_response_lift])
       # test_lift[1] = np.vstack([test_lift[1], best_profit_lift])
        test_lift[0].append(best_response_lift)
        test_lift[1].append(best_profit_lift)
        logger.info(
            (
                'Optimization complete with best validation score of %f %%, '
                'on iteration %i, '
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
        )
        # logger.info(('The training dllib for file ' +
        #        os.path.split(__file__)[1] +
        #        ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    logger.info(
        'pre-epoch:%d\ntrain_epoch:%d\npre_lr:%lf\nfine_lr:%lf\nhidden_layer:%s\nCoruption level:%s\nCost_vec:%s\nBeta:%s' % (
            pretraining_epochs, training_epochs, pretrain_lr, finetune_lr, hidden_l_size, cl, cost_vec, beta))
    logger.info("====Overall: trainset====")
    for p, n, a in train_result:
        logger.info("%f\t%f\t%f" % (p, n, a))
    mean = np.mean(train_result, axis=0)
    logger.info('Mean=%f\t%f\t%f' % (mean[0], mean[1], mean[2]))
    logger.info("====Overall: testset====")
    temp_response = []
    temp_profit = []
    mean = []
    for p, n, a in test_result:
        logger.info("%f\t%f\t%f" % (p, n, a))

    mean = np.mean(test_result, axis=0)

    logger.info('Mean=%f\t%f\t%f' % (mean[0], mean[1], mean[2]))
    mean_response_lift = np.mean(test_lift[0], axis=0)
    mean_profit_lift = np.mean(test_lift[1], axis=0)
    logger.info('Mean Response lift=%s', mean_response_lift)
    logger.info('Mean Profit lift=%s', mean_profit_lift)
    # logger.info("====Overall: AUC for testset====")
    # for a in auc_list:
    #     logger.info("%f" % a)
    #train_all.append(train_result)
   # test_all.append(test_result)
  #  test_lift_all.append((test_lift[0], test_lift[1]))
    return train_result,test_result,test_lift
    # auc_all.append(auc_list)


def run_helper(log_dir,p_dict):  # contain dict
    nn = str(random.random()) + '.txt'
    logger = get_logger(name=nn, file_name=log_dir + '/' + nn, verbose=True)

    train_result, test_result, test_lift = test_SdA(finetune_lr=p_dict['finetune_lr'], pretraining_epochs=p_dict['pretraining_epochs'],
             pretrain_lr=p_dict['pretrain_lr'], training_epochs=p_dict['training_epochs'],
             batch_size=p_dict['batch_size'], h=p_dict['h'], cl=p_dict['cl'], cost_vec=p_dict['cost_vec'],beta=p_dict['beta'],logger=logger)
    return train_result, test_result, test_lift, p_dict
def run_experiment():
    experiment_list = []
    experiment_list.append(
        {'h': [300, 300], 'training_epochs': 100, 'pretraining_epochs': 20, 'finetune_lr': 0.01, 'beta': 1,
         'cl': [0.1, 0.1], 'cost_vec': [1, 3], 'batch_size': 1, 'pretrain_lr': 0.001})
    result = [run_helper(log_dir,ex) for ex in experiment_list]
    summary_result(result=result)

from functools import  partial
def run_experiment_parallel():
    experiment_list = []
    experiment_list.append(
        {'h': [60, 60], 'training_epochs': 100, 'pretraining_epochs': 20, 'finetune_lr': 0.01, 'beta': 1,
         'cl': [0.1, 0.1], 'cost_vec': [1, 3], 'batch_size': 1, 'pretrain_lr': 0.001})
    experiment_list.append(
        {'h': [100, 100], 'training_epochs': 100, 'pretraining_epochs': 20, 'finetune_lr': 0.01, 'beta': 1,
         'cl': [0.1, 0.1], 'cost_vec': [1, 3], 'batch_size': 1, 'pretrain_lr': 0.001})
    experiment_list.append(
        {'h': [120, 120], 'training_epochs': 100, 'pretraining_epochs': 20, 'finetune_lr': 0.01, 'beta': 1,
         'cl': [0.1, 0.1], 'cost_vec': [1, 3], 'batch_size': 1, 'pretrain_lr': 0.001})
    experiment_list.append(
        {'h': [60], 'training_epochs': 100, 'pretraining_epochs': 20, 'finetune_lr': 0.01, 'beta': 1,
         'cl': [0.1, 0.1], 'cost_vec': [1, 3], 'batch_size': 1, 'pretrain_lr': 0.001})
    experiment_list.append(
        {'h': [80], 'training_epochs': 100, 'pretraining_epochs': 20, 'finetune_lr': 0.01, 'beta': 1,
         'cl': [0.1, 0.1], 'cost_vec': [1, 3], 'batch_size': 1, 'pretrain_lr': 0.001})
    experiment_list.append(
        {'h': [100], 'training_epochs': 100, 'pretraining_epochs': 20, 'finetune_lr': 0.01, 'beta': 1,
         'cl': [0.1, 0.1], 'cost_vec': [1, 3], 'batch_size': 1, 'pretrain_lr': 0.001})
    experiment_list.append(
        {'h': [120], 'training_epochs': 100, 'pretraining_epochs': 20, 'finetune_lr': 0.01, 'beta': 1,
         'cl': [0.1, 0.1], 'cost_vec': [1, 3], 'batch_size': 1, 'pretrain_lr': 0.001})
    experiment_list.append(
        {'h': [130], 'training_epochs': 100, 'pretraining_epochs': 20, 'finetune_lr': 0.01, 'beta': 1,
         'cl': [0.1, 0.1], 'cost_vec': [1, 3], 'batch_size': 1, 'pretrain_lr': 0.001})
    logger = get_logger(name='experiment_settings', file_name=log_dir + '/experiment_setting.txt', verbose=True)
    for e in experiment_list:
        logger.info('experiment_list.append(%s)', e)
    p = multiprocessing.Pool()
    func = partial(run_helper,log_dir)

    result = (p.map(func, experiment_list))
    summary_result(result)

def summary_result(result):

    logger = get_logger(name='summary_result   ', file_name=log_dir + '/summary_result', verbose=True)
    logger.info('\tTraining set\t\t\t\t\t\t\tTest set\t\t\t\t\t\t')
    logger.info('TPR\tTNR\tAUC\tTPR\tTNR\tAUC\tR1\tR2\tR3\tR4\tR5\tP1\tP2\tP3\tP4\tP5')
    for train_result, test_result, lift_result, param in result:

        response_matrix = np.array(lift_result[0])[:, 0:5]  # take only first 5 decile
        profit_matrix = np.array(lift_result[1])[:, 0:5]  # take only first 5 decile
        # train_result = np.array()
        all_metric = np.hstack([train_result, test_result, response_matrix, profit_matrix])
        all_metric = np.vstack([all_metric, np.mean(all_metric, axis=0), np.std(all_metric, axis=0)])
        # param = experiment_list[i]
        logger.info(param)

        for x in all_metric:
            s = ''
            for y in x:
                s = s + str(y) + '\t'
            logger.info('%s' % s)


if __name__ == '__main__':
    datetim = time.strftime("%d_%m_%Y_%H_%M_%S")
    log_dir = 'log/' +data_dir+'/'+ datetim
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    start_time = timeit.default_timer()

    run_experiment_parallel()
    #run_experiment()

    end_time = (timeit.default_timer() - start_time) / 60
    #logger.info('Time took:%f mn', end_time)
    logger = get_logger(name='summary_result', file_name=log_dir + '/summary_result', verbose=True)
    logger.info('Time took:%f mn', end_time)

    # combine test file to summary.txt
    time.sleep(10)
    file_list = [f for f in os.listdir(log_dir)]
    with open(log_dir+'/'+datetim+'.txt', 'w') as outfile:
        for fname in file_list:
            with open(log_dir+'/'+fname) as infile:
                for line in infile:
                    outfile.write(line)
    print (file_list)