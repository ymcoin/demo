from sklearn import metrics

import numpy as np
import _pickle as cPickle

from sklearn.metrics import confusion_matrix

from dllib import settings
from dllib.SdA_Ensemble import SdA_Ensemble
from dllib.util import load_data_without_valid, get_tpr_tnr, load_data_from_path, load_model


def csdnn_classifier(log_dir, logger,p_dict):

    finetune_lr = p_dict['finetune_lr'];
    pretraining_epochs = p_dict['pretraining_epochs'];
    pretrain_lr = p_dict['pretrain_lr']
    training_epochs = p_dict['training_epochs'];
    batch_size = p_dict['batch_size'];
    h = p_dict['h'];
    cl = p_dict['cl']
    cost_vec = p_dict['cost_vec'];
    beta = p_dict['beta'];
    reg_coef = p_dict['reg_coef'];
    pretrain_batchsize = p_dict['pretrain_batchsize']
    drop_ps = p_dict['drop_ps']
    exp_name = p_dict['exp_name']
    train_path = p_dict['train_path']
    test_path = p_dict['test_path']
    valid_path = p_dict['valid_path']
    logger.info(
        'pre-epoch:%d\ntrain_epoch:%d\npre_lr:%lf\nfine_lr:%lf\nhidden_layer:%s\nCoruption level:%s\nCost_vec:%s\nBeta:%s' % (
            pretraining_epochs, training_epochs, pretrain_lr, finetune_lr, h, cl, cost_vec, beta))
    cost_vec = np.array(cost_vec, dtype="float32")
    train_result = []
    test_result = []
    hidden_l_size = h
    logger.info('Load data from %s', train_path)
    train_set_x, train_set_y = load_data_from_path(train_path)
    test_set_x, test_set_y = load_data_from_path(test_path)
    valid_set_x, valid_set_y = load_data_from_path(valid_path)
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_pretrain_batches = int(n_train_batches / pretrain_batchsize)
    n_train_batches //= batch_size

    # numpy random generator
    numpy_rng = np.random.RandomState(89677)
    logger.info('... building the model')
    # construct the stacked denoising autoencoder class
    sda = SdA_Ensemble(
        numpy_rng=numpy_rng,
        n_ins=train_set_x.eval().shape[1],
        hidden_layers_sizes=hidden_l_size,
        n_outs=2,
        costVec=cost_vec,
        reg_coef=reg_coef,
        drop_ps=drop_ps
    )
    corruption_levels = cl

    #########################
    # PRETRAINING THE MODEL #
    #########################
    if pretraining_epochs>0:
        logger.info('... getting the pretraining functions')
        pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=pretrain_batchsize)
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
            logger.info('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c, dtype='float64')))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model

    logger.info('... getting the finetuning functions')
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=[(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)],
        batch_size=batch_size,
        learning_rate=finetune_lr,

    )

    logger.info('... finetunning the model')
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 3  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # improvement_threshold = 1.005
    # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    print('patience:%s, validation_freq:%s' % (patience, validation_frequency))
    best_validation_loss = np.inf
    test_score = 0.

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            # logger.info ("AVG_COST:%f",(minibatch_avg_cost))
            iter = (epoch - 1) * n_train_batches + minibatch_index
            # print('iter:%s' % iter)
            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                ff_test = sda.test_model([valid_set_x, valid_set_y])
                [y_pred, y_pred_score] = ff_test()
                y = np.argmax(valid_set_y.eval(), axis=1)
                TPR, TNR = get_tpr_tnr(y, y_pred)
                # this_validation_loss = np.abs((TPR - TNR))
                this_validation_loss = 1-metrics.roc_auc_score(y, y_pred_score[:, 1])
                # this_validation_loss = np.mean(validation_losses, dtype='float64')
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
                    # Save model
                    logger.info("Save best model by pickle")
                    model_name = log_dir + '/tmp_model/' + exp_name + '_'
                    save_file = open(model_name, 'wb')
                    cPickle.dump(sda, save_file)
                    save_file.close()
                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    logger.info("========new best with testset=========")
                    ff_test = sda.test_model([test_set_x, test_set_y])
                    [y_pred, y_pred_score] = ff_test()
                    y = np.argmax(test_set_y.eval(), axis=1)

                    best_confusion_matrix = confusion_matrix(y, y_pred, labels=[0, 1])
                    logger.info(best_confusion_matrix)
                    TPR, TNR = get_tpr_tnr(y, y_pred)
                    AUC = metrics.roc_auc_score(y, y_pred_score[:, 1])
                    logger.info('AUC=%s', AUC);
                    logger.info('TPR=%s', TPR);
                    logger.info('TNR=%s', TNR);

            if patience <= iter:
                done_looping = True
                break
    # load best model
    model = load_model(model_name)
    logger.info("========Test with training set=========")
    ff_test = model.test_model([train_set_x, train_set_y])
    [y_pred, y_pred_score_train] = ff_test()
    y = np.argmax(train_set_y.eval(), axis=1)
    TPR, TNR = get_tpr_tnr(y, y_pred)
    AUC = metrics.roc_auc_score(y, y_pred_score_train[:, 1])
    logger.info('AUC=%s', AUC);
    logger.info('TPR=%s', TPR);
    logger.info('TNR=%s', TNR);
    train_result.append([TPR, TNR, AUC])

    # evaluate
    logger.info("========Test with test set=========")
    ff_test = model.test_model([test_set_x, test_set_y])
    [y_pred, y_pred_score_test] = ff_test()
    y = np.argmax(test_set_y.eval(), axis=1)
    TPR, TNR = get_tpr_tnr(y, y_pred)
    AUC = metrics.roc_auc_score(y, y_pred_score_test[:, 1])
    logger.info('AUC=%s', AUC);
    logger.info('TPR=%s', TPR);
    logger.info('TNR=%s', TNR);
    test_result.append([TPR, TNR, AUC])

    return train_result, test_result,model_name,y_pred_score_test
