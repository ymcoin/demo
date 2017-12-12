datasets = load_data_without_valid(mm + 1, data_dir)

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # profit_train_old, profit_test = datasets[3]
        # profit_train = get_cost_vector(profit_train_old, beta, cost_vec)
        # datasets[3] = (profit_train, profit_test)
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_pretrain_batches = int(n_train_batches / pretrain_batchsize)
        n_train_batches //= batch_size
        p = None
        try:
            f = open('log/last_model', 'rb')
            [model, d] = cPickle.load(f)
            p = dict(d);del p['ens_param']
            t = dict(p_dict); del t['ens_param']
        except Exception:
            print('No last model')
        if (p != t):
            # numpy random generator
            # start-snippet-3
            numpy_rng = numpy.random.RandomState(89677)
            logger.info('... building the model')
            # construct the stacked denoising autoencoder class
            sda = SdA_Ensemble(reg_coef=reg_coef,
                               numpy_rng=numpy_rng,
                               n_ins=train_set_x.eval().shape[1],
                               hidden_layers_sizes=hidden_l_size,
                               n_outs=2,
                               costVec=cost_vec

                               )
            #    corruption_levels = [.1, .2]
            corruption_levels = cl

            #########################
            # PRETRAINING THE MODEL #
            #########################
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

            # improvement_threshold = 1.005
            # considered significant
            validation_frequency = min(n_train_batches, patience // 2)
            print('patience:%s, validation_freq:%s' % (patience, validation_frequency))
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
                    #   sub = random_undersampling(train_set_x.eval(), train_set_y.eval(), 3, False)
                    #  l = extract_feature(sda.get_feature_function(),sub)
                    # logger.info ("AVG_COST:%f",(minibatch_avg_cost))
                    iter = (epoch - 1) * n_train_batches + minibatch_index
                    # print('iter:%s' % iter)
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
                        # logger.info('TPR=%s'% TPR)
                        # temp[0] = TPR
                        # logger.info('TNR=%s'% TNR)
                        # temp[1] = TNR
                        # response_lift, profit_lift = evaluate_decile(prob=y_pred_score, label=y,
                        #                                           actual_profit=profit_train_old, isPlot=True)
                        # this_validation_loss = profit_lift[0]
                        this_validation_loss = numpy.abs((TPR - TNR))
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
                            # Save model
                            logger.info("Save best model by pickle")
                            model_name = log_dir + '/tmp_model/model_FOLD_' + str(mm + 1)
                            save_file = open(model_name, 'wb')
                            cPickle.dump(sda, save_file)
                            save_file.close()
                            save_file = open('log/last_model', 'wb');
                            cPickle.dump([sda, p_dict], save_file);
                            save_file.close()
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

                            # response_lift, profit_lift = evaluate_decile(prob=y_pred_score, label=y,
                            #                                              actual_profit=profit_test, isPlot=True)
                            # logger.info('response_lift:%s', response_lift)
                            # logger.info('profit lift:%s', profit_lift)
                            # best_response_lift = (response_lift)
                            # best_profit_lift = (profit_lift)
                            # logger.info(classification_report(y_true, y_pred, target_names=['no','yes'],labels=[0,1]))
                            best_confusion_matrix = confusion_matrix(y, y_pred, labels=[0, 1])
                            logger.info(best_confusion_matrix)

                            # best_response_lift = []
                            # best_profit_lift = []
                            TPR, TNR = get_tpr_tnr(y, y_pred)

                            AUC = metrics.roc_auc_score(y, y_pred_score[:, 1])

                            logger.info('AUC=%s', AUC)
                            logger.info('TPR=%s', TPR)
                            logger.info('TNR=%s', TNR)
                        if patience <= iter:
                            done_looping = True
                            break
                            # end fold
            # load best model
            model = load_model(model_name)



        logger.info("========Test with training set=========")
        ff_test = model.test_model([train_set_x, train_set_y])
        [y_pred, y_pred_score] = ff_test()
        y = np.argmax(train_set_y.eval(), axis=1)
        TPR, TNR = get_tpr_tnr(y, y_pred)
        AUC = metrics.roc_auc_score(y, y_pred_score[:, 1])
        logger.info('AUC=%s', AUC)
        logger.info('TPR=%s', TPR)
        logger.info('TNR=%s', TNR)
        train_result.append([TPR, TNR, AUC])
        # evaluate
        logger.info("========Test with test set=========")
        ff_test = model.test_model([test_set_x, test_set_y])
        [y_pred, y_pred_score] = ff_test()
        y = np.argmax(test_set_y.eval(), axis=1)
        TPR, TNR = get_tpr_tnr(y, y_pred)
        AUC = metrics.roc_auc_score(y, y_pred_score[:, 1])
        logger.info('AUC=%s', AUC)
        logger.info('TPR=%s', TPR)
        logger.info('TNR=%s', TNR)
        test_result.append([TPR, TNR, AUC])