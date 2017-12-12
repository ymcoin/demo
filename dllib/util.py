import itertools
import random

import theano
import theano.tensor as T
from numpy import genfromtxt
import numpy as np
from sklearn.metrics import confusion_matrix
#from log import logger
import _pickle as cPickle

from dllib import settings
from dllib.log import get_logger


def load_data(i):
    path = 'data/even/train_bin_evenly' + str(i) + '.csv'
    xx = genfromtxt(path, delimiter=',', dtype=np.float32)
    class_index = xx.shape[1] - 1
    yy = onehot(np.array(xx[:, class_index], dtype=np.int))
    xx = np.delete(xx, class_index, 1)

    path = 'data/even/test_bin_minmax' + str(i) + '.csv'
    xx_test = genfromtxt(path, delimiter=',', dtype=np.float32)
    yy_test = onehot(np.array(xx_test[:, class_index], dtype=np.int32))
    xx_test = np.delete(xx_test, class_index, 1)

    train_set = [xx, yy]

    # cols = [46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 58]
    # [xx_test[:, cols], mean, std] = zero_centered_norm(xx_test[:, cols],mean,std)
    # [xx_test,_,__] = rangeNormalize(xx_test,r,b)
    test_set = [xx_test, yy_test]

    # np.savetxt('ntest1.csv',np.c_[xx_test,yy_test],delimiter=',')
    # test_set = [xx[28861:], yy[28861:]]
    # return [xx,yy]

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when dllib is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_data_with_valid(i, dir):
    path = dir + '/train' + str(i) + '.csv'
    print("Load%s" % path)
    xx = genfromtxt(path, delimiter=',', dtype=np.float32)
    class_index = xx.shape[1] - 1
    yy = onehot(np.array(xx[:, class_index], dtype=np.int))
    xx = np.delete(xx, class_index, 1)

    path = dir + '/test' + str(i) + '.csv'
    xx_test = genfromtxt(path, delimiter=',', dtype=np.float32)
    yy_test = onehot(np.array(xx_test[:, class_index], dtype=np.int32))
    xx_test = np.delete(xx_test, class_index, 1)

    path = dir + '/valid' + str(i) + '.csv'
    xx_valid = genfromtxt(path, delimiter=',', dtype=np.float32)
    yy_valid = onehot(np.array(xx_valid[:, class_index], dtype=np.int32))
    xx_valid = np.delete(xx_valid, class_index, 1)

    train_set = [xx, yy]
    test_set = [xx_test, yy_test]
    valid_set = [xx_valid, yy_valid]
    #valid_set = test_set
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when dllib is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
def load_data_without_valid(i, dir):
    path = dir + '/train' + str(i) + '.csv'
    print("Load%s" % path)
    xx = genfromtxt(path, delimiter=',', dtype=np.float32)
    class_index = xx.shape[1] - 1
    yy = onehot(np.array(xx[:, class_index], dtype=np.int))
    xx = np.delete(xx, class_index, 1)

    path = dir + '/test' + str(i) + '.csv'
    xx_test = genfromtxt(path, delimiter=',', dtype=np.float32)
    yy_test = onehot(np.array(xx_test[:, class_index], dtype=np.int32))
    xx_test = np.delete(xx_test, class_index, 1)


    xx_valid = xx_test
    yy_valid = yy_test


    train_set = [xx, yy]
    test_set = [xx_test, yy_test]
    valid_set = [xx_valid, yy_valid]
    #valid_set = test_set
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when dllib is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
def load_data_from_path(data_path):
    path = data_path
    ext = path.split('.')[-1]
    if ext == 'cpk':
        fil = open(data_path, 'rb')
        xx = cPickle.load(fil)
        fil.close()
    else:
        xx = genfromtxt(path, delimiter=',', dtype=np.float32)
    class_index = xx.shape[1] - 1
    yy = onehot(np.array(xx[:, class_index], dtype=np.int))
    xx = np.delete(xx, class_index, 1)


    data_set = [xx, yy]

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when dllib is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    set_x, set_y = shared_dataset(data_set)

    rval = (set_x, set_y)
    return rval
def load_data_lift(i, dir):
    path = dir + '/train' + str(i) + '.csv'
    #logger.info("Load from %s" % path)
    xx = genfromtxt(path, delimiter=',', dtype=np.float32)
    class_index = xx.shape[1] - 1
    yy = onehot(np.array(xx[:, class_index], dtype=np.int))
    cost_train = np.array(xx[:, class_index - 1], dtype=np.float32)
    xx = np.delete(xx, class_index, 1)
    xx = np.delete(xx, class_index - 1, 1)
    path = dir + '/test' + str(i) + '.csv'
    xx_test = genfromtxt(path, delimiter=',', dtype=np.float32)
    yy_test = onehot(np.array(xx_test[:, class_index], dtype=np.int32))
    cost_test = np.array(xx_test[:, class_index - 1], dtype=np.float32)
    xx_test = np.delete(xx_test, class_index, 1)
    xx_test = np.delete(xx_test, class_index - 1, 1)




    train_set = [xx, yy]
    test_set = [xx_test, yy_test]
    valid_set = train_set
    cost_list = [cost_train, cost_test]
    #valid_set = test_set
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when dllib is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    #profit_train = theano.shared(np.asarray(cost_train,dtype=theano.config.floatX),borrow=True)
    #profit_test = theano.shared(np.asarray(cost_test, dtype=theano.config.floatX), borrow=True)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y),(cost_train,cost_test)]
    return rval

def load_data_unblanced(i):
    path = 'data/original_ccsf_even/train_even' + str(i) + '.csv'
    xx = genfromtxt(path, delimiter=',', dtype=np.float32)
    class_index = xx.shape[1] - 1
    yy = onehot(np.array(xx[:, class_index], dtype=np.int))

    xx = np.delete(xx, class_index, 1)


    path = 'data/even/test_bin_minmax' + str(i) + '.csv'
    xx_test = genfromtxt(path, delimiter=',', dtype=np.float32)
    yy_test = onehot(np.array(xx_test[:, class_index], dtype=np.int32))

    xx_test = np.delete(xx_test, class_index, 1)


    train_set = [xx, yy]

    test_set = [xx_test, yy_test]


    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when dllib is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def onehot(labels):
    temp = np.zeros([labels.shape[0], 2], dtype=np.int)
    i = 0
    for l in labels:
        temp[i][l] = 1
        i = i + 1
    return temp

def get_tpr_tnr(y,y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    if (tp + fn) != 0:
        TPR = float(tp) / (tp + fn)
    if (tn + fp) != 0:
        TNR = float(tn) / (tn + fp)

    return (TPR,TNR)
def prepare_data_ensemble_learning(train_path,test_path,model_path,num_subset,valid_path=None):
    model = load_model(model_path)
    train_set_x, train_set_y = load_data_from_path(train_path)
    test_set_x, test_set_y = load_data_from_path(test_path)

    sub = random_undersampling(train_set_x.eval(), train_set_y.eval(),num_subset, False)
    l = extract_feature(model.get_feature_function(), sub, 'train')
    yy = np.argmax(test_set_y.eval(), axis=1);
    yy = np.reshape(yy, (len(yy), 1))
    test_feature = extract_feature(model.get_feature_function(), [np.hstack([test_set_x.eval(), yy])], 'test')
    combine = list(zip(l, test_feature))
    data_list = []
    for e in combine:
        data_list += list(itertools.product(e[0], e[1]))
    return  data_list
class Log(object):
    def __init__(self):
        self.content = ''

    def p(self, s):
        print(s)
        if (isinstance(s, (str, unicode)) == False):
            s = str(s)
        self.content = self.content + s + '\n'

    def save(self, path):
        f = open(path, 'wb')
        f.write(self.content)
        f.close()
        print('log saved')
def random_undersampling(X,Y,num_subset,replacement=False):
    list_subset = []
    if Y.shape[1]>1:
        Y=np.argmax(Y,axis=1)
    # Y = np.array(Y,dtype=np.int32)
    num_pos = np.sum(Y)
    num_neg = Y.shape[0] - num_pos
    neg_index = np.where(Y==0)[0]
    pos_index = np.where(Y==1)[0]

    data = np.hstack([X,np.reshape(Y,(len(Y),1)).astype(np.float32)])
    neg_data = data[neg_index]
    pos_data = data[pos_index]
    # np.random.seed(0)
    if(replacement==False):
        for i in range(num_subset):
            # if(random.choice([True,False])):
            #     num_sample = int(num_pos-num_pos/4)
            # else:
            #     num_sample = int(num_pos+num_pos/4)
            num_sample = int(num_pos)
            index = np.random.randint(0,len(neg_index),num_sample)
            subset = np.vstack([pos_data,neg_data[index]])
            np.random.shuffle(subset)
            list_subset.append(subset)

    return list_subset
def extract_feature(fn_list, list_subset,datype):
    list_feature_subset = []
    for i,f in enumerate(fn_list):
        tmp = []
        for j,subset in enumerate(list_subset):
            y = subset[:,-1]
            s = np.hstack([f(subset[:,0:-1]),np.reshape(y,(len(y),1))])
            name = settings.log_dir + '/tmp_data/'+datype+'layer_' + str(i + 1) +'_subset_'+str(j+1)+'.cpk'
            save_file = open(name, 'wb')
            cPickle.dump(s, save_file)
            save_file.close()

            # filename = name+'.csv'
            # #form = '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%d,%f,%f,%f,%d,%f,%d'
            # np.savetxt(filename, s, delimiter=',')
            tmp.append(name)
        list_feature_subset.append(tmp)
    return list_feature_subset
def load_model(model_path):
    fil = open(model_path, 'rb')
    model = cPickle.load(fil)
    fil.close()
    return model
def save_file(des_path,file):
    fil = open(des_path, 'wb')
    model = cPickle.dump(file,fil)
    fil.close()
    return des_path
def summary_fold(all_metric,ens_param,logger,print_header=True):

    # logger = get_logger(name='summary_result   ', file_name=settings.log_dir + '/summary_result.txt', verbose=True)
    if print_header:
        logger.info('\tTraining set\t\t\t\t\t\t\tTest set\t\t\t\t\t\t')
        logger.info('TPR\tTNR\tAUC\tTPR\tTNR\tAUC')
    if (all_metric.shape[0] > 1):
        all_metric = np.vstack([all_metric, np.mean(all_metric, axis=0), np.std(all_metric, axis=0)])
    # logger.info(
    #     'Sub Model: Layer:%s, CostVec:%s, fineLR:%s, preLR:%s, fineBatch:%s, preBatch:%s, cl:%s, beta:%s'
    #     % (ens_param['h'], ens_param['cost_vec'], ens_param['finetune_lr'], ens_param['pretrain_lr'],
    #        ens_param['batch_size'],
    #        ens_param['pretrain_batchsize'], ens_param['cl'], ens_param['beta']))
    print_param(logger,ens_param,'Sub Model: ')
    for x in all_metric:
        s = ''
        for y in x:
            s = s + str(y) + '\t'
        logger.info('%s' % s)

def summary_result(result,logger):

    # logger = get_logger(name='summary_result   ', file_name=settings.log_dir + '/summary_result.txt', verbose=True)
    logger.info('\tTraining set\t\t\t\t\t\t\tTest set\t\t\t\t\t\t')
    # logger.info('TPR\tTNR\tAUC\tTPR\tTNR\tAUC\tR1\tR2\tR3\tR4\tR5\tP1\tP2\tP3\tP4\tP5')
    logger.info('TPR\tTNR\tAUC\tTPR\tTNR\tAUC')
    for train_result, test_result, param in result:

        # response_matrix = np.array(lift_result[0])[:, 0:5]  # take only first 5 decile
        # profit_matrix = np.array(lift_result[1])[:, 0:5]  # take only first 5 decile
        # # train_result = np.array()
        # all_metric = np.hstack([train_result, test_result, response_matrix, profit_matrix])
        all_metric = np.hstack([train_result, test_result])
        if(all_metric.shape[0]>1):
            all_metric = np.vstack([all_metric, np.mean(all_metric, axis=0), np.std(all_metric, axis=0)])
        # param = experiment_list[i]
        # logger.info(param)
        print_param(logger,param)
        # logger.info('Layer:%s, CostVec:%s, fineLR:%s, preLR:%s, fineBatch:%s, preBatch:%s, cl:%s, beta:%s, Name:%s'
        #             %(param['h'],param['cost_vec'],param['finetune_lr'],param['pretrain_lr'],param['batch_size'],param['pretrain_batchsize'],param['cl'],param['beta'],param['exp_name']))
        for x in all_metric:
            s = ''
            for y in x:
                s = s + str(y) + '\t'
            logger.info('%s' % s)

# def summary_result_to_file(result,file_name):
#     name = file_name.split('.')[0]
#     logger = get_logger(name=file_name, file_name=settings.log_dir + '/'+file_name, verbose=True)
#     logger.info('\tTraining set\t\t\t\t\t\t\tTest set\t\t\t\t\t\t')
#     # logger.info('TPR\tTNR\tAUC\tTPR\tTNR\tAUC\tR1\tR2\tR3\tR4\tR5\tP1\tP2\tP3\tP4\tP5')
#     logger.info('TPR\tTNR\tAUC\tTPR\tTNR\tAUC')
#     for train_result, test_result, param in result:
#
#         # response_matrix = np.array(lift_result[0])[:, 0:5]  # take only first 5 decile
#         # profit_matrix = np.array(lift_result[1])[:, 0:5]  # take only first 5 decile
#         # # train_result = np.array()
#         # all_metric = np.hstack([train_result, test_result, response_matrix, profit_matrix])
#         all_metric = np.hstack([train_result, test_result])
#         if(all_metric.shape[0]>1):
#             all_metric = np.vstack([all_metric, np.mean(all_metric, axis=0), np.std(all_metric, axis=0)])
#         # param = experiment_list[i]
#         # logger.info(param)
#         logger.info('Layer:%s, CostVec:%s, fineLR:%s, preLR:%s, fineBatch:%s, preBatch:%s, cl:%s, beta:%s, Name:%s'
#                     %(param['h'],param['cost_vec'],param['finetune_lr'],param['pretrain_lr'],param['batch_size'],param['pretrain_batchsize'],param['cl'],param['beta'],param['exp_name']))
#         for x in all_metric:
#             s = ''
#             for y in x:
#                 s = s + str(y) + '\t'
#             logger.info('%s' % s)


def combine_log(path, file_list):
    with open(path, 'w') as outfile:
        for fname in file_list:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
def print_param(logger,p_dict,title=None):
    if(title==None):
        title = ''
    logger.info(
        title+'Layer:%s, CostVec:%s, fineLR:%s, preLR:%s, fineBatch:%s, \npreBatch:%s, pretrain_epoch:%s, finetune_epoch:%s, dropout:%s,reg_coef:%s cl:%s, beta:%s, Name:%s'
        % (p_dict['h'], p_dict['cost_vec'], p_dict['finetune_lr'], p_dict['pretrain_lr'], p_dict['batch_size'],
           p_dict['pretrain_batchsize'],p_dict['pretraining_epochs'],p_dict['training_epochs'],p_dict['drop_ps'],p_dict['reg_coef'] ,p_dict['cl'], p_dict['beta'], p_dict['exp_name']))

