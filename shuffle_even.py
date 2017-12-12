import numpy as np


def even(n):
    path = 'data/taiwan_split/train' + str(n) + '.csv'
    data = np.genfromtxt(path, delimiter=',', dtype=np.float32)
    class_index = data.shape[1] - 1
    data[:, class_index] = np.array(data[:, class_index], dtype=np.int)

    class_data = data[:, class_index]

    new_data = np.zeros(data.shape, dtype=data.dtype)
    pos_index = np.where(class_data == 1)
    neg_index = np.where(class_data == 0)
    pos_size = np.shape(pos_index)[1]
    neg_size = np.shape(neg_index)[1]
    step = int(float(neg_size + pos_size) // pos_size)
    print ('possize,negsize ')
    print(pos_size, neg_size)
    print('Step=%d' % step)
    # print(data)

    i = 0
    j = 0
    for d in data[pos_index]:
        new_data[i] = d
        i = i + step
        # if(j<(float(neg_size-pos_size)//(step+))):
        #     i = i+step +1
        # else:
        #     i = i + step
        # j = j +1

    # print(new_data)
    the_rest_index = np.where(new_data[:, class_index] == 0)[0]
    i = 0
    for d in data[neg_index]:
        j = the_rest_index[i]
        new_data[j] = d
        i = i + 1

    new_data = np.flip(new_data, axis=0)
    print(new_data)
    #kfraud
    #form = '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d'
    # taiwan
    form = '%f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d'
    filename = 'data/taiwan_split/even/train' + str(n) + '.csv'
    print (new_data.shape)
    np.savetxt(filename, new_data, delimiter=',', fmt=form)
    print("Save: " + filename)

if __name__ == '__main__':
    for i in range(1,11):
        even(i)


# ==Save ===
# from SAE_CSSF import run_helper
#
# for n in range(6,9):
#     np.random.seed(n)
#     #german
#     form = '%d,%d,%d,%d,%f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d,%f,%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d,%d,%d,%d,%d'   #german
#
#     path = 'data/germ_split_even/train'+str(n)+'.csv'
#     data = np.genfromtxt(path, delimiter=',', dtype=np.float32)
#     class_index = data.shape[1] - 1
#     data[:, class_index] = np.array(data[:, class_index], dtype=np.int)
#     np.random.shuffle(data)
#
#
#     filename ='data/germ_split_even/train'+str(n)+'.csv'
#     np.savetxt(filename, data, delimiter=',',fmt=form)
#     print("Save: " + filename)
#
#     print('dsf')
#     d = {'finetune_lr': 0.005, 'pretraining_epochs': 30, 'pretrain_lr': 0.01,
#                          'training_epochs':427,'batch_size':5,
#                          'h':[200],'cl':[.1,.1],'cost_vec':[1,1.55]}
#     run_helper(d)

    # inetune_lr = 0.005, pretraining_epochs = 30,
    # pretrain_lr = 0.01, training_epochs = 427,
    # batch_size = 5, h = [200], cl = [.1, .1, .1], cost_vec = [1, 1.55])





