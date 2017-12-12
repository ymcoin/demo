import numpy as np
import theano
#from log import logger
def evaluate_decile(prob,label,actual_profit,isPlot):
    commulative_response_rate = []
    commulative_average_profit = []
    total_example = len(label)
    max_output_index = np.argmax(prob, axis=1)
    prob[:, 0] = 1 - prob[:, 0]

    # get max prob only
    prob = prob[[np.arange(len(prob)), max_output_index]]
    prob_label = np.column_stack((prob, label))
    prob_label = np.column_stack((prob_label,actual_profit))
    #sort decending by probability
    prob_label = prob_label[prob_label[:,0].argsort()[::-1]]
    sorted_response = prob_label[:,1]
    sorted_profit = prob_label[:,2]
    overall_response = np.average(label)
    average_profit = np.average(sorted_profit)
    decile_size = int(np.floor(total_example/10.0))
    for i in range(1,10):
        # get i decile
        decile_response = sorted_response[0:decile_size*i]
        decile_profit = sorted_profit[0:decile_size*i]
        # response rate up to i decile
        decile_response_rate = np.average(decile_response)
        commulative_response_rate.append(decile_response_rate)
        # commulative average profit up to i decile
        decile_average_profit = np.average(decile_profit)
        commulative_average_profit.append(decile_average_profit)

    # remain part
    decile_response = sorted_response
    decile_response_rate = (np.sum(decile_response) / total_example)
    commulative_response_rate.append(decile_response_rate)

    decile_profit = sorted_profit
    decile_average_profit = np.average(decile_profit)
    commulative_average_profit.append(decile_average_profit)

    commulative_response_lift = (commulative_response_rate/overall_response)*100
    commulative_average_profit_lift = (commulative_average_profit/average_profit)*100




    return commulative_response_lift, commulative_average_profit_lift

def get_cost_vector(profit,beta,cost):
    m = np.mean(profit)
    threshold = m * beta
    index = np.where(profit<=threshold) # get index of profit smaller than threshold
    profit[0:len(profit)] = cost[1] #profit / m
    profit[index] = cost[0]  # smaller than threshold no cost is applied, the rest no change Bp

    return theano.shared(np.asarray(profit, dtype=theano.config.floatX), borrow=True)

if __name__ == '__main__':
    pro = np.array([1,2,3,3.2,5.4])
