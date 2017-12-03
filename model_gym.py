import single_model
import pandas as pd
import matplotlib.pyplot as pyplot
from keras.regularizers  import L1L2


ticker_list = ['AAPL', 'GOOG', 'FB', 'NFLX']
# TODO: Watch out. Global variables.
batch_size = 32
timesteps = 16
train_test_val_ratio = [0.6, 0.3, 0.1]
classification = False  # if classification True, then model is trained to predict whether
                       # the stock will be up or down. Otherwise, it's trained on regression.
epochs = 50
is_synthetic_data = False
# reg = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]
reg = L1L2(l1=0.01, l2=0.01)
                                                                # bias weight regularization;
                                                                # input weight regularization;
                                                                # recurrect weight regularization;


if __name__ == '__main__':
    # do something
    result_folder = single_model._prepare_results_folder('_'.join([i for i in ticker_list]))

    for ticker in ticker_list:

        # train the model
        # save the weight file
        # if file exist, then pick it. otherwise, train from scratch.

        error_score = pd.DataFrame()
        TS_X_train, TS_Y_train, TS_X_test, TS_Y_test, TS_X_val, TS_Y_val = \
                single_model.get_formated_data(ticker, train_test_val_ratio=[0.6, 0.3, 0.1], output_shape=(batch_size, timesteps, 5),
                                  classification=classification)

        val_score = []
        test_score = []
        model, config = single_model.training_model(ticker, epochs=epochs, classification=classification, result_folder=result_folder, load_weights=True, save_weights=True)
        val_score.append(
            single_model.evaluate_model(model, X=TS_X_val, Y=TS_Y_val, result_folder=result_folder, ticker=ticker,
                                        classification=classification, labels=['val_true', 'val_predicted']))
        test_score.append(
            single_model.evaluate_model(model, X=TS_X_test, Y=TS_Y_test, result_folder=result_folder, ticker=ticker,
                                        classification=classification, labels=['test_true', 'test_predicted']))


        # Print average scores
        error_score['validation'] = val_score
        error_score['test'] = test_score
        error_score.boxplot()
        pyplot.savefig(result_folder+'{0}_average_error_score.png'.format(ticker))

        #Print error trend based on the tikcer size
        # TODO


        config['runtime'] = {'ticker': ticker}  # adding some runtime parameters to config dump
        config['model'] = model.to_json()  # adding model description to config dump
        single_model._dump_config(result_folder, config)

    print(result_folder)


