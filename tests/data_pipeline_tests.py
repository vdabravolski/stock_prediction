import unittest
import pandas as pd
import math
import random
import numpy as np
import data_pipelines as dp

class TestStringMethods(unittest.TestCase):

    def _get_data(self, degree_range=(0, 361)):
        sin_data = [math.sin(math.radians(i)) for i in range(degree_range[0], degree_range[1], 1)]
        cos_data = [math.cos(math.radians(i)) for i in range(degree_range[0], degree_range[1], 1)]
        date_data = [random.randint(0,100) for i in range(degree_range[0], degree_range[1], 1)] #e for compatibility with expected data

        df = pd.DataFrame(data={'close': sin_data, 'open': cos_data, 'date': date_data})
        return df

    def test_timesteps_conversion(self):
        # This is a test to validate that data shape is correct after conversion to timesteps shape
        input_data = self._get_data(degree_range=(0,100))
        print("Input data shape: {0}".format(input_data.shape))
        exp_shape = (90, 10, 2) # we are passing only two feature 'open' & 'close. 'date' column is dropped.

        output_data = dp.convert_data_to_batch_timesteps(input_data, batch_size=10, timesteps=10, features=2)

        act_shape = np.shape(output_data)
        self.assertEqual(exp_shape, act_shape)

    def test_timesteps_values(self):
        # To verify that values are correct after timesteps conversion
        input_data = self._get_data(degree_range=(0, 100))

        output_data = dp.convert_data_to_batch_timesteps(input_data, batch_size=10, timesteps=10, features=2)

        i = random.randint(0, 9) # random batch size index
        exp_sample = input_data.close.tolist()[i:(i+10)]
        act_sample = output_data[i, :, 0].tolist()

        print(exp_sample)
        print(act_sample)

        self.assertListEqual(exp_sample, act_sample)


    def test_labels_conversion_1(self):
        data_df = self._get_data(degree_range=(0,361))
        exp_shape = (359, 3)
        output_data = dp.convert_ts_to_categorial(data_df, timesteps=1)

        act_shape = output_data.shape
        self.assertEqual(exp_shape, act_shape)

    def test_labels_conversion_5(self):
        data_df = self._get_data(degree_range=(0,361))
        exp_shape = (355, 3)

        output_data = dp.convert_ts_to_categorial(data_df, timesteps=5)

        act_shape = output_data.shape
        self.assertEqual(exp_shape, act_shape)

    def test_label_conversion_labels_value_up(self):
        data_df = self._get_data(degree_range=(0,90))

        exp_values = [1 for i in range(88)] # for sin function this will be monotoneously growing function from 0,90 degrees.
                                            # Therefore, expecting only ones.
        act_values = dp.convert_ts_to_categorial(data_df, timesteps=1).close_bool.tolist()

        print(exp_values)
        print(act_values)
        self.assertListEqual(exp_values, act_values)

    def test_label_conversion_labels_value_down(self):
        data_df = self._get_data(degree_range=(90,180))

        exp_values = [0 for _ in range(88)] # for sine function this will be monotoneously growing function from 0,90 degrees.
                                            # Therefore, expecting only ones.

        act_values = dp.convert_ts_to_categorial(data_df, timesteps=1).close_bool.tolist()

        print(exp_values)
        print(act_values)
        self.assertListEqual(exp_values, act_values)

    def test_label_conversion_news_1(self):
        day = [1]
        day_next = [2]
        day_prev = [0]
        [diff_bool, diff_bool_next, day] = dp._convert_labels_news(day_prev, day, day_next)
        self.assertListEqual([diff_bool, diff_bool_next, day], [1, 1, 1])

    def test_label_conversion_news_2(self):
        day = [1]
        day_next = [0]
        day_prev = [3]
        [diff_bool, diff_bool_next, day] = dp._convert_labels_news(day_prev, day, day_next)
        self.assertListEqual([diff_bool, diff_bool_next, day], [0, 0, 1])

    def test_label_conversion_news_2(self):
        day = [2]
        day_next = [1]
        day_prev = [1]
        [diff_bool, diff_bool_next, day] = dp._convert_labels_news(day_prev, day, day_next)
        self.assertListEqual([diff_bool, diff_bool_next, day], [1, 0, 2])


if __name__ == '__main__':
    unittest.main()