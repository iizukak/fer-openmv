import dataset
import config
from predictor import Predictor
import numpy as np
import time


def benchmark():
    p = Predictor()

    print("LOAD DATASET: START")
    (_, _), (_, _), (x_private_test, y_private_test) = \
            dataset.load_dataset(config.DATASET_PATH)
    print("LOAD DATASET: DONE")

    correct_results = np.zeros(dataset.CLASS_NUM, dtype=np.int32)
    all_results = np.zeros(dataset.CLASS_NUM, dtype=np.int32)
    
    print("INFERENCE: START")
    start = time.time()
    for i in range(len(x_private_test)):
        result = p(x_private_test[i])

        # Increment number of case
        all_results[y_private_test[i]] += 1

        if np.argmax(result) == y_private_test[i]:
            correct_results[y_private_test[i]] += 1
    end = time.time()
    measured_time = (end - start) * 1000
    print("INFERENCE: DONE\n")
    
    # Print result for easch class
    for i in range(dataset.CLASS_NUM):
        print("{0}: Correct cases: {1} / {2}, Accuracy: {3:.2f}"\
              .format(dataset.CLASS_NAME[i],
                      correct_results[i],
                      all_results[i],
                      float(correct_results[i]) / float(all_results[i]) * 100))

    # Print result for all class
    print("{0}: Correct cases: {1} / {2}, Accuracy: {3:.2f}\n"\
       .format("All category",
               sum(correct_results),
               sum(all_results),
               (float(sum(correct_results)) / float(sum(all_results)) * 100)))

    print("{0:.2f} ms for one data".format(measured_time / len(x_private_test)))

    return all_results, correct_results, measured_time


if __name__ == '__main__':
    benchmark()
