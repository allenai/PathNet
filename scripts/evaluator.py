import sys
import json


def load(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def compute_accuracy(predictions, gold_answers):
    assert len(predictions) == len(gold_answers)
    correct_prediction_counter = 0
    for (ID, gold_answer) in gold_answers.items():
        correct_prediction_counter += (gold_answer == predictions[ID])
    accuracy = float(correct_prediction_counter)/float(len(gold_answers))
    return accuracy


def main(args):
    """
    Takes two arguments: (1) the directory of a file with predictions in JSON format,
    and (2) the name of the dataset part for which these predictions were computed.
    """

    # parse input arguments
    predictions_path = args[1]  # predictions file
    gold_path = args[2]         # gold annotations file

    # load predictions and gold answers from corresponding files
    predictions = load(predictions_path)
    gold_data = load(gold_path)

    # dictionary with gold answer for each id
    gold_answers = {element['id']: element['answer'] for element in gold_data}

    # # compute accuracy and print.
    accuracy = compute_accuracy(predictions, gold_answers)
    print(accuracy)


if __name__ == "__main__":
    main(sys.argv)
