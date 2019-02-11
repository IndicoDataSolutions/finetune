import json
import os
import codecs
import random
import requests
from pathlib import Path
import finetune
from finetune import MultipleChoice
from sklearn.model_selection import train_test_split
import numpy as np

QnA = "QandA"
DATA_PATH = os.path.join('Data', 'QA', QnA)
CHECKSUM = ""


def download_data():
    base_url = "https://raw.githubusercontent.com/uberspot/OpenTriviaQA/master/categories/{}"

    file_list = [
        "animals",
        "general",
        "geography",
        "history",
        "literature",
        "people"
    ]

    for filename in file_list:
        folder = DATA_PATH
        if not os.path.exists(folder):
            os.mkdir(folder)

        local_filepath = os.path.join(folder, filename)

        print(base_url.format(filename))
        data = requests.get(base_url.format(filename)).content
        fd = open(local_filepath, 'wb')
        fd.write(data)
        fd.close()

def parse_file(filename):
    with codecs.open(filename, encoding='ISO-8859-2') as file:
        file_content = file.readlines()
    questions = []
    last_question = None
    inside_question = False
    correct_answer = None
    for line in file_content:
        line = line.rstrip()
        if line == "":
            continue
        elif line[0] == "#":
            if last_question is not None:
                last_question["correct_answer"] = last_question["answers"].index(correct_answer)
                questions.append(last_question)

            last_question = {
                "content": line[3:],
                "answers": [],
            }
            inside_question = True
        elif line[0] == "^":
            correct_answer = line[2:]
            inside_question = False
        elif line[0] == "A" or line[0] == "B" or line[0] == "C" or line[0] == "D":
            last_question["answers"].append(line[2:])
        elif inside_question:
            last_question["content"] += "\n" + line
        elif line.strip() != "":
            raise Exception("Can't parse line in file {0}: {1}".format(filename, line))

    if last_question is not None:
        last_question["correct_answer"] = last_question["answers"].index(correct_answer)
        questions.append(last_question)
    return questions

def get_dataset(nrows=None):
    download_data()
    data = []
    files = os.listdir(DATA_PATH)
    for file in files:
        if not file.startswith('.'):
            data.extend(parse_file(os.path.join(DATA_PATH, file)))
    random.shuffle(data)
    return data[:nrows]


if __name__ == "__main__":
    print("DISCLAIMER: THIS DATASET IS NOT WELL SUITED TO THIS MODEL AND DOES NOT ACHIEVE RESULTS MUCH BETTER THAN "
          "RANDOM CHANCE. IT IS HERE FOR API DEMO PURPOSES DUE TO LICENCES OF AVAILABLE DATASETS.")

    data = get_dataset()

    model = MultipleChoice(n_epochs=1)
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    train_qs = []
    train_ans_correct = []
    train_ans = []
    for item in train_data:
        answers = item["answers"]
        correct_ans_idx = item["correct_answer"]
        train_ans_correct.append(answers[correct_ans_idx])
        train_qs.append(item["content"])
        train_ans.append(answers)

    test_qs = []
    test_ans_all = []
    test_ans_correct = []
    for item in test_data:
        answers = item["answers"]
        correct_ans_idx = item["correct_answer"]
        test_ans_correct.append(answers[correct_ans_idx])
        test_qs.append(item["content"])
        test_ans_all.append(answers)

    model.fit(train_qs, train_ans, train_ans_correct)
    print(list(zip(model.predict(test_qs, test_ans_all), test_qs)))

    accuracy = np.mean([p == t for p, t in zip(model.predict(test_qs, test_ans_all), test_ans_correct)])

    print('Test Accuracy: {:0.2f}'.format(accuracy))



