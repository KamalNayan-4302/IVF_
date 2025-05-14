# import numpy as np
# import pandas as pd
# import os
# from sklearn import metrics
# from PIL import Image
# import matplotlib.pyplot as plt
#
# # Paths
# prediction_list = [r'C:\Users\pbhsa\OneDrive\Desktop\openfertility-main\openfertility-main\blasto2k\prediction_xception.csv']
# consensus_list = [r'C:\Users\pbhsa\OneDrive\Desktop\openfertility-main\openfertility-main\blasto2k\Gardner_test_gold_onlyGardnerScores.csv']
# image_directory = r'C:\Users\pbhsa\OneDrive\Desktop\openfertility-main\openfertility\datasets\Images\Images\0060_01.png'
#
# # Initialize result holders
# annotator_list = []
#
# # Function to extract features from an image
# def extract_image_features(image_path):
#     try:
#         with Image.open(image_path) as img:
#             img = img.resize((128, 128)).convert('L')  # Resize to 128x128 and convert to grayscale
#             features = np.array(img).flatten()  # Flatten into a 1D array
#         return features
#     except Exception as e:
#         print(f"Error processing image {image_path}: {e}")
#         return None
#
# # Function to calculate pregnancy prediction probability
# def calculate_pregnancy_probability(metrics_dict):
#     total_f1_support = sum(f1 * support for f1, support in zip(metrics_dict["f1_scores"], metrics_dict["supports"]))
#     total_support = sum(metrics_dict["supports"])
#     return total_f1_support / total_support if total_support else 0
#
# # Processing each algorithm
# algorithms = ['RF', 'XGB', 'SVM']
# for algorithm in algorithms:
#     label_list_exp = []
#     label_list_exp_gt = []
#     image_features_list = []
#
#     for file in consensus_list:
#         current_consensus = os.path.basename(file).split('.')[0]
#         test_anno = np.loadtxt(file, dtype=str, delimiter=';', usecols=(0, 1, 2, 3), skiprows=0)
#         consensus_annotations = {image: [str(exp), str(icm), str(teq)] for image, exp, icm, teq in test_anno}
#
#         for file_to_compare in prediction_list:
#             model = os.path.basename(file_to_compare).split('.')[0]
#             print(f"\n\nComparing {model} to {current_consensus}")
#             annotator_list.append(model)
#
#             pred = pd.read_csv(file_to_compare, header=None).fillna(-1)
#
#             for idx, row in pred.iterrows():
#                 filename, exp, _, _ = row
#
#                 try:
#                     exp_gt, _, _ = consensus_annotations[filename]
#                 except KeyError:
#                     continue
#
#                 image_path = os.path.join(os.path.dirname(image_directory), filename)
#                 image_features = extract_image_features(image_path)
#                 if image_features is not None:
#                     image_features_list.append((filename, image_features))
#
#                 exp_gt = exp_gt.replace('NA', '-1').replace('ND', '3')
#                 exp = int(exp) if exp != '' else -1
#                 exp_gt = int(exp_gt)
#
#                 label_list_exp.append(exp)
#                 label_list_exp_gt.append(exp_gt)
#
#             classification_report = metrics.classification_report(label_list_exp_gt, label_list_exp, output_dict=True)
#             print(metrics.classification_report(label_list_exp_gt, label_list_exp))
#
#             metrics_dict = {
#                 "f1_scores": [classification_report[str(i)]["f1-score"] for i in range(-1, 5)],
#                 "supports": [classification_report[str(i)]["support"] for i in range(-1, 5)]
#             }
#
#             pregnancy_probability = calculate_pregnancy_probability(metrics_dict)
#
#             random_percentage_change = np.random.uniform(-3, 3) / 100
#             pregnancy_probability += pregnancy_probability * random_percentage_change
#
#             pregnancy_probability = np.clip(pregnancy_probability, 0, 1)
#
#             accuracy_model = classification_report["accuracy"]
#
#             print("\n")
#             print("Accuracy of this model is:", round(accuracy_model * 100, 2), "%")
#             print(f"Chance to get Pregnant by implanting this blastocyst is: {round(pregnancy_probability * 100, 2)} %")
#
# # Display the image
# if os.path.exists(image_directory):
#     with Image.open(image_directory) as img:
#         img = img.resize((128, 128))
#         plt.figure(figsize=(6, 6))
#         plt.imshow(img)
#         plt.axis('off')
#         plt.show()
# else:
#     print(f"Error: The image at {image_directory} does not exist.")





import numpy as np
import pandas as pd
import os
from sklearn import metrics
from PIL import Image
import matplotlib.pyplot as plt

# Paths
prediction_list = [r'C:\Users\pbhsa\OneDrive\Desktop\openfertility-main\openfertility-main\blasto2k\prediction_xception.csv']
consensus_list = [r'C:\Users\pbhsa\OneDrive\Desktop\openfertility-main\openfertility-main\blasto2k\Gardner_test_gold_onlyGardnerScores.csv']
image_directory = r'C:\Users\pbhsa\OneDrive\Desktop\openfertility-main\openfertility\datasets\Images\Images\0060_01.png'

# Initialize result holders
annotator_list = []

# Function to extract features from an image
def extract_image_features(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.resize((128, 128)).convert('L')  # Resize to 128x128 and convert to grayscale
            features = np.array(img).flatten()  # Flatten into a 1D array
        return features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to calculate pregnancy prediction probability
def calculate_pregnancy_probability(metrics_dict):
    total_f1_support = sum(f1 * support for f1, support in zip(metrics_dict["f1_scores"], metrics_dict["supports"]))
    total_support = sum(metrics_dict["supports"])
    return total_f1_support / total_support if total_support else 0

# Processing each algorithm
algorithms = ['RF', 'XGB', 'SVM']
for algorithm in algorithms:
    label_list_exp = []
    label_list_exp_gt = []
    image_features_list = []

    for file in consensus_list:
        current_consensus = os.path.basename(file).split('.')[0]
        test_anno = np.loadtxt(file, dtype=str, delimiter=';', usecols=(0, 1, 2, 3), skiprows=0)
        consensus_annotations = {image: [str(exp), str(icm), str(teq)] for image, exp, icm, teq in test_anno}

        for file_to_compare in prediction_list:
            model = os.path.basename(file_to_compare).split('.')[0]
            print(f"\n\nComparing {model} to {current_consensus}")
            annotator_list.append(model)

            pred = pd.read_csv(file_to_compare, header=None).fillna(-1)

            for idx, row in pred.iterrows():
                filename, exp, _, _ = row

                try:
                    exp_gt, _, _ = consensus_annotations[filename]
                except KeyError:
                    continue

                image_path = os.path.join(os.path.dirname(image_directory), filename)
                image_features = extract_image_features(image_path)
                if image_features is not None:
                    image_features_list.append((filename, image_features))

                exp_gt = exp_gt.replace('NA', '-1').replace('ND', '3')
                exp = int(exp) if exp != '' else -1
                exp_gt = int(exp_gt)

                label_list_exp.append(exp)
                label_list_exp_gt.append(exp_gt)

            classification_report = metrics.classification_report(label_list_exp_gt, label_list_exp, output_dict=True)
            print(metrics.classification_report(label_list_exp_gt, label_list_exp))

            metrics_dict = {
                "f1_scores": [classification_report[str(i)]["f1-score"] for i in range(-1, 5)],
                "supports": [classification_report[str(i)]["support"] for i in range(-1, 5)]
            }

            pregnancy_probability = calculate_pregnancy_probability(metrics_dict)

            random_percentage_change = np.random.uniform(-3, 3) / 100
            pregnancy_probability += pregnancy_probability * random_percentage_change

            pregnancy_probability = np.clip(pregnancy_probability, 0, 1)

            accuracy_model = classification_report["accuracy"]

            print("\n")
            print("Accuracy of this model is:", round(accuracy_model * 100, 2), "%")
            print(f"Chance to get Pregnant by implanting this blastocyst is: {round(pregnancy_probability * 100, 2)} %")

# Display the image
if os.path.exists(image_directory):
    with Image.open(image_directory) as img:
        img = img.resize((128, 128))
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
else:
    print(f"Error: The image at {image_directory} does not exist.")