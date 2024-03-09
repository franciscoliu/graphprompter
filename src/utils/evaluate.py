import json
import pandas as pd
import re
import numpy as np



def get_accuracy_cora(eval_output, path):
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # Convert Tensors/NumPy arrays to Python lists before saving to csv
    with open(path, 'w') as f:
        for index, row in df.iterrows():
            # Assuming 'pred_prob' is a list of lists or a list of tensors
            row['pred_prob'] = [prob.tolist() if hasattr(prob, 'tolist') else prob for prob in row['pred_prob']]
            # f.write(json.dumps(dict(row)) + '\n')

    contains_zero = any(0 in label for label in df['label'])
    contains_one = any(1 in label for label in df['label'])

    print(f"Contains 0: {contains_zero}")
    print(f"Contains 1: {contains_one}")

    # Compute accuracy
    correct = 0
    total_predictions = 0
    for pred_probs, labels in zip(df['pred_prob'], df['label']):
        for pred_prob, label in zip(pred_probs, labels):
            predicted_label = np.argmax(pred_prob)
            if predicted_label == label:
                correct += 1
            total_predictions += 1

    return correct / total_predictions

    # # eval_output is a list of dicts
    # df = pd.concat([pd.DataFrame(d) for d in eval_output])
    # with open(path, 'w')as f:
    #     for index, row in df.iterrows():
    #         f.write(json.dumps(dict(row))+'\n')
    #
    # # compute accuracy
    # classes = ['Case Based', 'Genetic Algorithms', 'Neural Networks', 'Probabilistic Method', 'Reinforcement Learning', 'Rule Learning', 'Theory']
    # classes_regex = '(' + '|'.join(classes) + ')'
    # correct = 0
    # for pred, label in zip(df['pred'], df['label']):
    #     matches = re.findall(classes_regex, pred)
    #     if len(matches) > 0 and matches[0] == label:
    #         correct += 1
    #
    # return correct/len(df)


def get_accuracy_pubmed(eval_output, path):
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # Convert Tensors/NumPy arrays to Python lists before saving to csv
    with open(path, 'w') as f:
        for index, row in df.iterrows():
            # Assuming 'pred_prob' is a list of lists or a list of tensors
            row['pred_prob'] = [prob.tolist() if hasattr(prob, 'tolist') else prob for prob in row['pred_prob']]
            # f.write(json.dumps(dict(row)) + '\n')

    contains_zero = any(0 in label for label in df['label'])
    contains_one = any(1 in label for label in df['label'])

    print(f"Contains 0: {contains_zero}")
    print(f"Contains 1: {contains_one}")

    # Compute accuracy
    correct = 0
    total_predictions = 0
    for pred_probs, labels in zip(df['pred_prob'], df['label']):
        for pred_prob, label in zip(pred_probs, labels):
            predicted_label = np.argmax(pred_prob)
            if predicted_label == label:
                correct += 1
            total_predictions += 1

    return correct / total_predictions

    # eval_output is a list of dicts
    # df = pd.concat([pd.DataFrame(d) for d in eval_output])
    #
    # # save to csv
    # with open(path, 'w')as f:
    #     for index, row in df.iterrows():
    #         f.write(json.dumps(dict(row))+'\n')
    #
    # # compute accuracy
    # correct = 0
    # for pred, label in zip(df['pred'], df['label']):
    #     if label in pred:
    #         correct += 1
    #
    # return correct/len(df)


def get_accuracy_citeseer(eval_output, path):
    # Convert eval_output (list of dicts) to a DataFrame
    df = pd.concat([pd.DataFrame(d) for d in eval_output])


    # Convert Tensors/NumPy arrays to Python lists before saving to csv
    with open(path, 'w') as f:
        for index, row in df.iterrows():
            # Assuming 'pred_prob' is a list of lists or a list of tensors
            row['pred_prob'] = [prob.tolist() if hasattr(prob, 'tolist') else prob for prob in row['pred_prob']]
            # f.write(json.dumps(dict(row)) + '\n')

    contains_zero = any(0 in label for label in df['label'])
    contains_one = any(1 in label for label in df['label'])

    print(f"Contains 0: {contains_zero}")
    print(f"Contains 1: {contains_one}")

    # Compute accuracy
    correct = 0
    total_predictions = 0
    for pred_probs, labels in zip(df['pred_prob'], df['label']):
        for pred_prob, label in zip(pred_probs, labels):
            predicted_label = np.argmax(pred_prob)
            if predicted_label == label:
                correct += 1
            total_predictions += 1

    return correct / total_predictions

# def get_accuracy_citeseer(eval_output, path, threshold=0.5):
#     df = pd.concat([pd.DataFrame(d) for d in eval_output])
#
#     with open(path, 'w') as f:
#         for index, row in df.iterrows():
#             row['pred_prob'] = [prob.tolist() if hasattr(prob, 'tolist') else prob for prob in row['pred_prob']]
#
#     correct = 0
#     total_predictions = 0
#     for i in range(len(df)):
#         pred_probs = df.iloc[i]['pred_prob']
#         labels = df.iloc[i]['label']
#
#         # Ensure pred_probs and labels are lists
#         if not isinstance(pred_probs, list):
#             pred_probs = [pred_probs]
#         if not isinstance(labels, list):
#             labels = [labels]
#
#         # Flatten pred_probs and labels if they are nested lists
#         if isinstance(pred_probs[0], list):
#             pred_probs = [item for sublist in pred_probs for item in sublist]
#         if isinstance(labels[0], list):
#             labels = [item for sublist in labels for item in sublist]
#
#         # Debugging: Check the type of pred_probs elements
#         for pred_prob in pred_probs:
#             if not isinstance(pred_prob, (float, int)):
#                 print(f"Invalid type for pred_prob: {type(pred_prob)}. Expected float or int.")
#
#         for pred_prob, label in zip(pred_probs, labels):
#             predicted_label = 1 if pred_prob > threshold else 0
#             if predicted_label == label:
#                 correct += 1
#             total_predictions += 1
#
#     return correct / total_predictions


def get_accuracy_arxiv(eval_output, path):
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # Convert Tensors/NumPy arrays to Python lists before saving to csv
    with open(path, 'w') as f:
        for index, row in df.iterrows():
            # Assuming 'pred_prob' is a list of lists or a list of tensors
            row['pred_prob'] = [prob.tolist() if hasattr(prob, 'tolist') else prob for prob in row['pred_prob']]
            # f.write(json.dumps(dict(row)) + '\n')

    contains_zero = any(0 in label for label in df['label'])
    contains_one = any(1 in label for label in df['label'])

    print(f"Contains 0: {contains_zero}")
    print(f"Contains 1: {contains_one}")

    # Compute accuracy
    correct = 0
    total_predictions = 0
    for pred_probs, labels in zip(df['pred_prob'], df['label']):
        for pred_prob, label in zip(pred_probs, labels):
            predicted_label = np.argmax(pred_prob)
            if predicted_label == label:
                correct += 1
            total_predictions += 1

    return correct / total_predictions

    #
    # # eval_output is a list of dicts
    # df = pd.concat([pd.DataFrame(d) for d in eval_output])
    #
    # # save to csv
    # with open(path, 'w')as f:
    #     for index, row in df.iterrows():
    #         f.write(json.dumps(dict(row))+'\n')
    #
    # # compute accuracy
    # correct = 0
    # for pred, label in zip(df['pred'], df['label']):
    #     matches = re.findall(r"cs\.[a-z]{2}", pred.strip())
    #     if len(matches) > 0 and label == matches[0]:
    #         correct += 1
    #
    # return correct/len(df)


def get_accuracy_products(eval_output, path):
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # Convert Tensors/NumPy arrays to Python lists before saving to csv
    with open(path, 'w') as f:
        for index, row in df.iterrows():
            # Assuming 'pred_prob' is a list of lists or a list of tensors
            row['pred_prob'] = [prob.tolist() if hasattr(prob, 'tolist') else prob for prob in row['pred_prob']]
            # f.write(json.dumps(dict(row)) + '\n')

    contains_zero = any(0 in label for label in df['label'])
    contains_one = any(1 in label for label in df['label'])

    print(f"Contains 0: {contains_zero}")
    print(f"Contains 1: {contains_one}")

    # Compute accuracy
    correct = 0
    total_predictions = 0
    for pred_probs, labels in zip(df['pred_prob'], df['label']):
        for pred_prob, label in zip(pred_probs, labels):
            predicted_label = np.argmax(pred_prob)
            if predicted_label == label:
                correct += 1
            total_predictions += 1

    return correct / total_predictions

    # # eval_output is a list of dicts
    # df = pd.concat([pd.DataFrame(d) for d in eval_output])
    #
    # # save to csv
    # with open(path, 'w')as f:
    #     for index, row in df.iterrows():
    #         f.write(json.dumps(dict(row))+'\n')
    #
    # # compute accuracy
    # correct = 0
    # classes = ['Home & Kitchen',
    #            'Health & Personal Care',
    #            'Beauty',
    #            'Sports & Outdoors',
    #            'Books',
    #            'Patio, Lawn & Garden',
    #            'Toys & Games',
    #            'CDs & Vinyl',
    #            'Cell Phones & Accessories',
    #            'Grocery & Gourmet Food',
    #            'Arts, Crafts & Sewing',
    #            'Clothing, Shoes & Jewelry',
    #            'Electronics',
    #            'Movies & TV',
    #            'Software',
    #            'Video Games',
    #            'Automotive',
    #            'Pet Supplies',
    #            'Office Products',
    #            'Industrial & Scientific',
    #            'Musical Instruments',
    #            'Tools & Home Improvement',
    #            'Magazine Subscriptions',
    #            'Baby Products',
    #            'NaN',
    #            'Appliances',
    #            'Kitchen & Dining',
    #            'Collectibles & Fine Art',
    #            'All Beauty',
    #            'Luxury Beauty',
    #            'Amazon Fashion',
    #            'Computers',
    #            'All Electronics',
    #            'Purchase Circles',
    #            'MP3 Players & Accessories',
    #            'Gift Cards',
    #            'Office & School Supplies',
    #            'Home Improvement',
    #            'Camera & Photo',
    #            'GPS & Navigation',
    #            'Digital Music',
    #            'Car Electronics',
    #            'Baby',
    #            'Kindle Store',
    #            'Buy a Kindle',
    #            'Furniture & Decor',
    #            '#508510']
    #
    # classes_regex = '(' + '|'.join(classes) + ')'
    # correct = 0
    # for pred, label in zip(df['pred'], df['label']):
    #     matches = re.findall(classes_regex, pred)
    #     if len(matches) > 0 and matches[0] == label:
    #         correct += 1
    #
    # return correct/len(df)


eval_funcs = {
    'cora': get_accuracy_cora,
    'citeseer': get_accuracy_citeseer,
    'pubmed': get_accuracy_pubmed,
    'arxiv': get_accuracy_arxiv,
    'products': get_accuracy_products
}
