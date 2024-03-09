import json
import pandas as pd
import re


def get_accuracy_cora(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    classes = ['Case Based', 'Genetic Algorithms', 'Neural Networks', 'Probabilistic Method', 'Reinforcement Learning', 'Rule Learning', 'Theory']
    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(classes_regex, pred)
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)


def get_accuracy_pubmed(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        if label in pred:
            correct += 1

    return correct/len(df)


def get_accuracy_citeseer(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        if label in pred:
            correct += 1

    return correct/len(df)


def get_accuracy_arxiv(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(r"cs\.[a-z]{2}", pred.strip())
        if len(matches) > 0 and label == matches[0]:
            correct += 1

    return correct/len(df)


def get_accuracy_products(eval_output, path):

    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for index, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    # compute accuracy
    correct = 0
    classes = ['Home & Kitchen',
               'Health & Personal Care',
               'Beauty',
               'Sports & Outdoors',
               'Books',
               'Patio, Lawn & Garden',
               'Toys & Games',
               'CDs & Vinyl',
               'Cell Phones & Accessories',
               'Grocery & Gourmet Food',
               'Arts, Crafts & Sewing',
               'Clothing, Shoes & Jewelry',
               'Electronics',
               'Movies & TV',
               'Software',
               'Video Games',
               'Automotive',
               'Pet Supplies',
               'Office Products',
               'Industrial & Scientific',
               'Musical Instruments',
               'Tools & Home Improvement',
               'Magazine Subscriptions',
               'Baby Products',
               'NaN',
               'Appliances',
               'Kitchen & Dining',
               'Collectibles & Fine Art',
               'All Beauty',
               'Luxury Beauty',
               'Amazon Fashion',
               'Computers',
               'All Electronics',
               'Purchase Circles',
               'MP3 Players & Accessories',
               'Gift Cards',
               'Office & School Supplies',
               'Home Improvement',
               'Camera & Photo',
               'GPS & Navigation',
               'Digital Music',
               'Car Electronics',
               'Baby',
               'Kindle Store',
               'Buy a Kindle',
               'Furniture & Decor',
               '#508510']

    classes_regex = '(' + '|'.join(classes) + ')'
    correct = 0
    for pred, label in zip(df['pred'], df['label']):
        matches = re.findall(classes_regex, pred)
        if len(matches) > 0 and matches[0] == label:
            correct += 1

    return correct/len(df)


eval_funcs = {
    'cora': get_accuracy_cora,
    'citeseer': get_accuracy_citeseer,
    'pubmed': get_accuracy_pubmed,
    'arxiv': get_accuracy_arxiv,
    'products': get_accuracy_products
}
