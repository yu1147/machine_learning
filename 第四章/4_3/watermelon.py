
# using pandas dataframe for .csv read which contains chinese char.
import pandas as pd
import decision_tree
data_file_encode = "gb18030"  # the watermelon_3.csv is file codec type
with open("D:/mechain_learning/第四章/4_3/watermelon3a.csv", mode = 'r', encoding = data_file_encode) as data_file:
    df = pd.read_csv(data_file)


root = decision_tree.Treegen(df)

accuracy_scores = []

'''
from random import sample
for i in range(10):
    train = sample(range(len(df.index)), int(1*len(df.index)/2))

    df_train = df.iloc[train]
    df_test = df.drop(train)
    # generate the tree
    root = decision_tree.TreeGenerate(df_train)
    # test the accuracy
    pred_true = 0
    for i in df_test.index:
        label = decision_tree.Predict(root, df[df.index == i])
        if label == df_test[df_test.columns[-1]][i]:
            pred_true += 1

    accuracy = pred_true / len(df_test.index)
    accuracy_scores.append(accuracy)
'''
# k-folds cross prediction

# k-folds cross prediction

n = len(df.index)
k = 5
for i in range(k):
    m = int(n / k)
    test = []
    for j in range(i * m, i * m + m):
        test.append(j)

    df_train = df.drop(test)
    df_test = df.iloc[test]
    root = decision_tree.Treegen(df_train)  # generate the tree

    # test the accuracy
    pred_true = 0
    for i in df_test.index:
        label = decision_tree.predict(root, df[df.index == i])
        if label == df_test[df_test.columns[-1]][i]:
            pred_true += 1

    accuracy = pred_true / len(df_test.index)
    accuracy_scores.append(accuracy)

# print the prediction accuracy result
accuracy_sum = 0
print("accuracy: ", end="")
for i in range(k):
    print("%.3f  " % accuracy_scores[i], end="")
    accuracy_sum += accuracy_scores[i]
print("\naverage accuracy: %.3f" % (accuracy_sum / k))

# dicision tree visualization using pydotplus.graphviz
root = decision_tree.Treegen(df)

decision_tree.DrawPNG(root, "decision_tree_ID3.png")