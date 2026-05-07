from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, f1_score
import pandas as pd
import numpy as np


df = pd.read_csv('large_tot_df.csv')

for kw in df["keyword"].unique():

    kw_df = df[df["keyword"] == kw]
    y_true = kw_df['record_type'].map(
                    {'positive': 1,
                    'negative': 0,
                    'close_negative': 0}
                    )
    y_pred = kw_df['wctc_loss']
    y_pred = stats.zscore(-y_pred)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
print('False positive rate:', fpr)
print('True positive rate:', tpr)
print('Threholds:', thresholds)

# mean Average Precision (mAP)
mAP = average_precision_score(y_true, y_pred)
print('mean Average Precision (mAP):', mAP)

# Precision–Recall
#precision, recall = precision_recall_curve(y_true, y_pred)

# Graph: Precision–Recall curve
#plt.figure()
#plt.plot(recall, precision, label=f"PR curve (AP = {mAP:.3f})")
#plt.xlabel("Recall")
#plt.ylabel("Precision")
#plt.title("Precision–Recall Curve")
#plt.legend()
#plt.show()

# Graph: ROC curve
#plt.figure()
#plt.plot(fpr, tpr)
#plt.xlabel("False Positive Rate")
#plt.ylabel("True Positive Rate")
#plt.title("ROC Curve")
#plt.show()