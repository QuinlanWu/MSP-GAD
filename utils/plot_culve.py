from sklearn.metrics import roc_curve,auc,average_precision_score,precision_recall_curve
import matplotlib.pyplot as plt

def plot_ROC(y_test, recon_error_test, fig_name):    
    fpr, tpr, _ = roc_curve(y_test, recon_error_test)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(fig_name, dpi=1200)
    plt.close()
    # plt.show()

def plot_PRC(y_test, recon_error_test):
    average_precision = average_precision_score(y_test, recon_error_test)

    precision,recall,_ = precision_recall_curve(y_test, recon_error_test)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.4f}'.format(
              average_precision))
    plt.savefig('PRC',dpi=1200)
    plt.show()


def plot_loss_curve(loss_list):
    """
    绘制loss曲线图
    
    Args:
        loss_list (list): 包含每个epoch的loss值的列表
    """
    epochs = range(1, len(loss_list) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_list, marker='o', linestyle='-', 
             linewidth=1.5, markersize=3, markeredgewidth=0.5)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('epoch_loss.png')