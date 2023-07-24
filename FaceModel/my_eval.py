from sklearn.metrics import roc_auc_score
def score(y_true,y_pred):
    return roc_auc_score(y_true,y_pred)

def score1(true_annos_path = "/home/lvlv/lv_nian_zu/2022年计算机能力挑战赛/人脸对比/my_project_Lv_shao/init_data/toUser/train/annos.csv",
           pred_path = "/home/lvlv/lv_nian_zu/2022年计算机能力挑战赛/人脸对比/my_project_Lv_shao/model/result.csv"):
    f_true = open(true_annos_path,'r')
    y_true = f_true.readlines()
    f_true.close()

    f_pred = open(pred_path,'r')
    y_pred = f_pred.readlines()
    f_pred.close()
    y_true = y_true[1:]
    y_true1 = [int(i.split(',')[1]) for i in y_true]
    y_pred = y_pred[1:]
    y_pred1 = [float(i.split(',')[1]) for i in y_pred]
    # print(y_true1)
    AUC = score(y_true1,y_pred1)
    return AUC

if __name__ == "__main__":
    # true_annos_path = r"E:/shao_mei_project/my_project_Lv_shao/init_data/toUser/train/annos.csv"
    # pred_path = r"E:/shao_mei_project/my_project_Lv_shao/model/result.csv"
    true_annos_path = "/home/lvlv/lv_nian_zu/2022年计算机能力挑战赛/人脸对比/new_lv_shao_mei/my_project_Lv_shao/init_data/toUser/train/annos.csv"
    pred_path = "/home/lvlv/lv_nian_zu/2022年计算机能力挑战赛/人脸对比/提交/shao/model/result.csv"
    f_true = open(true_annos_path,'r')
    y_true = f_true.readlines()
    f_true.close()

    f_pred = open(pred_path,'r')
    y_pred = f_pred.readlines()
    f_pred.close()
    y_true = y_true[1:]
    y_true1 = [int(i.split(',')[1]) for i in y_true]
    y_pred = y_pred[1:]
    y_pred1 = [float(i.split(',')[1]) for i in y_pred]
    # print(y_true1)
    AUC = score(y_true1,y_pred1)
    print('AUC is:',AUC)
    # score()
    pass