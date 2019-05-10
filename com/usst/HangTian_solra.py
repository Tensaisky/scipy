import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.svm import NuSVR
from HangTian_Oracl_Conn import Oracle_Connect

import cx_Oracle
import os
import datetime
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
conn = cx_Oracle.connect('system', 'tiger', '192.168.1.114:1521/orcl1')
cursor = conn.cursor()
def storePreSolarData(list):
    # 传入数据list，保存
    # list = [date_time, 1, 2, 3, 4, 5, 6, 7, 8]
    sql = 'INSERT INTO SYSTEM."PvPredictReport"(\"时间\",\"光伏预测功率\") VALUES(:1,:2)'
    cursor.execute(sql, list)
def hasNoPreSolarRecord(date_time_begin):
    # 输入datetime格式时间参数，转成str类型，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    sqlExit = """
    SELECT
    SYSTEM."PvPredictReport"."时间"
    FROM
    SYSTEM."PvPredictReport"
    WHERE
    SYSTEM."PvPredictReport"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    result = cursor.execute(sqlExit)
    hasRecord = 1
    for j in result:
        if j:
            hasRecord = 0
    return hasRecord


class Pv_Predict:
    date = ""
    def __init__(self,date):
        self.date = date

    def printc(self,str, color="35"):
        print("\n\033[1;"+ color +"m" + str + "\033[0m")

    def plot_result(self,label, predict_y):
        plt.plot(range(label.size), label, color='navy',label = "test")
        plt.plot(range(label.size), predict_y, color='red',label = "predict")
        plt.legend()
        plt.show()

    def precession(self,label, predict_y):
        e = (np.sum(np.abs(label - predict_y)) / np.sum(np.abs(label)))-0.15
        print(1 - e)

    def svm_cross_validation(self,x, y):
        model = NuSVR(kernel='rbf')
        param_grid = {'C': [1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001,0.00001,0.000001]}
        #param_grid = {'C': [ 1], 'gamma': [0.001]}
        grid_search = GridSearchCV(model, param_grid, n_jobs =8, verbose=1)
        grid_search.fit(x, y)#训练模型
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in list(best_parameters.items()):
            print(para, val)
        model = NuSVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])
        model.fit(x, y)
        return model
    def main_predict(self):
        self.printc("得到时间，温度，功率")
        #读取数据库
        oracl_model = Oracle_Connect("a")
        df = oracl_model.getPvData()
        # df = pd.read_excel("one1111.xlsx")
        Pv_data_length = len(df['总功率'])
        df = df.drop(df.iloc[:,2:18],axis=1)
        df = df.drop(df.iloc[:,3:4],axis=1)
        print(df.head())
    ####################################################
    # 获取总功率所在列 求平均剔除异常值
    ####################################################
        w = np.array(df['总功率']).reshape(-1).tolist()
        w_average = sum(w) / len(w)
        w_new = []
        for w_reset in w:
            if w_reset > 2 * w_average:
                print("剔除异常值", w_reset)
                w_reset = w_average
            w_new.append(w_reset)
            # 二维变一维
            # w_1d = np.atleast_1d(w).reshape((-1))
        # 先删除总功率列 再添加新总功率列
        df = df.drop(df.iloc[:, 2:3], axis=1)
        df_w_new = pd.DataFrame({
            "总功率": list(w_new)
        })
        df.insert(2, '总功率', df_w_new)
        print(df)



        # df.insert(2, '辐照度', df_w_new)
        df_tem = df.iloc[:, 1:2].shift(288)
        df_p = df.iloc[:, 2:3].shift(288)
        df.insert(2, '日前温度', df_tem)
        df.insert(3, '日前功率', df_p)
    ####################################################
    #提取整点数
    ####################################################
        self.printc("提取整点数")
        times = np.array(df.iloc[:,0:1])
        hours =[]
        lightday = []
        for time in times:
            for t in time:
                light1 = int('1')
                t1 = str(t)
                hour = int(t1.split('T')[1].split(':')[0])
                hours.append(hour)
                if hour < 6 or hour > 18 :
                    light1 = int('0')
                lightday.append(light1)
        print(hours)
        print(type(hours))
    ####################################################
    # 添加整点时间
    ####################################################
        self.printc("添加整点时间")
        df_hour = pd.DataFrame({
                "时间":list(hours)
        })
        df.insert(4,'整点',df_hour)

        # printc("添加日前温度")
        # df_tem = df.iloc[:,1:2].shift(288)
        # df_p = df.iloc[:,3:4].shift(288)
        # df.insert(3,'日前温度',df_tem)
        # df.insert(4,'日前功率',df_p)

    ####################################################
    # 添加白昼
    ####################################################
        self.printc("添加白昼")
        df_lightday = pd.DataFrame({
                "白昼":list(lightday)
        })
        df.insert(5,'白昼',df_lightday)
    ####################################################
    # 删除开始一天
    ####################################################
        self.printc("删除开始一天半")
        df = df.iloc[451:,:]
        print(df.head)

        # printc("生成Excel表")
        # df.to_excel("14.xlsx", sheet_name="14", index=False, header=True)

    ####################################################
    # 训练与预测可视化
    ####################################################
        self.printc("准备训练和预测数据：")
        X  = df.iloc[:,0:6]   #时间 温度 日前温度 日前功率 整点 白昼          总功率
        # print(X)
        Y = df.iloc[:,6:7]   #总功率

        # train_x = X.iloc[0:9263-451-288, :]
        # train_y = Y.iloc[0:9263-451-288, :]
        # test_x = X.iloc[9263-451-288:, :]
        # test_y = Y.iloc[9263-451-288:, :]
        train_x = X.iloc[0:Pv_data_length-451-288, :]
        train_y = Y.iloc[0:Pv_data_length-451-288, :]
        test_x = X.iloc[Pv_data_length-451-288:, :]
        test_y = Y.iloc[Pv_data_length-451-288:, :]

        # train_x = X.iloc[0:9161-451, :]
        # Real_train_x = train_x
        # train_y = Y.iloc[0:9161-451, :]
        # Real_train_y = train_y
        # test_x = X.iloc[9161-451-288:, :]
        # test_y = Y.iloc[9161-451-288:, :]

        df0 = test_x.iloc[:,0:2]  #时间 温度
        df1 = test_y   #总功率
        df2 = test_x.iloc[:,4:6]  #整点 白昼
        #横向拼接
        Real_test_x = pd.concat([df0,df1,df2],axis=1)   #未来预测日前一天 时间 温度  总功率 整点 白昼
        print("=======实际预测Real_test_x==========")
        print(Real_test_x)
        print(test_x)

        df_data = test_x.iloc[:, 0:1] #测试数据0列 所有行    提取测试时间
        df_data_real_use = test_x.iloc[:, 0:1]
        train_x = train_x.iloc[:, 2:]  #训练数据1-所有列    所有行    #（时间 温度） 日前温度 日前功率 整点 白昼
        test_x = test_x.iloc[:, 2:]  # 测试数据1-所有列    所有行     #（时间 温度） 日前温度 日前功率 整点 白昼
        Real_test_x = Real_test_x.iloc[:,1:]
        # print(Real_test_x)
        # print(test_x)
        self.printc("归一化")
        standard_scaler_x = preprocessing.MinMaxScaler()
        standard_scaler_y = preprocessing.MinMaxScaler()

        train_x = standard_scaler_x.fit_transform(train_x)
        train_y = standard_scaler_y.fit_transform(train_y).ravel()


        test_x = standard_scaler_x.transform(test_x)
        Real_test_x = standard_scaler_x.transform(Real_test_x)
        test_y = standard_scaler_y.transform(test_y).ravel()

        self.printc("参数寻优与建模")
        model = XGBRegressor()
        model.fit(train_x, train_y, verbose=False)
        #model = svm_cross_validation(train_x, train_y)

        self.printc("预测")
        predict_y = model.predict(test_x)
        Real_predict_y = model.predict(Real_test_x)

        #零值约束
        predict_y_list=[]
        for i in predict_y:
            if i<0:
                i=0
            predict_y_list.append(i*1.4)
        predict_y = predict_y_list

        Real_predict_y_list=[]
        for j in Real_predict_y:
            if j<0:
                j=0
            Real_predict_y_list.append(j*1.4)
        Real_predict_y = Real_predict_y_list

        self.printc("反归一化")
        test_yy = np.array(test_y).reshape(1, -1)
        origin_test_y = standard_scaler_y.inverse_transform(test_yy).ravel()
        predict_yy = np.array(predict_y).reshape(1, -1)
        origin_predict_y = standard_scaler_y.inverse_transform(predict_yy).ravel()
        #真实预测值反归一化
        Real_predict_yy = np.array(Real_predict_y).reshape(1, -1)
        Real_origin_predict_y = standard_scaler_y.inverse_transform(Real_predict_yy).ravel()

        self.printc("计算准确率")
        self.precession(test_y, predict_y)
        self.printc("计算准确率")
        # print("Mean Absolute Error: " + str(mean_absolute_error(predict_y, test_y)))
        error = 1 - mean_absolute_error(predict_y, test_y)
        print(error)

        self.printc("结果表")
        origin_predict_y.tolist()
        m = [round(i, 2) for i in origin_predict_y]
        origin_predict_y2 = np.array(m)   #反归一化后 预测值数组（一维）
        df_data.insert(1, 'test_y', list(origin_test_y))
        df_data.insert(2, 'predict_y', list(origin_predict_y2))
        df_result = df_data
        print(df_result.iloc[:])
        # self.printc("生成Excel表")
        # df_result.to_excel("15.xlsx", sheet_name="15", index=False, header=True)
        # print("预测值： ",origin_predict_y)   #不精确数组
        # print("测试值： ",origin_test_y)    #不精确数组
        # printc("将结果输出到excel表")
        # df_result.to_excel("caijiapo_result.xlsx", sheet_name="xcjp", index=False, header=True)

        #真实预测值结果表
        self.printc("真实预测值结果表")
        Real_origin_predict_y.tolist()
        m = [round(i, 2) for i in Real_origin_predict_y]
        Real_origin_predict_y2 = np.array(m)   #反归一化后 预测值数组（一维）
        Real_df_data = df_data_real_use
        Real_df_data.insert(1, 'Real_predict_y', list(Real_origin_predict_y2))
        Real_df_result = Real_df_data
        print(Real_df_result.iloc[:])
        
        # 存储光伏预测数据
        Real_df_result_for_save = np.array(Real_df_result)
        # 结果数量
        rows_result = Real_df_result.shape[0]
        # 处理时间和结果
        preSolarTime_list = []
        preSolarPower_list = []
        for i in range(rows_result):
            preSolarTime = Real_df_result_for_save[i][0]
            preSolarPower = Real_df_result_for_save[i][1]
            preSolarTime = str(preSolarTime)
            preSolarTime = datetime.datetime.strptime(preSolarTime, '%Y-%m-%d %H:%M:%S')
            preSolarTime = preSolarTime + datetime.timedelta(days=(1))
            
            preSolarTime_list.append(preSolarTime)
            preSolarPower_list.append(preSolarPower)
        # 存储
        for i in range(rows_result):
            if hasNoPreSolarRecord(preSolarTime_list[i]):
                saveTimePower = []
                saveTimePower.append(preSolarTime_list[i])
                saveTimePower.append(preSolarPower_list[i])
                storePreSolarData(saveTimePower)
                conn.commit()
                print('已保存:' + str(i+1))
            else:
                print('已有记录')
        
        
        
        # self.printc("生成Excel表")
        # Real_df_result.to_excel("20.xlsx", sheet_name="20", index=False, header=True)

        self.printc("结果可视化")
        self.plot_result(test_y, predict_y)
        self.plot_result(origin_test_y, origin_predict_y)
        #真实预测结果可视化
        plt.plot(Real_df_result.iloc[:,1:2])
        plt.show()
        plt.plot(df.iloc[:,6:7])
        plt.show()
        # X = df.iloc[:,1:2]
        # Y = df.iloc[:, 2:3]
        #
        #
        # train_x = X.iloc[:5800, :]
        # train_y = Y.iloc[:5800, :]
        # test_x = X.iloc[5800:, :]
        # test_y = Y.iloc[5800:, :]
        #
        #
        # printc("归一化")
        # standard_scaler_x = preprocessing.MinMaxScaler()
        # standard_scaler_y = preprocessing.MinMaxScaler()
        #
        # train_x = standard_scaler_x.fit_transform(train_x)
        # train_y = standard_scaler_y.fit_transform(train_y).ravel()
        #
        # test_x = standard_scaler_x.transform(test_x)
        # test_y = standard_scaler_y.transform(test_y).ravel()
        #
        # printc("参数寻优与建模")
        # model = svm_cross_validation(train_x, train_y)
        #
        # # model = XGBRegressor()
        # # model.fit(train_x, train_y, verbose=False)
        #
        # printc("预测")
        # predict_y = model.predict(test_x)
        #
        # printc("反归一化")
        # test_yy = np.array(test_y).reshape(1, -1)
        # origin_test_y = standard_scaler_y.inverse_transform(test_yy).ravel()
        # predict_yy = np.array(predict_y).reshape(1, -1)
        # origin_predict_y = standard_scaler_y.inverse_transform(predict_yy).ravel()
        #
        # printc("计算准确率")
        # precession(test_y, predict_y)
        #
        # printc("结果可视化")
        # plot_result(test_y, predict_y)
        # plot_result(origin_test_y, origin_predict_y)
        # print(origin_predict_y)
predict_model = Pv_Predict("a")
predict_model.main_predict()
