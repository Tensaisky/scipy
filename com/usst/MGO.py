import random
import math
import copy

class MGO:
    def __init__(self, Ppv, price_buy, price_sell, Pload, pair=[0]*96,Pc=0.9,Pm=0.1,m=300,genaration=1000,n=96,Q=5,dt=1/4,soc1=0.41,alpha=0.0513,beta=20,maxHigh=20,BatteryPower=[0]*96,PBsoc=[0]*96):
        self.Ppv = Ppv
        self.price_buy = price_buy
        self.price_sell = price_sell
        self.Pload = Pload
        self.pair = pair
        self.Pc = Pc
        self.Pm = Pm
        self.m = m
        self.genaration = genaration
        self.n = n
        self.Q = Q
        self.dt = dt
        self.soc1 = soc1
        self.alpha = alpha
        self.beta = beta
        self.maxHigh = maxHigh
        self.BatteryPower = BatteryPower
        self.PBsoc = PBsoc

    def MGO(self):
        Ppv = list(self.Ppv)
        price_buy = list(self.price_buy)
        price_sell = list(self.price_sell)
        Pload = list(self.Pload)
        Pair = list(self.pair)

        Pb_soc = [[0 for i in range(self.n)] for j in range(self.m)]
        VaOut = [[[] for i in range(self.m)] for j in range(self.genaration)]
        RandOut = [[0 for i in range(self.n)] for j in range(self.m)]

        for i in range(self.m):
            for j in range(self.n):
                RandOut[i][j] = random.randint(-1, 1)

        for ge in range(self.genaration):
            # 交叉
            RandOut1 = copy.deepcopy(RandOut)
            for i in range(int(self.m / 2)):
                rand1 = random.randint(0, len(RandOut1) - 1)
                cross1 = RandOut1[rand1]
                RandOut1.remove(RandOut1[rand1])
                rand2 = random.randint(0, len(RandOut1) - 1)
                cross2 = RandOut1[rand2]
                RandOut1.remove(RandOut1[rand2])
                if (random.random() < self.Pc):
                    crossPoint1 = random.randint(0, self.n - 2)
                    crossPoint2 = random.randint(crossPoint1 + 1, self.n)
                    temp = cross1[crossPoint1:crossPoint2]
                    cross1[crossPoint1:crossPoint2] = cross2[crossPoint1:crossPoint2]
                    cross2[crossPoint1:crossPoint2] = temp
                RandOut[2 * i - 1] = cross1
                RandOut[2 * i] = cross2

                # 变异
            for i in range(self.m):
                temp = [-1, 0, 1]
                if (random.random() < self.Pm):
                    mPoint = random.randint(0, self.n - 1)
                    temp.remove(RandOut[i][mPoint])
                    RandOut[i][mPoint] = temp[random.randint(0, len(temp) - 1)]

                    # soc检测
            batteryPower = copy.deepcopy(RandOut)
            for i in range(self.m):
                for j in range(self.n):
                    if (batteryPower[i][j] > 0):
                        batteryPower[i][j] = random.uniform(0, 2.5)
                    elif (batteryPower[i][j] < 0):
                        batteryPower[i][j] = random.uniform(-1.5, 0)
                for t in range(self.n):
                    temp = [-1, 0, 1]
                    if (t == 0):
                        Pb_soc[i][t] = self.soc1 + (-0.211 * max(0, batteryPower[i][t]) * self.dt - 0.9 * self.dt * batteryPower[i][
                            t]) / self.Q
                    else:
                        Pb_soc[i][t] = Pb_soc[i][t - 1] + (-0.211 * max(0, batteryPower[i][t]) * self.dt - 0.9 * self.dt *
                                                           batteryPower[i][t]) / self.Q
                    if (Pb_soc[i][t] > 0.8 or Pb_soc[i][t] < 0.3):
                        temp.remove(RandOut[i][t])
                        RandOut[i][t] = temp[random.randint(0, len(temp) - 1)]
                        if (batteryPower[i][t] > 0):
                            batteryPower[i][t] = random.uniform(-1.5, 0)
                        elif (batteryPower[i][t] < 0):
                            batteryPower[i][t] = random.uniform(0, 2.5)
                    if (t == 0):
                        Pb_soc[i][t] = self.soc1 + (-0.211 * max(0, batteryPower[i][t]) * self.dt - 0.9 * self.dt * batteryPower[i][
                            t]) / self.Q
                    else:
                        Pb_soc[i][t] = Pb_soc[i][t - 1] + (-0.211 * max(0, batteryPower[i][t]) * self.dt - 0.9 * self.dt *
                                                           batteryPower[i][t]) / self.Q

            # 适应度计算，比例选择
            Tc_suit = [0 for i in range(self.m)]
            for i in range(self.m):
                Total_cost = 0
                Pb = batteryPower[i]
                for t in range(self.n):
                    Pcc = Pair[t] + Pload[t] - Ppv[t] - Pb[t]
                    Fes = self.alpha * (math.pow(max(0, Pb[t]), 2))
                    if (Pcc > 0):
                        Pcc = min(Pcc, 20)
                        Fss = Pcc * price_buy[t]
                    else:
                        Pcc = max(Pcc, -20)
                        Fss = Pcc * price_sell[t]
                    Total_cost += Fss + Fes
                SocPunish = self.beta * abs(Pb_soc[i][self.n - 1] - self.soc1)  # 添加惩罚
                Total_cost = SocPunish + Total_cost
                Tc_suit[i] = self.maxHigh - Total_cost
                VaOut[ge][i].extend(batteryPower[i])
                VaOut[ge][i].append(Tc_suit[i])

            Tc_sum = sum(Tc_suit)
            for i in range(self.m):
                if (i == 0):
                    Tc_suit[i] = Tc_suit[i] / Tc_sum
                else:
                    Tc_suit[i] = Tc_suit[i] / Tc_sum + Tc_suit[i - 1]
            RandOut_New = copy.deepcopy(RandOut)
            for i in range(self.m):
                temp = random.random()
                for j in range(self.m):
                    if (temp < Tc_suit[j]):
                        RandOut_New[i] = RandOut[j]
                        break
            RandOut = copy.deepcopy(RandOut_New)

        # 筛选
        mymax = 0
        for i in range(self.genaration):
            for j in range(self.m):
                if (VaOut[i][j][self.n] > mymax):
                    mymax = VaOut[i][j][self.n]
                    self.BatteryPower = VaOut[i][j][0:self.n]
        for i in range(self.n):
            if (i == 0):
                self.PBsoc[i] = self.soc1 + (-0.211 * max(0, self.BatteryPower[i]) * self.dt - 0.9 * self.dt * self.BatteryPower[i]) / self.Q
            else:
                self.PBsoc[i] = self.PBsoc[i - 1] + (-0.211 * max(0, self.BatteryPower[i]) * self.dt - 0.9 * self.dt * self.BatteryPower[i]) / self.Q

        Total_cost = 0
        for t in range(self.n):
            Pcc = Pair[t] + Pload[t] - Ppv[t] - self.BatteryPower[t]
            Fes = self.alpha * (math.pow(max(0, self.BatteryPower[t]), 2))
            if (Pcc > 0):
                Pcc = min(Pcc, 20)
                Fss = Pcc * price_buy[t]
            else:
                Pcc = max(Pcc, -20)
                Fss = Pcc * price_sell[t]
            Total_cost += Fss + Fes
        # print(Total_cost)
        return Total_cost

    def get_result(self):
        return self.BatteryPower,self.PBsoc