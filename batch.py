import os

#for i in range(6):
#    K = (i+1) * 27
    #os.system("python jobutil.py 2 bagging %d all"%K)
for i in range(6):
    K = (i+1) * 14
    os.system("python jobutil.py 3 GPR RQ %d all"%K)
