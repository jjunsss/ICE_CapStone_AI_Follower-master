import glob
import shutil
import os

dir = "./dataset"
f = glob.glob(os.path.join(dir, '*.jpg'))


#d 축 생성  None으로 생성(좌 / 우 없으니까)
# for x in f:
#     last = x.split('_')[4]
#     new = x
#     new = new.replace(x.split('_')[4], "2_") #None을 의미
#     new = new + last
#     print(new)
#     os.rename(x, new)

#     print(x)
#     print(new)


#print(os.path.basename(f[0]))


for x in f:
    shutil.move(x, './combine')