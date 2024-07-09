from sklearn.datasets import load_iris
import pandas as pd
import math

#속성 하나만 거리 비교, 가장 가까운 것 찾기. (속성 하나씩 4번)
#속성 1,2 만 가지고./ 1,3 가지고.
#속성 1,2,3,4 전부 

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

print(iris_df.head())

iris_df['target'] = iris.target

print(iris_df.head())

print(iris_df.tail())

target_num = [0,0,0]
sepall_sum = [0.0, 0.0, 0.0]
sepalw_sum = [0.0, 0.0, 0.0]
petall_sum = [0.0, 0.0, 0.0]
petalw_sum = [0.0, 0.0, 0.0]
for i in range(30):
    if(iris_df['target'][i] == 0):
        target_num[0] += 1
        sepall_sum[0] += iris_df['sepal length (cm)'][i]
        sepalw_sum[0] += iris_df['sepal width (cm)'][i]
        petall_sum[0] += iris_df['petal length (cm)'][i]
        petalw_sum[0] += iris_df['petal width (cm)'][i]
    if(iris_df['target'][i] == 1):
        target_num[1] += 1
        sepall_sum[1] += iris_df['sepal length (cm)'][i]
        sepalw_sum[1] += iris_df['sepal width (cm)'][i]
        petall_sum[1] += iris_df['petal length (cm)'][i]
        petalw_sum[1] += iris_df['petal width (cm)'][i]
    if(iris_df['target'][i] == 2):
        target_num[2] += 1
        sepall_sum[2] += iris_df['sepal length (cm)'][i]
        sepalw_sum[2] += iris_df['sepal width (cm)'][i]
        petall_sum[2] += iris_df['petal length (cm)'][i]
        petalw_sum[2] += iris_df['petal width (cm)'][i]

for i in range(50,80):
    if(iris_df['target'][i] == 0):
        target_num[0] += 1
        sepall_sum[0] += iris_df['sepal length (cm)'][i]
        sepalw_sum[0] += iris_df['sepal width (cm)'][i]
        petall_sum[0] += iris_df['petal length (cm)'][i]
        petalw_sum[0] += iris_df['petal width (cm)'][i]
    if(iris_df['target'][i] == 1):
        target_num[1] += 1
        sepall_sum[1] += iris_df['sepal length (cm)'][i]
        sepalw_sum[1] += iris_df['sepal width (cm)'][i]
        petall_sum[1] += iris_df['petal length (cm)'][i]
        petalw_sum[1] += iris_df['petal width (cm)'][i]
    if(iris_df['target'][i] == 2):
        target_num[2] += 1
        sepall_sum[2] += iris_df['sepal length (cm)'][i]
        sepalw_sum[2] += iris_df['sepal width (cm)'][i]
        petall_sum[2] += iris_df['petal length (cm)'][i]
        petalw_sum[2] += iris_df['petal width (cm)'][i]

for i in range(100,130):
    if(iris_df['target'][i] == 0):
        target_num[0] += 1
        sepall_sum[0] += iris_df['sepal length (cm)'][i]
        sepalw_sum[0] += iris_df['sepal width (cm)'][i]
        petall_sum[0] += iris_df['petal length (cm)'][i]
        petalw_sum[0] += iris_df['petal width (cm)'][i]
    if(iris_df['target'][i] == 1):
        target_num[1] += 1
        sepall_sum[1] += iris_df['sepal length (cm)'][i]
        sepalw_sum[1] += iris_df['sepal width (cm)'][i]
        petall_sum[1] += iris_df['petal length (cm)'][i]
        petalw_sum[1] += iris_df['petal width (cm)'][i]
    if(iris_df['target'][i] == 2):
        target_num[2] += 1
        sepall_sum[2] += iris_df['sepal length (cm)'][i]
        sepalw_sum[2] += iris_df['sepal width (cm)'][i]
        petall_sum[2] += iris_df['petal length (cm)'][i]
        petalw_sum[2] += iris_df['petal width (cm)'][i]

sepall_sum = [round(num, 2) for num in sepall_sum]
sepalw_sum = [round(num, 2) for num in sepalw_sum]
petall_sum = [round(num, 2) for num in petall_sum]
petalw_sum = [round(num, 2) for num in petalw_sum]
#print(iris_df.to_string())
print("target num:", target_num)
print("Sepal Length Sum:", sepall_sum)
print("Sepal Width Sum:", sepalw_sum)
print("Petal Length Sum:", petall_sum)
print("Petal Width Sum:", petalw_sum)

sepall_avg = [sepall_sum[0]/target_num[0], sepall_sum[1]/target_num[1], sepall_sum[2]/target_num[2]]
sepalw_avg = [sepalw_sum[0]/target_num[0], sepalw_sum[1]/target_num[1], sepalw_sum[2]/target_num[2]]
petall_avg = [petall_sum[0]/target_num[0], petall_sum[1]/target_num[1], petall_sum[2]/target_num[2]]
petalw_avg = [petalw_sum[0]/target_num[0], petalw_sum[1]/target_num[1], petalw_sum[2]/target_num[2]]


sepall_avg = [round(num, 2) for num in sepall_avg]
sepalw_avg = [round(num, 2) for num in sepalw_avg]
petall_avg = [round(num, 2) for num in petall_avg]
petalw_avg = [round(num, 2) for num in petalw_avg]

print("Sepal Length avg:", sepall_avg)
print("Sepal Width avg:", sepalw_avg)
print("Petal Length avg:", petall_avg)
print("Petal Width avg:", petalw_avg)

sepall_dis = [0.0, 0.0, 0.0]
sepalw_dis = [0.0, 0.0, 0.0]
petall_dis = [0.0, 0.0, 0.0]
petalw_dis = [0.0, 0.0, 0.0]


for i in range(30):
    if iris_df['target'][i] == 0:
        sepall_dis[0] += (iris_df['sepal length (cm)'][i] - sepall_avg[0]) ** 2
        sepalw_dis[0] += (iris_df['sepal width (cm)'][i] - sepalw_avg[0]) ** 2
        petall_dis[0] += (iris_df['petal length (cm)'][i] - petall_avg[0]) ** 2
        petalw_dis[0] += (iris_df['petal width (cm)'][i] - petalw_avg[0]) ** 2
    elif iris_df['target'][i] == 1:
        sepall_dis[1] += (iris_df['sepal length (cm)'][i] - sepall_avg[1]) ** 2
        sepalw_dis[1] += (iris_df['sepal width (cm)'][i] - sepalw_avg[1]) ** 2
        petall_dis[1] += (iris_df['petal length (cm)'][i] - petall_avg[1]) ** 2
        petalw_dis[1] += (iris_df['petal width (cm)'][i] - petalw_avg[1]) ** 2
    elif iris_df['target'][i] == 2:
        sepall_dis[2] += (iris_df['sepal length (cm)'][i] - sepall_avg[2]) ** 2
        sepalw_dis[2] += (iris_df['sepal width (cm)'][i] - sepalw_avg[2]) ** 2
        petall_dis[2] += (iris_df['petal length (cm)'][i] - petall_avg[2]) ** 2
        petalw_dis[2] += (iris_df['petal width (cm)'][i] - petalw_avg[2]) ** 2

for i in range(50,80):
    if iris_df['target'][i] == 0:
        sepall_dis[0] += (iris_df['sepal length (cm)'][i] - sepall_avg[0]) ** 2
        sepalw_dis[0] += (iris_df['sepal width (cm)'][i] - sepalw_avg[0]) ** 2
        petall_dis[0] += (iris_df['petal length (cm)'][i] - petall_avg[0]) ** 2
        petalw_dis[0] += (iris_df['petal width (cm)'][i] - petalw_avg[0]) ** 2
    elif iris_df['target'][i] == 1:
        sepall_dis[1] += (iris_df['sepal length (cm)'][i] - sepall_avg[1]) ** 2
        sepalw_dis[1] += (iris_df['sepal width (cm)'][i] - sepalw_avg[1]) ** 2
        petall_dis[1] += (iris_df['petal length (cm)'][i] - petall_avg[1]) ** 2
        petalw_dis[1] += (iris_df['petal width (cm)'][i] - petalw_avg[1]) ** 2
    elif iris_df['target'][i] == 2:
        sepall_dis[2] += (iris_df['sepal length (cm)'][i] - sepall_avg[2]) ** 2
        sepalw_dis[2] += (iris_df['sepal width (cm)'][i] - sepalw_avg[2]) ** 2
        petall_dis[2] += (iris_df['petal length (cm)'][i] - petall_avg[2]) ** 2
        petalw_dis[2] += (iris_df['petal width (cm)'][i] - petalw_avg[2]) ** 2

for i in range(100,130):
    if iris_df['target'][i] == 0:
        sepall_dis[0] += (iris_df['sepal length (cm)'][i] - sepall_avg[0]) ** 2
        sepalw_dis[0] += (iris_df['sepal width (cm)'][i] - sepalw_avg[0]) ** 2
        petall_dis[0] += (iris_df['petal length (cm)'][i] - petall_avg[0]) ** 2
        petalw_dis[0] += (iris_df['petal width (cm)'][i] - petalw_avg[0]) ** 2
    elif iris_df['target'][i] == 1:
        sepall_dis[1] += (iris_df['sepal length (cm)'][i] - sepall_avg[1]) ** 2
        sepalw_dis[1] += (iris_df['sepal width (cm)'][i] - sepalw_avg[1]) ** 2
        petall_dis[1] += (iris_df['petal length (cm)'][i] - petall_avg[1]) ** 2
        petalw_dis[1] += (iris_df['petal width (cm)'][i] - petalw_avg[1]) ** 2
    elif iris_df['target'][i] == 2:
        sepall_dis[2] += (iris_df['sepal length (cm)'][i] - sepall_avg[2]) ** 2
        sepalw_dis[2] += (iris_df['sepal width (cm)'][i] - sepalw_avg[2]) ** 2
        petall_dis[2] += (iris_df['petal length (cm)'][i] - petall_avg[2]) ** 2
        petalw_dis[2] += (iris_df['petal width (cm)'][i] - petalw_avg[2]) ** 2

sepall_dis = [round(num, 2) for num in sepall_dis]
sepalw_dis = [round(num, 2) for num in sepalw_dis]
petall_dis = [round(num, 2) for num in petall_dis]
petalw_dis = [round(num, 2) for num in petalw_dis]

# 표본 분산 계산
sepall_dis = [sepall_dis[i]/target_num[i] for i in range(3)]
sepalw_dis = [sepalw_dis[i]/target_num[i] for i in range(3)]
petall_dis = [petall_dis[i]/target_num[i] for i in range(3)]
petalw_dis = [petalw_dis[i]/target_num[i] for i in range(3)]

sepall_dis = [round(num, 2) for num in sepall_dis]
sepalw_dis = [round(num, 2) for num in sepalw_dis]
petall_dis = [round(num, 2) for num in petall_dis]
petalw_dis = [round(num, 2) for num in petalw_dis]

print("Sepal Length Dispersion:", sepall_dis)
print("Sepal Width Dispersion:", sepalw_dis)
print("Petal Length Dispersion:", petall_dis)
print("Petal Width Dispersion:", petalw_dis)

sepall_std = [math.sqrt(num) for num in sepall_dis]
sepalw_std = [math.sqrt(num) for num in sepalw_dis]
petall_std = [math.sqrt(num) for num in petall_dis]
petalw_std = [math.sqrt(num) for num in petalw_dis]

print("Sepal Length  Deviation:", [round(num, 2) for num in sepall_std])
print("Sepal Width  Deviation:", [round(num, 2) for num in sepalw_std])
print("Petal Length  Deviation:", [round(num, 2) for num in petall_std])
print("Petal Width  Deviation:", [round(num, 2) for num in petalw_std])

a1 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]
b1 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]
c1 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]

for i in range(30,50):
       
        tmp0 = (iris_df['sepal length (cm)'][i] - sepall_avg[0]) ** 2
        tmp1 = (iris_df['sepal length (cm)'][i] - sepall_avg[1]) ** 2
        tmp2 = (iris_df['sepal length (cm)'][i] - sepall_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        a1[i-30] = tmp
        
        
        
    
for i in range(80,100):
       
        tmp0 = (iris_df['sepal length (cm)'][i] - sepall_avg[0]) ** 2
        tmp1 = (iris_df['sepal length (cm)'][i] - sepall_avg[1]) ** 2
        tmp2 = (iris_df['sepal length (cm)'][i] - sepall_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        b1[i-80] = tmp

for i in range(130,150):
       
        tmp0 = (iris_df['sepal length (cm)'][i] - sepall_avg[0]) ** 2
        tmp1 = (iris_df['sepal length (cm)'][i] - sepall_avg[1]) ** 2
        tmp2 = (iris_df['sepal length (cm)'][i] - sepall_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        c1[i-130] = tmp
print("1만\n")
print(a1)
print(b1)
print(c1)


a2 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]
b2 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]
c2 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]

for i in range(30,50):
       
        tmp0 = (iris_df['sepal width (cm)'][i] - sepalw_avg[0]) ** 2
        tmp1 = (iris_df['sepal width (cm)'][i] - sepalw_avg[1]) ** 2
        tmp2 = (iris_df['sepal width (cm)'][i] - sepalw_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        a2[i-30] = tmp
        
        
        
    
for i in range(80,100):
       
        tmp0 = (iris_df['sepal width (cm)'][i] - sepalw_avg[0]) ** 2
        tmp1 = (iris_df['sepal width (cm)'][i] - sepalw_avg[1]) ** 2
        tmp2 = (iris_df['sepal width (cm)'][i] - sepalw_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        b2[i-80] = tmp

for i in range(130,150):
       
        tmp0 = (iris_df['sepal width (cm)'][i] - sepalw_avg[0]) ** 2
        tmp1 = (iris_df['sepal width (cm)'][i] - sepalw_avg[1]) ** 2
        tmp2 = (iris_df['sepal width (cm)'][i] - sepalw_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        c2[i-130] = tmp
print("2만\n")
print(a2)
print(b2)
print(c2)
a3 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]
b3 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]
c3 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]

for i in range(30,50):
       
        tmp0 = (iris_df['petal length (cm)'][i] - petall_avg[0]) ** 2
        tmp1 = (iris_df['petal length (cm)'][i] - petall_avg[1]) ** 2
        tmp2 = (iris_df['petal length (cm)'][i] - petall_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        a3[i-30] = tmp
        
        
        
    
for i in range(80,100):
       
        tmp0 = (iris_df['petal length (cm)'][i] - petall_avg[0]) ** 2
        tmp1 = (iris_df['petal length (cm)'][i] - petall_avg[1]) ** 2
        tmp2 = (iris_df['petal length (cm)'][i] - petall_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        b3[i-80] = tmp

for i in range(130,150):
       
        tmp0 = (iris_df['petal length (cm)'][i] - petall_avg[0]) ** 2
        tmp1 = (iris_df['petal length (cm)'][i] - petall_avg[1]) ** 2
        tmp2 = (iris_df['petal length (cm)'][i] - petall_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        c3[i-130] = tmp
print("3만\n")
print(a3)
print(b3)
print(c3)
a4 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]
b4 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]
c4 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]

for i in range(30,50):
       
        tmp0 = (iris_df['petal width (cm)'][i] - petalw_avg[0]) ** 2
        tmp1 = (iris_df['petal width (cm)'][i] - petalw_avg[1]) ** 2
        tmp2 = (iris_df['petal width (cm)'][i] - petalw_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        a4[i-30] = tmp
        
        
        
    
for i in range(80,100):
       
        tmp0 = (iris_df['petal width (cm)'][i] - petalw_avg[0]) ** 2
        tmp1 = (iris_df['petal width (cm)'][i] - petalw_avg[1]) ** 2
        tmp2 = (iris_df['petal width (cm)'][i] - petalw_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        b4[i-80] = tmp

for i in range(130,150):
       
        tmp0 = (iris_df['petal width (cm)'][i] - petalw_avg[0]) ** 2
        tmp1 = (iris_df['petal width (cm)'][i] - petalw_avg[1]) ** 2
        tmp2 = (iris_df['petal width (cm)'][i] - petalw_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        c4[i-130] = tmp
print("4만\n")
print(a4)
print(b4)
print(c4)


aa1 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]
bb1 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]
cc1 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]

for i in range(30,50):
       
        tmp0 = (iris_df['sepal length (cm)'][i] - sepall_avg[0]) ** 2 + (iris_df['sepal width (cm)'][i] - sepalw_avg[0]) ** 2
        tmp1 = (iris_df['sepal length (cm)'][i] - sepall_avg[1]) ** 2 + (iris_df['sepal width (cm)'][i] - sepalw_avg[1]) ** 2
        tmp2 = (iris_df['sepal length (cm)'][i] - sepall_avg[2]) ** 2 + (iris_df['sepal width (cm)'][i] - sepalw_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        aa1[i-30] = tmp
        
        
        
    
for i in range(80,100):
       
        tmp0 = (iris_df['sepal length (cm)'][i] - sepall_avg[0]) ** 2 + (iris_df['sepal width (cm)'][i] - sepalw_avg[0]) ** 2
        tmp1 = (iris_df['sepal length (cm)'][i] - sepall_avg[1]) ** 2 + (iris_df['sepal width (cm)'][i] - sepalw_avg[1]) ** 2
        tmp2 = (iris_df['sepal length (cm)'][i] - sepall_avg[2]) ** 2 + (iris_df['sepal width (cm)'][i] - sepalw_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        bb1[i-80] = tmp

for i in range(130,150):
       
        tmp0 = (iris_df['sepal length (cm)'][i] - sepall_avg[0]) ** 2 + (iris_df['sepal width (cm)'][i] - sepalw_avg[0]) ** 2
        tmp1 = (iris_df['sepal length (cm)'][i] - sepall_avg[1]) ** 2 + (iris_df['sepal width (cm)'][i] - sepalw_avg[1]) ** 2
        tmp2 = (iris_df['sepal length (cm)'][i] - sepall_avg[2]) ** 2 + (iris_df['sepal width (cm)'][i] - sepalw_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        cc1[i-130] = tmp
print("1,2 비교\n")
print(aa1)
print(bb1)
print(cc1)

aa2 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]
bb2 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]
cc2 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]

for i in range(30,50):
       
        tmp0 = (iris_df['petal length (cm)'][i] - petall_avg[0]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[0]) ** 2
        tmp1 = (iris_df['petal length (cm)'][i] - petall_avg[1]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[1]) ** 2
        tmp2 = (iris_df['petal length (cm)'][i] - petall_avg[2]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        aa2[i-30] = tmp
        
        
        
    
for i in range(80,100):
       
        tmp0 = (iris_df['petal length (cm)'][i] - petall_avg[0]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[0]) ** 2
        tmp1 = (iris_df['petal length (cm)'][i] - petall_avg[1]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[1]) ** 2
        tmp2 = (iris_df['petal length (cm)'][i] - petall_avg[2]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        bb2[i-80] = tmp

for i in range(130,150):
       
        tmp0 = (iris_df['petal length (cm)'][i] - petall_avg[0]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[0]) ** 2
        tmp1 = (iris_df['petal length (cm)'][i] - petall_avg[1]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[1]) ** 2
        tmp2 = (iris_df['petal length (cm)'][i] - petall_avg[2]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        cc2[i-130] = tmp


print("1,3 비교\n")
print(aa2)
print(bb2)
print(cc2)



aaaa2 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]
bbbb2 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]
cccc2 = [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1]

for i in range(30,50):
       
        tmp0 = (iris_df['sepal width (cm)'][i] - sepalw_avg[0]) ** 2 +(iris_df['petal length (cm)'][i] - petall_avg[0]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[0]) ** 2 + (iris_df['petal width (cm)'][i] - petalw_avg[0]) ** 2
        tmp1 = (iris_df['sepal width (cm)'][i] - sepalw_avg[1]) ** 2 +(iris_df['petal length (cm)'][i] - petall_avg[1]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[1]) ** 2 + (iris_df['petal width (cm)'][i] - petalw_avg[1]) ** 2
        tmp2 = (iris_df['sepal width (cm)'][i] - sepalw_avg[2]) ** 2 +(iris_df['petal length (cm)'][i] - petall_avg[2]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[2]) ** 2 + (iris_df['petal width (cm)'][i] - petalw_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        aaaa2[i-30] = tmp
        
        
        
    
for i in range(80,100):
       
        tmp0 = (iris_df['sepal width (cm)'][i] - sepalw_avg[0]) ** 2 +(iris_df['petal length (cm)'][i] - petall_avg[0]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[0]) ** 2 + (iris_df['petal width (cm)'][i] - petalw_avg[0]) ** 2
        tmp1 = (iris_df['sepal width (cm)'][i] - sepalw_avg[1]) ** 2 +(iris_df['petal length (cm)'][i] - petall_avg[1]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[1]) ** 2 + (iris_df['petal width (cm)'][i] - petalw_avg[1]) ** 2
        tmp2 = (iris_df['sepal width (cm)'][i] - sepalw_avg[2]) ** 2 +(iris_df['petal length (cm)'][i] - petall_avg[2]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[2]) ** 2 + (iris_df['petal width (cm)'][i] - petalw_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        bbbb2[i-80] = tmp

for i in range(130,150):
       
        tmp0 = (iris_df['sepal width (cm)'][i] - sepalw_avg[0]) ** 2 +(iris_df['petal length (cm)'][i] - petall_avg[0]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[0]) ** 2 + (iris_df['petal width (cm)'][i] - petalw_avg[0]) ** 2
        tmp1 = (iris_df['sepal width (cm)'][i] - sepalw_avg[1]) ** 2 +(iris_df['petal length (cm)'][i] - petall_avg[1]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[1]) ** 2 + (iris_df['petal width (cm)'][i] - petalw_avg[1]) ** 2
        tmp2 = (iris_df['sepal width (cm)'][i] - sepalw_avg[2]) ** 2 +(iris_df['petal length (cm)'][i] - petall_avg[2]) ** 2 + (iris_df['sepal length (cm)'][i] - sepall_avg[2]) ** 2 + (iris_df['petal width (cm)'][i] - petalw_avg[2]) ** 2
        min = tmp0
        tmp = 0
        if(min > tmp1):
            min = tmp1
            tmp = 1
        if(min > tmp2):
            min = tmp2
            tmp = 2
        cccc2[i-130] = tmp
print("1234\n")
print(aaaa2)
print(bbbb2)
print(cccc2)