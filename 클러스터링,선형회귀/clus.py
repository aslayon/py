from sklearn.datasets import load_iris
import pandas as pd

# Load iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Initialize representative points
representsl = [5.0, 6.0, 7.0]
representsw = [3.0, 3.5, 2.5]
representpl = [1.5, 4.5, 6.0]
representpw = [0.2, 1.5, 2.0]

a = []
b = []
c = []

def makediff():
    global a, b, c
    a = []
    b = []
    c = []
    
    for i in range(150):
        tmp0 = (representsl[0] - iris_df.iloc[i, 0])**2 + \
               (representsw[0] - iris_df.iloc[i, 1])**2 + \
               (representpl[0] - iris_df.iloc[i, 2])**2 + \
               (representpw[0] - iris_df.iloc[i, 3])**2
        
        tmp1 = (representsl[1] - iris_df.iloc[i, 0])**2 + \
               (representsw[1] - iris_df.iloc[i, 1])**2 + \
               (representpl[1] - iris_df.iloc[i, 2])**2 + \
               (representpw[1] - iris_df.iloc[i, 3])**2
        
        tmp2 = (representsl[2] - iris_df.iloc[i, 0])**2 + \
               (representsw[2] - iris_df.iloc[i, 1])**2 + \
               (representpl[2] - iris_df.iloc[i, 2])**2 + \
               (representpw[2] - iris_df.iloc[i, 3])**2
        
        min_dist = min(tmp0, tmp1, tmp2)
        
        if min_dist == tmp0:
            a.append(iris_df.iloc[i])
        elif min_dist == tmp1:
            b.append(iris_df.iloc[i])
        else:
            c.append(iris_df.iloc[i])

def makeavg():
    global representsl, representsw, representpl, representpw

    avgsl = [0.0, 0.0, 0.0]
    avgsw = [0.0, 0.0, 0.0]
    avgpl = [0.0, 0.0, 0.0]
    avgpw = [0.0, 0.0, 0.0]

    for i in range(len(a)):
        avgsl[0] += a[i]['sepal length (cm)']
        avgsw[0] += a[i]['sepal width (cm)']
        avgpl[0] += a[i]['petal length (cm)']
        avgpw[0] += a[i]['petal width (cm)']
    
    for i in range(len(b)):
        avgsl[1] += b[i]['sepal length (cm)']
        avgsw[1] += b[i]['sepal width (cm)']
        avgpl[1] += b[i]['petal length (cm)']
        avgpw[1] += b[i]['petal width (cm)']
    
    for i in range(len(c)):
        avgsl[2] += c[i]['sepal length (cm)']
        avgsw[2] += c[i]['sepal width (cm)']
        avgpl[2] += c[i]['petal length (cm)']
        avgpw[2] += c[i]['petal width (cm)']

    if len(a) > 0:
        avgsl[0] /= len(a)
        avgsw[0] /= len(a)
        avgpl[0] /= len(a)
        avgpw[0] /= len(a)
    if len(b) > 0:
        avgsl[1] /= len(b)
        avgsw[1] /= len(b)
        avgpl[1] /= len(b)
        avgpw[1] /= len(b)
    if len(c) > 0:
        avgsl[2] /= len(c)
        avgsw[2] /= len(c)
        avgpl[2] /= len(c)
        avgpw[2] /= len(c)

    representsl = avgsl
    representsw = avgsw
    representpl = avgpl
    representpw = avgpw


for _ in range(150):  
    makediff()
    makeavg()



pd.set_option('display.max_rows', None)


df_a = pd.DataFrame(a)
df_b = pd.DataFrame(b)
df_c = pd.DataFrame(c)


print("a:")
print(df_a)
print(f"[{df_a.shape[0]} rows x {df_a.shape[1]} columns]")

print("b:")
print(df_b)
print(f"[{df_b.shape[0]} rows x {df_b.shape[1]} columns]")

print("c:")
print(df_c)
print(f"[{df_c.shape[0]} rows x {df_c.shape[1]} columns]")
