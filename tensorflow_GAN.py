import pandas as pd

def learn_pca(df):
    from sklearn.decomposition import PCA   
    pca = PCA(n_components = 2)
    pca.fit(df)
    df_pca = pca.transform(df)
    df_pca = pd.DataFrame(df_pca, columns = ['pca_1', 'pca_2'])
    df_pca = df_pca.astype(float)

    return df_pca


df = pd.read_csv("/Users/ytgw/Desktop/dog_data")


pet_age = df['pet_age']
pet_age_anl_r = []

pet_weight = df['pet_weight']
pet_weight_anl_r = []




pet_age_anl = pet_age.value_counts()
pet_age_anl = pd.DataFrame(pet_age_anl)

pet_weight_anl = pet_weight.value_counts()
pet_weight_anl = pd.DataFrame(pet_weight)

for i in range(len(pet_age_anl)):
    if i in pet_age_anl.index.tolist():
        o = pet_age_anl.loc[i]
        pet_age_anl_r.append(o[0])
    if i in pet_weight_anl.index.tolist():
        o = pet_weight_anl.loc[i]
        pet_weight_anl_r.append(o[0])
        
        
df_dog = df[['pet_age','pet_weight','sex']]
df_dog = df_dog.dropna()
df_dog.loc[df_dog['sex'] == '남','sex'] = 1
df_dog.loc[df_dog['sex'] == '여','sex'] = 0
df_dog['sex'].value_counts()[0]



import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.bar(range(len(pet_age_anl_r)),pet_age_anl_r)

plt.figure(figsize=(6,4))
plt.bar(range(len(pet_weight_anl_r)),pet_weight_anl_r)

plt.figure(figsize=(6,4))
plt.bar(range(2),[df_dog['sex'].value_counts()[0],df_dog['sex'].value_counts()[1]])


pca_dog = learn_pca(df_dog)

plt.figure(figsize=(6,4))
plt.scatter(pca_dog['pca_1'],pca_dog['pca_2'])

for i in range(len(pca_dog['pca_1'])):
    if pca_dog['pca_1'][i]>50:
        pca_dog = pca_dog.drop(i)
        
plt.figure(figsize=(6,4))
plt.scatter(pca_dog['pca_1'],pca_dog['pca_2'])