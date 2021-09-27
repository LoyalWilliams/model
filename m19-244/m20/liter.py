import shutil,os

#复制单个文件
# name='D:\\program\\MATLAB\\toolbox\\gatbx-toolbox\\CONTENTS.M'

def move(name):
    ln2=name.split('.')
    shutil.move("D:\\program\\MATLAB\\toolbox\\gatbx-toolbox\\gatbx\\TEST_FNS\\"+name,
    "D:\\program\\MATLAB\\toolbox\\gatbx-toolbox\\gatbx\\TEST_FNS\\"+ln2[0].lower()+'.m')


fs=os.listdir("D:\\program\\MATLAB\\toolbox\\gatbx-toolbox\\gatbx\\TEST_FNS")
for old_name in fs: 
    if old_name.endswith(".M"):
        move(old_name)
