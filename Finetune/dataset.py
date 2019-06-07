import os

os.makedirs('./train_file',exist_ok=True)
os.makedirs('./test_file',exist_ok=True)


def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')

    for _,_,img_list in os.walk(img_dir,topdown=True):
        img_list.sort(key = lambda x:int(x[6:-4]))
        for i in range(len(img_list)):
            if not img_list[i].endswith('png'):
                continue
            img_path = os.path.join(img_dir, img_list[i])
            f.write(img_path+'\n')
    f.close()


for i in range(8):
    train_txt_path = './train_file/train_'+str(i+1)+'.txt'
    train_path = 'train_frames/train_'+str(i+1)
    gen_txt(train_txt_path,train_path)

for i in range(2):
    test_txt_path = './test_file/test_'+str(i+1)+'.txt'
    test_path = 'test_frames/test_' + str(i + 1)
    gen_txt(test_txt_path, test_path)
