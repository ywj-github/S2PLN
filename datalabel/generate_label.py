import os
import json
import sys
import glob
import random

# change your data path
data_dir = '/home/ywj/paper/S2PLN/data/'

def msu_process():
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    nolabel_final_json = []
    label_save_dir = './msu/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    f_nolabel = open(label_save_dir + 'no_label.json', 'w')
    dataset_path = data_dir + 'msu_224/'

    choise_list = [i for i in range(35)]

    label_path_list = random.sample(choise_list, (len(choise_list)//3))
    label_path_list.sort()

    nolabel_path_list = list(set(choise_list) - set(label_path_list))
    nolabel_path_list.sort()

    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True)

    id_dict = {}
    m = 0
    for i in range(0, len(path_list)):
        photo_id = path_list[i].split('/')[7]
        if int(photo_id) in label_path_list:
            photo_id = int(photo_id)
            if photo_id not in id_dict:
                id_dict[photo_id] = m
                m += 1
                id_label = id_dict[photo_id]
            else:
                id_label = id_dict[photo_id]

            flag = path_list[i].find('/real')
            if (flag != -1):
                label = 1
            else:
                label = 0
            dict = {}
            dict['photo_path'] = path_list[i]
            dict['photo_label'] = label
            dict['photo_id'] = id_label

            all_final_json.append(dict)
            if(label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
        else:
            dict_no = {}
            dict_no['photo_path'] = path_list[i]
            dict_no['photo_label'] = 'no'
            dict_no['photo_id'] = 'no'

            nolabel_final_json.append(dict_no)

    print('\nMSU: ', len(path_list))
    print('MSU(label_path_list): ', len(label_path_list), label_path_list)
    print('MSU(id_dict): ', id_dict)
    print('MSU(all): ', len(all_final_json))
    print('MSU(real): ', len(real_final_json))
    print('MSU(fake): ', len(fake_final_json))
    print('MSU(nolabel_path_list): ', len(nolabel_path_list), nolabel_path_list)
    print('MSU(nolabel): ', len(nolabel_final_json))


    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()
    json.dump(nolabel_final_json, f_nolabel, indent=4)
    f_nolabel.close()


def casia_process():
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    nolabel_final_json = []
    label_save_dir = './casia/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)

    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    f_nolabel = open(label_save_dir + 'no_label.json', 'w')
    dataset_path = data_dir + 'casia_224/'

    choise_list = [i for i in range(50)]

    label_path_list = random.sample(choise_list, (len(choise_list)//3))
    label_path_list.sort()

    nolabel_path_list = list(set(choise_list) - set(label_path_list))
    nolabel_path_list.sort()

    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True)
    # /home/ywj/paper/S2PLN/data/casia_224/0/HR_1/2.jpg
    id_dict = {}
    m = 0
    for i in range(0, len(path_list), 2):
        photo_id = int(path_list[i].split('/')[7])
        if photo_id in label_path_list:
            if photo_id not in id_dict:
                id_dict[photo_id] = m
                m += 1
                id_label = id_dict[photo_id]
            else:
                id_label = id_dict[photo_id]

            flag = path_list[i].split('/')[-2]
            if (flag in ["1", "2", "HR_1"]):
                label = 1
            else:
                label = 0
            dict = {}
            dict['photo_path'] = path_list[i]
            dict['photo_label'] = label
            dict['photo_id'] = id_label

            all_final_json.append(dict)
            if(label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
        else:
            dict_no = {}
            dict_no['photo_path'] = path_list[i]
            dict_no['photo_label'] = 'no'
            dict_no['photo_id'] = 'no'

            nolabel_final_json.append(dict_no)

    print('\nCasia: ', len(path_list))
    print('Casia(label_path_list): ', len(label_path_list), label_path_list)
    print('Casia(id_dict): ', id_dict)
    print('Casia(all): ', len(all_final_json))
    print('Casia(real): ', len(real_final_json))
    print('Casia(fake): ', len(fake_final_json))
    print('Casia(nolabel_path_list): ', len(nolabel_path_list), nolabel_path_list)
    print('Casia(nolabel): ', len(nolabel_final_json))

    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()
    json.dump(nolabel_final_json, f_nolabel, indent=4)
    f_nolabel.close()

def replay_process():

    all_final_json = []
    real_final_json = []
    fake_final_json = []
    nolabel_final_json = []
    label_save_dir = './replay/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)

    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    f_nolabel = open(label_save_dir + 'no_label.json', 'w')
    dataset_path = data_dir + 'replay_224/'

    choise_list = [i for i in range(35)]

    label_path_list = random.sample(choise_list, (len(choise_list) // 3))
    label_path_list.sort()

    nolabel_path_list = list(set(choise_list) - set(label_path_list))
    nolabel_path_list.sort()

    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True)
    path_list.sort()
    # /home/ywj/paper/S2PLN/data/replay_224/0/attack/hand/attack_highdef_client115_session01_highdef_photo_adverse/1.jpg
    # /home/ywj/paper/S2PLN/data/replay_224/0/real/client115_session01_webcam_authenticate_adverse_1/1.jpg
    id_dict = {}
    m = 0
    for i in range(0, len(path_list), 2):
        photo_id = int(path_list[i].split('/')[7])
        if photo_id in label_path_list:
            if photo_id not in id_dict:
                id_dict[photo_id] = m
                m += 1
                id_label = id_dict[photo_id]
            else:
                id_label = id_dict[photo_id]

            flag = path_list[i].find('/real/')
            if (flag != -1):
                label = 1
            else:
                label = 0
            dict = {}
            dict['photo_path'] = path_list[i]
            dict['photo_label'] = label
            dict['photo_id'] = id_label

            all_final_json.append(dict)
            if(label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
        else:
            dict_no = {}
            dict_no['photo_path'] = path_list[i]
            dict_no['photo_label'] = 'no'
            dict_no['photo_id'] = 'no'

            nolabel_final_json.append(dict_no)

    print('\nReplay: ', len(path_list))
    print('Replay(label_path_list): ', len(label_path_list), label_path_list)
    print('Replay(id_dict): ', id_dict)
    print('Replay(all): ', len(all_final_json))
    print('Replay(real): ', len(real_final_json))
    print('Replay(fake): ', len(fake_final_json))
    print('Replay(nolabel_path_list): ', len(nolabel_path_list), nolabel_path_list)
    print('Replay(nolabel): ', len(nolabel_final_json))

    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()
    json.dump(nolabel_final_json, f_nolabel, indent=4)
    f_nolabel.close()

def oulu_process():

    all_final_json = []
    real_final_json = []
    fake_final_json = []
    nolabel_final_json = []
    label_save_dir = './oulu/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)

    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    f_nolabel = open(label_save_dir + 'no_label.json', 'w')

    dataset_path = data_dir + 'oulu_224/'

    choise_list = [i for i in range(40)]

    label_path_list = random.sample(choise_list, (len(choise_list) // 3))
    label_path_list.sort()

    nolabel_path_list = list(set(choise_list) - set(label_path_list))
    nolabel_path_list.sort()

    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True)
    # /home/ywj/paper/S2PLN/data/oulu_224/0/1_1_55_1/1.jpg
    path_list.sort()
    id_dict = {}
    m = 0
    for i in range(0, len(path_list), 2):
        photo_id = int(path_list[i].split('/')[7])
        if photo_id in label_path_list:
            if photo_id not in id_dict:
                id_dict[photo_id] = m
                m += 1
                id_label = id_dict[photo_id]
            else:
                id_label = id_dict[photo_id]

            label = path_list[i].split('/')[-2].split('_')[-1]
            if int(label) == 1:
                label = 1
            else:
                label = 0
            dict = {}
            dict['photo_path'] = path_list[i]
            dict['photo_label'] = label
            dict['photo_id'] = id_label

            all_final_json.append(dict)
            if(label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
        else:
            dict_no = {}
            dict_no['photo_path'] = path_list[i]
            dict_no['photo_label'] = 'no'
            dict_no['photo_id'] = 'no'

            nolabel_final_json.append(dict_no)

    print('\nOulu: ', len(path_list))
    print('Oulu(label_path_list): ', len(label_path_list), label_path_list)
    print('Oulu(id_dict): ', id_dict)
    print('Oulu(all): ', len(all_final_json))
    print('Oulu(real): ', len(real_final_json))
    print('Oulu(fake): ', len(fake_final_json))
    print('Oulu(nolabel_path_list): ', len(nolabel_path_list), nolabel_path_list)
    print('Oulu(nolabel): ', len(nolabel_final_json))

    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()
    json.dump(nolabel_final_json, f_nolabel, indent=4)
    f_nolabel.close()


if __name__=="__main__":
    msu_process()
    casia_process()
    replay_process()
    oulu_process()
