import torch
from torch.utils.data import DataLoader
import numpy as np
import nibabel as nib
import os
import random
import torch.backends.cudnn
import pathlib
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm, trange
# random_seed = 77
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed)
# torch.cuda.manual_seed(random_seed)
# #torch.backends.cudnn.deterministic = True
# #torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False
# np.random.seed(random_seed)
# random.seed(random_seed)
import time

from diffusion_model import DiffusionProcess
from models.unet import *
from utils import *



def run():
    test_batch_size = 100
    n_cls = 1  # number of classes to predict (background and tumor)
    in_channels = 1  # number of input modalities
    n_filters = 64  # number of filters after the input (24 was used in the paper)


    cuda_device = "cuda:0"
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")

    if device.type == 'cpu':
        print('Start training the model on CPU')
    else:
        print(f'Start training the model on {torch.cuda.get_device_name(torch.cuda.current_device())}')

    test_paths = get_paths_to_patient_files_val('C:/Dataset/test')


    test_loader = DataLoader(test_set,batch_size=test_batch_size,shuffle=False, num_workers=0)

    criterion = nn.MSELoss()
    metric = NMSE

    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(1, 2, 0)

    # model = BaselineUNet2D(in_channels=in_channels, n_cls=n_cls, n_filters=n_filters).cuda()
    model = GeneratorUNet(in_channels=7,n_cls=7,n_filters=32).cuda()
    process = DiffusionProcess()
    model.load_state_dict(torch.load("./Results(UNET_DDPM_step_500)/best_model_weights.pt"))
    model.eval()


    phase_loss = 0.0  # Train or val loss
    phase_metric = 0.0
    phase_me = 0.0

    result_txt = "./Results(UNET_DDPM_step_500)" + "/test_result.txt"
    if os.path.isfile(result_txt):
        os.unlink(result_txt)
    if not os.path.isfile(result_txt):
        f = open(result_txt, 'w')
        f.close()

    # 결과 저장할 폴더 있으면 삭제하고 생성 없으면 생성
    result_path = "./Results(UNET_DDPM_step_500)/result_image_batch_1"
    if os.path.isdir(result_path):
        os.rmdir(result_path)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    start_time = time.time()

    with torch.no_grad():
        val_bar = tqdm(test_loader, leave=False)
        for data in val_bar:
            # forward pass
            ac_path = data['ac_path']
            id = data['id']

            target = data['ac_img']
            input = data['nac_img']
            input, target = input.to(device), target.to(device)

            output = torch.randn(test_batch_size, 7, 200, 200).to(device)

            for t in trange(499, -1, -1):
                time2 = torch.ones(1) * t
                et_ = model(output.to(device), time2.to(device))
                output = process.proposed_inverse(output, input, et_, t)


            for n in range(test_batch_size):
                id_n = id[n].split('_')[-1]
                ac_path_n = ac_path[n]
                patient_name = ac_path_n.split('\\')[1]
                os.makedirs(result_path + '/' + str(patient_name), exist_ok=True)

                ac_hd = nib.load(ac_path_n)
                ac_arr = ac_hd.get_fdata()
                ac_mean = np.mean(ac_arr)
                ac_std = np.std(ac_arr)

                loss = nn.MSELoss()(output[n,:], target[n,:])

                output[n,:] = (output[n,:] * (ac_std + 1e-3))+ac_mean
                print(output[n,:].shape)
                output_np = np.squeeze(fn_tonumpy(output[n,:]))
                print(output_np.shape)

                nmse = NMSE(output_np,ac_arr)



                phase_loss += loss.item()
                phase_metric += nmse.item()



                with open(result_txt, 'a') as f:
                    data = f'loss: \t{loss:.6f} \tnmse: \t{nmse:.6f} \n'
                    f.write(data)


                proxy = nib.load(ac_path_n)
                img_out = nib.Nifti1Image(output_np,proxy.affine,proxy.header)
                img_out.to_filename(os.path.join(result_path+'/'+str(patient_name),'Out_'+str(id_n)+'.nii.gz'))

    phase_loss /= len(test_loader)
    phase_metric /= len(test_loader)


    with open(result_txt, 'a') as f:
        data = f'Test loss: \t{phase_loss:.6f} \tnmse: \t{phase_metric:.6f} \n'
        f.write(data)

if __name__ =='__main__':

    run()