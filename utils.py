import os
import numpy as np



def inference_schedule():
    training_noise_schedule = np.array(np.linspace(1e-4, 0.01, 200).tolist())
    inference_noise_schedule = np.array(np.linspace(1e-4, 0.01, 200).tolist())
    # inference_noise_schedule2 = np.array(np.linspace(1e-4, 0.01, 100).tolist())
    # print(inference_noise_schedule)
    # print(inference_noise_schedule2)

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    # print("beta",beta)
    # print("alpha_cum",talpha_cum)
    # print("gamma_cum",alpha_cum)
    sigmas = [0 for i in alpha]
    for n in range(len(alpha) - 1, -1, -1):
        sigmas[n] = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n])

    T = []
    for s in range(len(inference_noise_schedule)):
        for t in range(len(training_noise_schedule) - 1):
            if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                            talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
                T.append(t + twiddle)
                break

    T = np.array(T, dtype=np.float32)

    m = [0 for i in alpha]
    gamma = [0 for i in alpha]
    delta = [0 for i in alpha]
    d_x = [0 for i in alpha]
    d_y = [0 for i in alpha]
    delta_cond = [0 for i in alpha]
    delta_bar = [0 for i in alpha]
    c1 = [0 for i in alpha]
    c2 = [0 for i in alpha]
    c3 = [0 for i in alpha]
    oc1 = [0 for i in alpha]
    oc3 = [0 for i in alpha]

    for n in range(len(alpha)):
        m[n] = ((1 - alpha_cum[n]) / (alpha_cum[n] ** 0.5)) ** 0.5
    m[-1] = 1

    for n in range(len(alpha)):
        # print(1 - (1 + m[n] ** 2) * alpha_cum[n])
        delta[n] = max(1 - (1 + m[n] ** 2) * alpha_cum[n], 0)
        if delta[n] == 0:
            delta[n] = 1e-6
        gamma[n] = sigmas[n]

    for n in range(len(alpha)):
        if n > 0:
            delta_cond[n] = delta[n] - (((1 - m[n]) / (1 - m[n - 1]))) ** 2 * alpha[n] * delta[n - 1]
            delta_bar[n] = (delta_cond[n]) * delta[n - 1] / delta[n]
        else:

            delta_cond[n] = 0
            delta_bar[n] = 0

    for n in range(len(alpha)):
        oc1[n] = 1 / alpha[n] ** 0.5
        oc3[n] = oc1[n] * beta[n] / (1 - alpha_cum[n]) ** 0.5
        if n > 0:
            c1[n] = (1 - m[n]) / (1 - m[n - 1]) * (delta[n - 1] / delta[n]) * alpha[n] ** 0.5 + (1 - m[n - 1]) * (
                        delta_cond[n] / delta[n]) / alpha[n] ** 0.5
            c2[n] = (m[n - 1] * delta[n] - (m[n] * (1 - m[n])) / (1 - m[n - 1]) * alpha[n] * delta[n - 1]) * (
                        alpha_cum[n - 1] ** 0.5 / delta[n])
            c3[n] = (1 - m[n - 1]) * (delta_cond[n] / delta[n]) * (1 - alpha_cum[n]) ** 0.5 / (alpha[n]) ** 0.5
        else:
            c1[n] = 1 / alpha[n] ** 0.5
            c3[n] = c1[n] * beta[n] / (1 - alpha_cum[n]) ** 0.5

    return alpha, beta, alpha_cum, sigmas, T, c1, c2, c3, delta, delta_bar

# 파일 (ct경로,pt경로,gtvt경로)의 튜플로 리스트에 append
def get_paths_to_patient_files(ac_path_to_imgs,nac_path_to_imgs, append_mask=True):
    ac_patients = os.listdir(ac_path_to_imgs)
    nac_patients = os.listdir(nac_path_to_imgs)
    patient_num = int(len(ac_patients))
    paths = []

    for i in range(patient_num):
        ac_path = ac_path_to_imgs + '/' + str(ac_patients[i])
        nac_path = nac_path_to_imgs + '/' + str(nac_patients[i])
        paths.append((ac_path,nac_path))

    return paths

def get_paths_to_patient_files_val(train_path_to_imgs):

    patients = os.listdir(train_path_to_imgs)
    paths = []

    for patient in patients:
        ac_path = os.path.join(train_path_to_imgs, patient, '2.5D/AC_PET')
        nac_path = os.path.join(train_path_to_imgs, patient, '2.5D/NAC_PET')

        file_list = os.listdir(ac_path)
        for i in file_list:
            num = i.split('.')[0].split('_')[-1]
            patient_ac_path = os.path.join(ac_path,'ac_'+str(num)+'.nii.gz')
            patient_nac_path = os.path.join(nac_path,'nac_'+str(num)+'.nii.gz')

            paths.append((patient_nac_path,patient_ac_path))

    return paths

def NMSE(output, target):
    out_np = output#.cpu().numpy()
    tar_np = target#.cpu().numpy()

    numer = np.sum((tar_np - out_np) ** 2)
    denom = np.sum((tar_np) ** 2)

    nmse = numer / denom

    return nmse

def ME(output, target):
    out_np = output.cpu().numpy()
    tar_np = target.cpu().numpy()

    numer = np.sum(tar_np - out_np)
    denom = np.sum(tar_np)

    nmse = numer / denom

    return nmse
