import torch
from data import valid_dataloader
from utils import Adder
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np


def _valid(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset = valid_dataloader(args.test_dir, num_workers=0, is_test=False)

    model.eval()
    psnr_adder = Adder()
    ssim_adder = Adder()

    with torch.no_grad():
        print('Start Evaluation')

        for idx, data in enumerate(val_dataset):
            input_img, label_img = data
            input_img = input_img.to(device)

            pred = model(input_img)
            pred = pred.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            pred_numpy = np.clip(pred * 255.0, 0, 255.0).astype('uint8')

            label = label_img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            label_numpy = np.clip(label * 255.0, 0, 255.0).astype('uint8')

            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=255)
            ssim = structural_similarity(pred_numpy, label_numpy, data_range=255, multichannel=True)

            psnr_adder(psnr)
            ssim_adder(ssim)
            print('\r%03d' % idx, end=' ')

    print('\n')
    model.train()
    return psnr_adder.average(), ssim_adder.average()
