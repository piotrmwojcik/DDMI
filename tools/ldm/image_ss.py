import os
import torch
import torchvision
import numpy as np
import copy
from tqdm import tqdm
from torchvision import utils as vtils
import torchvision.transforms.functional as trans_F
from timeit import default_timer as timer
from accelerate import Accelerator
from ema_pytorch import EMA

from models.siren import Siren, get_mgrid, generate_mlp_from_weights
from utils.general_utils import symmetrize_image_data, unsymmetrize_image_data, exists, convert_to_coord_format_2d, get_scale_injection
from evals.eval import test_fid_ddpm, test_fid_ddpm_N

# Trainer class
class LDMSSTrainer(object):
    def __init__(
            self,
            args,
            diffusionmodel,
            diffusion_process,
            data,
            test_data=None,
            ):
        super().__init__()

        ## Accelerator
        self.accelerator = Accelerator(
                split_batches = False,
                mixed_precision = 'fp16' if args.use_fp16 else 'no'
                )
        self.accelerator.native_amp = args.amp
        
        self.data = data
        self.test_data = test_data
        self.args = args

        # Models
        self.diffusionmodel = diffusionmodel

        # Diffusion process
        self.diffusion_process = diffusion_process
        
        self.epochs = args.loss_config.epochs
        self.save_and_sample_every = args.loss_config.save_and_sample_every
        self.latent_dim = args.ddpmconfig.channels
        self.image_size = args.ddpmconfig.image_size
        self.test_batch_size = args.data_config.test_batch_size
        self.channels = args.embed_dim
        self.num_total_iters = len(data) * self.epochs
        self.gradient_accumulate_every = args.loss_config.gradient_accumulate_every
        self.test_resolution = args.data_config.test_resolution

        # Optimizers
        self.dae_opt = torch.optim.AdamW(diffusion_process.parameters(), lr = args.lr, weight_decay=0.)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_process, beta = args.loss_config.ema_decay, update_every = args.loss_config.ema_update_every)
            self.ema.to(self.accelerator.device)
        # Reset epochs and iters
        self.step = 0
        self.current_iters = 0

        if args.resume:
            print('Loading Models from previous training!')
            self.load(os.path.join(args.data_config.save_pth, 'ldm-last.pt'))
            print('Current Epochs :', self.step)
            print('Current iters :', self.current_iters)
        #elif args.pretrained:
        #    print('Loading Pretrained Models!')
        #    data_pth = torch.load(os.path.join(args.data_config.save_pth, 'ldm-last.pt'), map_location='cpu')
        #    self.diffusion_process.load_state_dict(data_pth['diffusion'])
        #    if self.accelerator.is_main_process:
        #        self.ema.load_state_dict(data_pth['ema'])
        #else:
        #    # Load from checkpoint
        #    print('Load VAE checkpoints!')
        #    data_pth = torch.load(os.path.join(args.data_config.save_pth, 'model-last.pt'), map_location='cpu')

        #config = {'ldm': self.ema.ema_model, 'vae': self.vaemodel, 'mlp': self.mlp}
        #ddmi = DDMI(self.ema.ema_model, self.vaemodel, self.mlp)
        #ddmi.push_to_hub('ddmi_afhqcat_ema')
        #ddmi.from_pretrained('DogyunPark/ddmi_afhqcat_ema')
        #import pdb; pdb.set_trace()
        # Wrap with accelerator
        self.data, self.diffusion_process, self.dae_opt = self.accelerator.prepare(self.data, self.diffusion_process, self.dae_opt)

        ## Save directory
        self.results_folder = args.data_config.save_pth
        os.makedirs(self.results_folder, exist_ok=True)
        self.results_pth = os.path.join(self.results_folder, 'results')
        os.makedirs(self.results_pth, exist_ok=True)
       
    def save(self, step = 0):
        if not self.accelerator.is_local_main_process:
            return
        data = {
                'args' : self.args,
                'step' : self.step,
                'current_iters' : self.current_iters,
                'diffusion' : self.accelerator.get_state_dict(self.diffusion_process),
                'dae_opt' : self.dae_opt.state_dict(),
                'ema' : self.ema.state_dict(),
                'scaler' : self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
                }
        torch.save(data, os.path.join(self.results_folder, 'ldm-{}.pt'.format(step)))
        torch.save(data, os.path.join(self.results_folder, 'ldm-last.pt'.format(step)))


    def load(self, pth):
        data = torch.load(pth, map_location= 'cpu')
        self.diffusion_process.load_state_dict(data['diffusion'])
        self.step = data['step']
        self.current_iters = data['current_iters']
        self.dae_opt.load_state_dict(data['dae_opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        device = self.accelerator.device
        self.diffusion_process.train()
        shape = (self.test_batch_size, 50307)
        noise_fix = torch.randn((self.test_batch_size, 50307), device = device)

        cache = {}

        with tqdm(initial = self.step, total = self.epochs) as pbar:
            while self.step < self.epochs:
                for idx, (x, fn) in enumerate(self.data):
                    #x = symmetrize_image_data(x)
                    #y = trans_F.resize(x, 256, antialias = True)
                    #y = y.clamp(-1., 1.)
                    #b, c, h, w = x.shape
                    bs = x.shape[0]
                    x = x.permute(0, 2, 3, 1).view(bs, 128*128, 3).repeat(5, 1, 1)
                    input = get_mgrid(128, dim=2).cuda().unsqueeze(0)
                    input = input.repeat(5, 1, 1)

                    _mlp_list = []

                    for i in range(bs):
                        if fn[i] in cache:
                            _mlp_list.append(generate_mlp_from_weights(cache[fn[i]]).cuda())
                        else:
                            mlp = Siren(in_features=2, out_features=3, hidden_features=128,
                                        hidden_layers=3, outermost_linear=True).cuda()
                            _mlp_list.append(mlp)

                    combined_parameters = []
                    for _code in _mlp_list:
                        combined_parameters += list(_code.parameters())
                    optim = torch.optim.Adam(lr=1e-4, params=combined_parameters)

                    for i in range(100):
                        model_output = []
                        with self.accelerator.autocast():
                            for _code in _mlp_list:
                                mo, _ = _code(input)
                                model_output.append(mo)
                            model_output = torch.cat(model_output, dim=0)
                            loss = ((model_output - x) ** 2).mean()
                        self.accelerator.backward(loss)

                        optim.step()
                        optim.zero_grad()
                        if self.step % self.save_and_sample_every == 0 and self.accelerator.is_main_process:
                            print('done')
                            img_out = model_output[0]
                            gt_out = x[0]
                            vtils.save_image(img_out.view(128, 128, 3).permute(2, 0, 1),
                                             os.path.join('/data/pwojcik/ddmi_dump/', 'inr_t_{}.jpg'.format(self.step)),
                                             normalize=False, scale_each=False)
                            vtils.save_image(gt_out.view(128, 128, 3).permute(2, 0, 1),
                                             os.path.join('/data/pwojcik/ddmi_dump/', 'gt_t_{}.jpg'.format(self.step)),
                                             normalize=False, scale_each=False)

                    z = []

                    for idx, mlp in enumerate(_mlp_list):
                        state_dict = mlp.state_dict()
                        layers = []
                        layer_names = []
                        input = []
                        for l in state_dict:
                            st_shape = state_dict[l].shape
                            layers.append(np.prod(st_shape))
                            layer_names.append(l)
                            input.append(state_dict[l].flatten())
                        input = torch.hstack(input).cuda()
                        cache.update({fn[idx]: input})
                        z.append(input.unsqueeze(0))

                    z = torch.cat(z, dim=0)

                    with self.accelerator.autocast():
                        ## Encode latent
                        #with torch.no_grad():
                        #    if isinstance(self.vaemodel, torch.nn.parallel.DistributedDataParallel):
                        #        z = self.vaemodel.module.encode(y).sample()
                        #    else:
                        #        z = self.vaemodel.encode(y).sample()
                        ## LDM
                        z = z.detach()
                        p_loss,_ = self.diffusion_process(z)
                        p_loss = p_loss / self.gradient_accumulate_every

                    self.accelerator.backward(p_loss)
                    self.current_iters += 1

                    pbar.set_description('Dae loss : {:.3f}'.format(p_loss.item()))

                    self.accelerator.wait_for_everyone()

                    if self.current_iters % self.gradient_accumulate_every == self.gradient_accumulate_every - 1:
                        self.dae_opt.step()
                        self.dae_opt.zero_grad()
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            self.ema.update()

                if self.step % self.save_and_sample_every == 0 and self.accelerator.is_main_process: 
                    #coords = convert_to_coord_format_2d(1, 256, 256, device = self.accelerator.device, hstart=-255/256, hend = 255/256, wstart=-255/256, wend = 255/256)
                    self.ema.ema_model.eval()
                    with self.accelerator.autocast():
                        with torch.inference_mode():
                            z_test = self.ema.ema_model.sample(shape = shape, noise = noise_fix)
                            mlp = generate_mlp_from_weights(z_test[0]).cuda()
                            model_input = get_mgrid(128, 2).cuda().unsqueeze(0)
                            #if isinstance(self.vaemodel, torch.nn.parallel.DistributedDataParallel):
                            #    pe_test = self.vaemodel.module.decode(z_test)
                            #else:
                            #    pe_test = self.vaemodel.decode(z_test)
                            output_img, _ = mlp(model_input)
                            output_img = output_img.squeeze()
                            print('Saving in ', self.step)
                    #output_img = output_img.clamp(min = -1., max = 1.)
                    #output_img = unsymmetrize_image_data(output_img) =

                    vtils.save_image(output_img.view(128, 128, 3).permute(2, 0, 1), os.path.join('/data/pwojcik/ddmi_dump/', '{}.jpg'.format(self.step)), normalize = False, scale_each = False)
                    self.save(step = self.step)
                
                #if self.step % 100 == 0 and self.accelerator.is_main_process and self.step > 300:
                #    if self.test_data is not None:
                #        coords = convert_to_coord_format_2d(1, 256, 256, device = device, hstart=-255/256, hend = 255/256, wstart=-255/256, wend = 255/256)
                #        fid = test_fid_ddpm(self.ema, self.vaemodel, self.mlp, coords, self.test_data, self.accelerator)
                #        print('Step {} FID: {}'.format(self.step, fid))
                #    else:
                #        self.accelerator.print('Not found test dataset to evaluate!')

                self.accelerator.wait_for_everyone()
                self.step += 1
                pbar.update(1)

    @torch.no_grad()
    def eval(self):
        device = self.accelerator.device
        coords = convert_to_coord_format_2d(1, self.test_resolution, 
                                            self.test_resolution, 
                                            device = device, 
                                            hstart=-(self.test_resolution-1)/self.test_resolution, 
                                            hend = (self.test_resolution-1)/self.test_resolution, 
                                            wstart=-(self.test_resolution-1)/self.test_resolution, 
                                            wend = (self.test_resolution-1)/self.test_resolution)
        
        shape = [self.test_batch_size, self.channels, self.image_size, self.image_size]
        self.results_fid = os.path.join(self.results_folder, 'fid50k')
        os.makedirs(self.results_fid, exist_ok=True)
        fid = test_fid_ddpm_N(self.ema, self.vaemodel, self.mlp, coords, self.data, self.accelerator, shape, 10000, self.results_fid, save=True)
        print('Step {} FID: {}'.format(self.step, fid))

    @torch.no_grad()
    def generate(self):
        device = self.accelerator.device
        coords = convert_to_coord_format_2d(1, self.test_resolution, 
                                            self.test_resolution, 
                                            device = device, 
                                            hstart=-(self.test_resolution-1)/self.test_resolution, 
                                            hend = (self.test_resolution-1)/self.test_resolution, 
                                            wstart=-(self.test_resolution-1)/self.test_resolution, 
                                            wend = (self.test_resolution-1)/self.test_resolution)
        
        si = get_scale_injection(self.test_resolution)
        shape = [self.test_batch_size, self.channels, self.image_size, self.image_size]
        self.ema.ema_model.eval()
        with self.accelerator.autocast():
            z_test = self.ema.ema_model.sample(shape = shape)
            if isinstance(self.vaemodel, torch.nn.parallel.DistributedDataParallel):
                pe_test = self.vaemodel.module.decode(z_test)
            else:
                pe_test = self.vaemodel.decode(z_test)
            output_img = self.mlp(coords, hdbf=pe_test, si=si)
        output_img = output_img.clamp(min = -1., max = 1.)
        output_img = unsymmetrize_image_data(output_img)
        vtils.save_image(output_img, os.path.join(self.results_pth, 'generation.jpg'), normalize = False, scale_each = False)
        print('Finished generating images!')