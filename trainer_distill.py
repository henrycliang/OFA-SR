import os
import math
from decimal import Decimal

import utility
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from thop import profile
from thop import clever_format
from model.modules.super_modules import SuperConv2d,SuperLinear
from model.modules.slimmable_ops import SlimmableConv2d,FLAGS,SlimmableUpConv2d,FLAGS_UpConv2d,SlimmableUpLinear,FLAGS_UpLinear
from thop.count_hooks import count_convNd,count_softmax,count_linear


custom_ops={SuperConv2d:count_convNd,SlimmableConv2d:count_convNd,SuperLinear:count_linear,nn.Softmax:count_softmax,
            SlimmableUpConv2d:count_convNd,SlimmableUpLinear:count_linear}

class Trainer_Distill():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8


    ######by given the scale and the size of input image
    ######we caculate the input matrix for the weight prediction network
    ###### input matrix for weight prediction network

    def input_matrix_wpn_new(self, inH, inW, scale, add_scale=True):
        '''
        inH, inW: the size of the feature maps
        scale: is the upsampling times
        '''
        outH, outW = int(scale * inH), int(scale * inW)
        #### mask records which pixel is invalid, 1 valid or o invalid
        #### h_offset and w_offset caculate the offset to generate the input matrix
        scale_int = int(math.ceil(scale))
        h_offset = torch.ones(inH, scale_int, 1)
        mask_h = torch.zeros(inH, scale_int, 1)
        w_offset = torch.ones(1, inW, scale_int)
        mask_w = torch.zeros(1, inW, scale_int)


        ####projection  coordinate  and caculate the offset
        h_project_coord = torch.arange(0, outH, 1.).mul(1.0 / scale)
        int_h_project_coord = torch.floor(h_project_coord)

        offset_h_coord = h_project_coord - int_h_project_coord
        int_h_project_coord = int_h_project_coord.int()

        w_project_coord = torch.arange(0, outW, 1.).mul(1.0 / scale)
        int_w_project_coord = torch.floor(w_project_coord)

        offset_w_coord = w_project_coord - int_w_project_coord
        int_w_project_coord = int_w_project_coord.int()

        ####flag for   number for current coordinate LR image
        flag = 0
        number = 0
        for i in range(outH):
            if int_h_project_coord[i] == number:
                h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], flag, 0] = 1
                flag += 1
            else:
                h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], 0, 0] = 1
                number += 1
                flag = 1

        flag = 0
        number = 0
        for i in range(outW):
            if int_w_project_coord[i] == number:
                w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], flag] = 1
                flag += 1
            else:
                w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], 0] = 1
                number += 1
                flag = 1

        ## the size is scale_int* inH* (scal_int*inW)
        h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
        ####
        mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

        pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)


        mask_mat = torch.sum(torch.cat((mask_h, mask_w), 2), 2).view(scale_int * inH, scale_int * inW)
        mask_mat = mask_mat.eq(2)

        # i = 1
        # h, w,_ = pos_mat.size()
        # while(pos_mat[i][0][0]>= 1e-6 and i<h):
        #     i = i+1
        #
        # j = 1
        # #pdb.set_trace()
        # h, w,_ = pos_mat.size()
        # while(pos_mat[0][j][1]>= 1e-6 and j<w):
        #     j = j+1
        #
        # pos_mat_small = pos_mat[0:i,0:j,:]
        pos_mat_small=pos_mat

        pos_mat_small = pos_mat_small.contiguous().view(1, -1, 2)
        if add_scale:
            scale_mat = torch.zeros(1, 1)
            scale_mat[0, 0] = 1.0 / scale
            scale_mat = torch.cat([scale_mat] * (pos_mat_small.size(1)), 0)  ###(inH*inW*scale_int**2, 4)
            pos_mat_small = torch.cat((scale_mat.view(1, -1, 1), pos_mat_small), 2)

        return pos_mat_small, mask_mat  ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW

        ########speed up the model by removing the computation

    def pos_mapping_and_mask(self,inH,inW,scale,add_scale=True):
        """Get a new mask and pos_mat with size of [outH,outW]"""
        outH,outW=int(scale*inH),int(scale*inW)
        #### mask records which pixel is invalid, 1 valid or o invalid
        scale_int = int(math.ceil(scale))
        mask_h = torch.zeros(inH, scale_int, 1)
        mask_w = torch.zeros(1, inW, scale_int)

        # projection coordinate
        h_project_coord = torch.arange(0, outH, 1.).mul(1.0 / scale)
        int_h_project_coord = torch.floor(h_project_coord) #取出SR[i,j]像素SR映射到LR的像素的位置
        offset_h_coord = h_project_coord - int_h_project_coord #SR[i,j]像素投影到LR后的小数部分
        int_h_project_coord = int_h_project_coord.int()

        w_project_coord = torch.arange(0, outW, 1.).mul(1.0 / scale)
        int_w_project_coord = torch.floor(w_project_coord)
        offset_w_coord = w_project_coord - int_w_project_coord
        int_w_project_coord = int_w_project_coord.int()

        # 产生投影矩阵作为Meta层输入
        offset_h_coord=offset_h_coord.view(-1,1,1)
        offset_h_coord=torch.cat([offset_h_coord]*outW,1)

        offset_w_coord=offset_w_coord.view(1,-1,1)
        offset_w_coord = torch.cat([offset_w_coord] * outH, 0)

        pos_mat=torch.cat((offset_h_coord,offset_w_coord),2)
        pos_mat=pos_mat.contiguous().view(1,-1,2)

        if add_scale:
            scale_mat = torch.zeros(1, 1)
            scale_mat[0, 0] = 1.0 / scale
            scale_mat = torch.cat([scale_mat] * (pos_mat.size(1)), 0)
            pos_mat=torch.cat((scale_mat.view(1,-1,1),pos_mat),2)

        # generate the mask for valid pixels
        flag=0
        number=0
        for i in range(outH):
            if int_h_project_coord[i]==number:
                mask_h[int_h_project_coord[i], flag, 0]=1
                flag+=1
            else:
                mask_h[int_h_project_coord[i], 0, 0]=1
                number+=1
                flag=1

        flag = 0
        number = 0
        for i in range(outW):
            if int_w_project_coord[i] == number:
                mask_w[0,int_w_project_coord[i], flag] = 1
                flag += 1
            else:
                mask_w[0,int_w_project_coord[i], 0] = 1
                number += 1
                flag = 1

        mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        # 先cat成[50,2,100]再view成[100,100], view的时候是按2的那一维的次序
        mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
        mask_mat = torch.sum(torch.cat((mask_h, mask_w), 2), 2).view(scale_int * inH, scale_int * inW)
        mask_mat = mask_mat.eq(2)
        mask_mat =mask_mat.view(1,mask_mat.size(0),mask_mat.size(1))


        return pos_mat,mask_mat



    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )

        self.loss.start_log()
        self.model.train()

        
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        
        timer_data, timer_model = utility.timer(), utility.timer()

        # Set the scale, return the index of scale
        index_scale=utility.get_scale_config(self.args,0)
        self.loader_train.dataset.set_scale(index_scale)
        # divide the scale to full-model and narrow model
        scale_wide=[3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]
        # scale_wide=[4.0]
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            N, C, H, W = lr.size()
            _,_,outH,outW = hr.size()
            out_size=[outH,outW]

            scale = self.args.scale[idx_scale.numpy()[0]]
            scale_int=math.ceil(scale)

            self.optimizer.zero_grad()

            # ----------------------------------------------------------------------------------------------------------------------
            if self.args.model.find('metashuffle')>=1 and self.args.small_mask: # 使用small_mask时执行这一行
                # Using the scale size output
                scale_coord_map, mask = self.pos_mapping_and_mask(H, W, scale)
                if self.args.n_GPUs > 1 and not self.args.cpu:
                    mask            =torch.cat([mask]*self.args.n_GPUs,0)
                    scale_coord_map = torch.cat([scale_coord_map] * self.args.n_GPUs, 0)
                else:
                    scale_coord_map = scale_coord_map.to(device)
                # 将mask和scale_coord_map打包成字典
                scale_coord_map = {'pos_mat': scale_coord_map,
                                    'mask': mask}
                # ## Compute the Flops #########################################################################
                # if epoch == 1 and batch == 0:
                #     # FLAGS.width_mult = math.ceil(scale_int ** 2) * 1. / 16
                #     FLAGS.width_mult = (20./64*scale)-16./64
                #     FLAGS_UpConv2d.width_mult =(20./64*scale)-16./64
                #     FLAGS_UpLinear.width_mult = 1.0
                #     input = torch.randn(1, 3, 50, 50).cuda()
                #     flops, params = profile(self.model, inputs=(input, scale, out_size, scale_coord_map),
                #                             verbose=True,custom_ops=custom_ops)
                #     flops, params = clever_format([flops, params], "%.3f")
                #     print('The Flops and in  the model:', flops)
                #     print('The parameters in the model:', params)
                # ##############################################################################################
                ## Compute the widest model's output sr_w ##
                FLAGS.width_mult=1.0
                FLAGS_UpConv2d.width_mult=1.0
                FLAGS_UpLinear.width_mult=1.0
                sr_w = self.model(lr, scale, out_size, scale_coord_map)
                loss_w = self.loss(sr_w, hr)
                if loss_w.item() < self.args.skip_threshold * self.error_last:
                    loss_w.backward()
                else:
                    print('Skip this batch {}! (Loss: {})'.format(
                        batch + 1, loss_w.item()
                    ))

                ## Compute the narrow channel_width model's output sr_n ##
                if scale in scale_wide:
                    # loss=loss_w
                    self.optimizer.step()
                    # print('in scale wide')
                else:
                    # FLAGS.width_mult = (20./64*scale)-16./64
                    # FLAGS_UpConv2d.width_mult =(20./64*scale)-16./64
                    # FLAGS.width_mult = math.ceil(scale_int ** 2) * 1. / 16
                    # FLAGS_UpConv2d.width_mult = math.ceil(scale_int ** 2) * 1. / 16
                    # FLAGS.width_mult = (10. * scale+24) / 64
                    FLAGS.width_mult = scale_int*0.25
                    if scale_int<=3:
                        FLAGS_UpConv2d.width_mult = 0.5625
                        FLAGS_UpLinear.width_mult = 0.5625
                    else:
                        FLAGS_UpConv2d.width_mult = 1.0
                        FLAGS_UpLinear.width_mult = 1.0
                    # print(FLAGS.width_mult)

                    sr_n = self.model(lr, scale, out_size, scale_coord_map)
                    loss = self.loss(sr_n,sr_w.detach())
                    if loss.item() < self.args.skip_threshold * self.error_last:
                        loss.backward()
                        self.optimizer.step()
                    else:
                        print('Skip this batch {}! (Loss: {})'.format(
                            batch + 1, loss.item()
                        ))

            else:
                print('Please input correct configuration!')



            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

            # random set the scale after each iteration
            index_scale=utility.get_scale_config(self.args,0)
            self.loader_train.dataset.set_scale(index_scale)

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        if self.args.n_GPUs == 1:
            target = self.model
        else:
            target = self.model  #.module

        torch.save(
            target.state_dict(),
            os.path.join(self.ckp.dir,'model', 'model_{}.pt'.format(epoch))
        )
        ## save models

    def test(self):  
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()
        timer_test = utility.timer()
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                eval_acc_ssim = 0
                self.loader_test.dataset.set_scale(idx_scale)
                #tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, _) in enumerate(self.loader_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    N,C,H,W = lr.size()
                    scale = self.args.scale[idx_scale]
                    outH,outW = int(H*scale),int(W*scale)
                    out_size=[outH,outW]
                    #_,_,outH,outW = hr.size()
                    #timer_test.tic()

                    timer_test.tic()
                    if self.args.model.find('metashuffle') >= 1 and self.args.small_mask:
                        # Using the scale size output
                        scale_coord_map, mask = self.pos_mapping_and_mask(H, W, scale)
                        if self.args.n_GPUs > 1 and not self.args.cpu:
                            scale_coord_map = torch.cat([scale_coord_map] * self.args.n_GPUs, 0)
                        else:
                            scale_coord_map = scale_coord_map.to(device)
                        # 将mask和scale_coord_map打包成字典
                        scale_coord_map = {'pos_mat': scale_coord_map,
                                           'mask': mask}
                        # FLAGS.width_mult = (20./64*scale)-16./64
                        # FLAGS_UpConv2d.width_mult = (20./64*scale)-16./64
                        # FLAGS.width_mult = (10. * scale + 24) / 64
                        scale_int=math.ceil(scale)
                        FLAGS.width_mult=scale_int*0.25
                        if scale_int <= 3:
                            FLAGS_UpConv2d.width_mult = 0.5625
                            FLAGS_UpLinear.width_mult = 0.5625
                        else:
                            FLAGS_UpConv2d.width_mult = 1.0
                            FLAGS_UpLinear.width_mult = 1.0

                        sr = self.model(lr, scale, out_size, scale_coord_map)
                    else:
                        print('Please input correct configuration!')

                    timer_test.hold()

                    sr = utility.quantize(sr, self.args.rgb_range)
                    #timer_test.hold()
                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        eval_acc_ssim += utility.calc_ssim(
                            sr, hr, scale,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results and idx_img%10==0:  #保存输出的图像
                        a=1
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
               # print(timer_test.acc/100)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} SSIM: {:.4f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        eval_acc_ssim / len(self.loader_test),
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )
        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

    def config_proxy(self,model_name):
        return 0

