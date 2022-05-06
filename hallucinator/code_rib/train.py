import torch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from LaFan import LaFan1
from torch.utils.data import Dataset, DataLoader
from model import StateEncoder, \
                  OffsetEncoder, \
                  TargetEncoder, \
                  LSTM, \
                  Decoder, \
                  ShortMotionDiscriminator, \
                  LongMotionDiscriminator
from skeleton import Skeleton
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from functions import gen_ztta
import yaml
import time
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='train-base.yaml')
    args = parser.parse_args()

    # opt = yaml.load(open('./config/test-base.yaml', 'r').read())
    opt = yaml.load(open('./config/' + args.cfg, 'r').read())
    # opt = yaml.load(open('.\\config\\train-base.yaml', 'r').read())
    # opt = yaml.load(open('./config/train-base.yaml', 'r').read())

    stamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    stamp  = stamp + '-' + opt['train']['method']
    # print(local_time)
    # assert 0
    if opt['train']['debug']:
        stamp = 'debug'
    log_dir = os.path.join('../log', stamp)
    model_dir = os.path.join('../model', stamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    def copydirs(from_file, to_file):
        if not os.path.exists(to_file):  
            os.makedirs(to_file)
        files = os.listdir(from_file)  
        for f in files:
            if os.path.isdir(from_file + '/' + f):
                copydirs(from_file + '/' + f, to_file + '/' + f)
            else:
                if '.git' not in f and '.zip' not in f and '.bvh' not in f:
                # if '.py' in f or '.yaml' in f or '.yml' in f:
                    shutil.copy(from_file + '/' + f, to_file + '/' + f)
    copydirs('./', log_dir + '/src')

    ## initilize the skeleton ##
    skeleton_mocap = Skeleton(offsets=opt['data']['offsets'], parents=opt['data']['parents'])
    skeleton_mocap.cuda()
    skeleton_mocap.remove_joints(opt['data']['joints_to_remove'])

    ## load train data ##
    lafan_data_train = LaFan1(opt['data']['data_dir'], \
                              seq_len = opt['model']['seq_length'], \
                              offset = opt['data']['offset'],\
                              train = True, debug=opt['train']['debug'])
    x_mean = lafan_data_train.x_mean.cuda()
    x_std = lafan_data_train.x_std.cuda().view(1, 1, opt['model']['num_joints'], 3)
    if opt['train']['debug']:
        opt['data']['num_workers'] = 1
    lafan_loader_train = DataLoader(lafan_data_train, \
                                    batch_size=opt['train']['batch_size'], \
                                    shuffle=True, num_workers=opt['data']['num_workers'])

    ## load test data ##
    # lafan_data_test = LaFan1(opt['data']['data_dir'], \
    #                           seq_len = opt['model']['seq_length'], \
    #                           train = False, debug=False)
    # lafan_loader_test = DataLoader(lafan_data_test, \
    #                                batch_size=opt['train']['batch_size'], \
    #                                shuffle=True, num_workers=opt['data']['num_workers'])

    ## initialize model ##
    
    state_encoder = StateEncoder(in_dim=opt['model']['state_input_dim'])
    state_encoder = state_encoder.cuda()
    offset_encoder = OffsetEncoder(in_dim=opt['model']['offset_input_dim'])
    offset_encoder = offset_encoder.cuda()
    target_encoder = TargetEncoder(in_dim=opt['model']['target_input_dim'])
    target_encoder = target_encoder.cuda()
    lstm = LSTM(in_dim=opt['model']['lstm_dim'], hidden_dim = opt['model']['lstm_dim'] * 2)
    lstm = lstm.cuda()
    decoder = Decoder(in_dim=opt['model']['lstm_dim'] * 2, out_dim=opt['model']['state_input_dim'])
    decoder = decoder.cuda()
    if len(opt['train']['pretrained']) > 0:
        state_encoder.load_state_dict(torch.load(os.path.join(opt['train']['pretrained'], 'state_encoder.pkl')))
        offset_encoder.load_state_dict(torch.load(os.path.join(opt['train']['pretrained'], 'offset_encoder.pkl')))
        target_encoder.load_state_dict(torch.load(os.path.join(opt['train']['pretrained'], 'target_encoder.pkl')))
        lstm.load_state_dict(torch.load(os.path.join(opt['train']['pretrained'], 'lstm.pkl')))
        decoder.load_state_dict(torch.load(os.path.join(opt['train']['pretrained'], 'decoder.pkl')))
        print('generator model loaded')

    if opt['train']['use_adv']:
        short_discriminator = ShortMotionDiscriminator(in_dim = (opt['model']['num_joints'] * 3 * 2))
        short_discriminator = short_discriminator.cuda()
        long_discriminator = LongMotionDiscriminator(in_dim = (opt['model']['num_joints'] * 3 * 2))
        long_discriminator = long_discriminator.cuda()
        if len(opt['train']['pretrained']) > 0:
            short_discriminator.load_state_dict(torch.load(os.path.join(opt['train']['pretrained'], 'short_discriminator.pkl')))
            long_discriminator.load_state_dict(torch.load(os.path.join(opt['train']['pretrained'], 'long_discriminator.pkl')))
            print('discriminator model loaded')

    # print('ztta:', ztta.size())
    # assert 0

    ## initilize optimizer_g ##
    optimizer_g = optim.Adam(lr = opt['train']['lr'], params = list(state_encoder.parameters()) +\
                                             list(offset_encoder.parameters()) +\
                                             list(target_encoder.parameters()) +\
                                             list(lstm.parameters()) +\
                                             list(decoder.parameters()), \
                                             betas = (opt['train']['beta1'], opt['train']['beta2']), \
                                             weight_decay = opt['train']['weight_decay'])
    if len(opt['train']['pretrained']) > 0:
        optimizer_g.load_state_dict(torch.load(os.path.join(opt['train']['pretrained'], 'optimizer_g.pkl')))
        print('optimizer_g model loaded')
    ## initialize optimizer_d ##
    if opt['train']['use_adv']:
        optimizer_d = optim.Adam(lr = opt['train']['lr'] * 0.1, params = list(short_discriminator.parameters()) +\
                                             list(long_discriminator.parameters()), \
                                             betas = (opt['train']['beta1'], opt['train']['beta2']), \
                                             weight_decay = opt['train']['weight_decay'])
        if len(opt['train']['pretrained']) > 0:
            optimizer_d.load_state_dict(torch.load(os.path.join(opt['train']['pretrained'], 'optimizer_d.pkl')))
            print('optimizer_d model loaded')
    
    writer = SummaryWriter(log_dir)
    loss_total_min = 10000000.0
    for epoch in range(opt['train']['num_epoch']):
        state_encoder.train()
        offset_encoder.train()
        target_encoder.train()
        lstm.train()
        decoder.train()
        loss_total_list = []

        if opt['train']['progressive_training']:
            ## get positional code ##
            if opt['train']['use_ztta']:
                ztta = gen_ztta(length = lafan_data_train.cur_seq_length).cuda()
                if (10 + (epoch // 2)) < opt['model']['seq_length']:
                    lafan_data_train.cur_seq_length = 10 + (epoch // 2)
                else:
                    lafan_data_train.cur_seq_length = opt['model']['seq_length']
        else:
            ## get positional code ##
            if opt['train']['use_ztta']:
                lafan_data_train.cur_seq_length = opt['model']['seq_length']
                ztta = gen_ztta(length = opt['model']['seq_length']).cuda()
                
        for i_batch, sampled_batch in tqdm(enumerate(lafan_loader_train)):
            # print(i_batch, sample_batched['local_q'].size())
            loss_pos = 0
            loss_quat = 0
            loss_contact = 0
            loss_root = 0
            # with torch.no_grad():
            if True:
                # state input
                local_q = sampled_batch['local_q'].cuda()
                root_v = sampled_batch['root_v'].cuda()
                contact = sampled_batch['contact'].cuda()
                # offset input
                root_p_offset = sampled_batch['root_p_offset'].cuda()
                local_q_offset = sampled_batch['local_q_offset'].cuda()
                local_q_offset = local_q_offset.view(local_q_offset.size(0), -1)
                # target input
                target = sampled_batch['target'].cuda()
                target = target.view(target.size(0), -1)
                # root pos
                root_p = sampled_batch['root_p'].cuda()
                # X
                X = sampled_batch['X'].cuda()

                if False:
                    print('local_q:', local_q.size(), \
                        'root_v:', root_v.size(), \
                        'contact:', contact.size(), \
                        'root_p_offset:', root_p_offset.size(), \
                        'local_q_offset:', local_q_offset.size(), \
                        'target:', target.size())
                
                lstm.init_hidden(local_q.size(0))
                h_list = []
                pred_list = []
                pred_list.append(X[:,0])
                # for t in range(opt['model']['seq_length'] - 1):
                for t in range(lafan_data_train.cur_seq_length - 1):
                    # root pos
                    if t  == 0:
                        root_p_t = root_p[:,t]
                        local_q_t = local_q[:,t]
                        local_q_t = local_q_t.view(local_q_t.size(0), -1)
                        contact_t = contact[:,t]
                        root_v_t = root_v[:,t]
                    else:
                        root_p_t = root_pred[0]
                        local_q_t = local_q_pred[0]
                        contact_t = contact_pred[0]
                        root_v_t = root_v_pred[0]
                        
                    # state input
                    state_input = torch.cat([local_q_t, root_v_t, contact_t], -1)
                    # offset input
                    # print('root_p_offset:', root_p_offset.size(), 'root_p_t:', root_p_t.size())
                    # print('local_q_offset:', local_q_offset.size(), 'local_q_t:', local_q_t.size())
                    root_p_offset_t = root_p_offset - root_p_t
                    local_q_offset_t = local_q_offset - local_q_t
                    # print('root_p_offset_t:', root_p_offset_t.size(), 'local_q_offset_t:', local_q_offset_t.size())
                    offset_input = torch.cat([root_p_offset_t, local_q_offset_t], -1)
                    # target input
                    target_input = target
                    

                    # print('state_input:',state_input.size())
                    h_state = state_encoder(state_input)
                    h_offset = offset_encoder(offset_input)
                    h_target = target_encoder(target_input)
                    
                    if opt['train']['use_ztta']:
                        h_state += ztta[:, t]
                        h_offset += ztta[:, t]
                        h_target += ztta[:, t]
                    # print('h_state:', h_state.size(),\
                    #       'h_offset:', h_offset.size(),\
                    #       'h_target:', h_target.size())
                    if opt['train']['use_adv']:
                        tta = lafan_data_train.cur_seq_length - 2 - t
                        if tta < 5:
                            lambda_target = 0.0
                        elif tta >=5 and tta < 30:
                            lambda_target = (tta - 5) / 25.0
                        else:
                            lambda_target = 1.0
                        h_offset += 0.5 * lambda_target * torch.cuda.FloatTensor(h_offset.size()).normal_()
                        h_target += 0.5 * lambda_target * torch.cuda.FloatTensor(h_target.size()).normal_()

                    h_in = torch.cat([h_state, h_offset, h_target], -1).unsqueeze(0)
                    h_out = lstm(h_in)
                    # print('h_out:', h_out.size())
                
                    h_pred, contact_pred = decoder(h_out)
                    local_q_v_pred = h_pred[:,:,:opt['model']['target_input_dim']]
                    local_q_pred = local_q_v_pred + local_q_t
                    # print('q_pred:', q_pred.size())
                    local_q_pred_ = local_q_pred.view(local_q_pred.size(0), local_q_pred.size(1), -1, 4)
                    local_q_pred_ = local_q_pred_ / torch.norm(local_q_pred_, dim = -1, keepdim = True)

                    root_v_pred = h_pred[:,:,opt['model']['target_input_dim']:]
                    root_pred = root_v_pred + root_p_t
                    # print(''contact:'', contact_pred.size())
                    # print('root_pred:', root_pred.size())
                    pos_pred = skeleton_mocap.forward_kinematics(local_q_pred_, root_pred)

                    pos_next = X[:,t+1]
                    local_q_next = local_q[:,t+1]
                    local_q_next = local_q_next.view(local_q_next.size(0), -1)
                    root_p_next = root_p[:,t+1]
                    contact_next = contact[:,t+1]
                    # print(pos_pred.size(), x_std.size())
                    loss_pos += torch.mean(torch.abs(pos_pred[0] - pos_next) / x_std) / lafan_data_train.cur_seq_length #opt['model']['seq_length']
                    loss_quat += torch.mean(torch.abs(local_q_pred[0] - local_q_next)) / lafan_data_train.cur_seq_length #opt['model']['seq_length']
                    loss_root += torch.mean(torch.abs(root_pred[0] - root_p_next) / x_std[:,:,0]) / lafan_data_train.cur_seq_length #opt['model']['seq_length']
                    loss_contact += torch.mean(torch.abs(contact_pred[0] - contact_next)) / lafan_data_train.cur_seq_length #opt['model']['seq_length']
                    pred_list.append(pos_pred[0])
                
                if opt['train']['use_adv']:
                    fake_input = torch.cat([x.reshape(x.size(0), -1).unsqueeze(-1) for x in pred_list], -1)
                    fake_v_input = torch.cat([fake_input[:,:,1:] - fake_input[:,:,:-1], torch.zeros_like(fake_input[:,:,0:1]).cuda()], -1)
                    fake_input = torch.cat([fake_input, fake_v_input], 1)

                    real_input = torch.cat([X[:, i].view(X.size(0), -1).unsqueeze(-1) for i in range(lafan_data_train.cur_seq_length)], -1)
                    real_v_input = torch.cat([real_input[:,:,1:] - real_input[:,:,:-1], torch.zeros_like(real_input[:,:,0:1]).cuda()], -1)
                    real_input = torch.cat([real_input, real_v_input], 1)
                    
                    optimizer_d.zero_grad()
                    short_fake_logits = torch.mean(short_discriminator(fake_input.detach())[:,0], 1)
                    short_real_logits = torch.mean(short_discriminator(real_input)[:,0], 1)
                    short_d_fake_loss = torch.mean((short_fake_logits) ** 2)
                    short_d_real_loss = torch.mean((short_real_logits -  1) ** 2)
                    short_d_loss = (short_d_fake_loss + short_d_real_loss) / 2.0

                    long_fake_logits = torch.mean(long_discriminator(fake_input.detach())[:,0], 1)
                    long_real_logits = torch.mean(long_discriminator(real_input)[:,0], 1)
                    long_d_fake_loss = torch.mean((long_fake_logits) ** 2)
                    long_d_real_loss = torch.mean((long_real_logits -  1) ** 2)
                    long_d_loss = (long_d_fake_loss + long_d_real_loss) / 2.0
                    total_d_loss = opt['train']['loss_adv_weight'] * long_d_loss + \
                                   opt['train']['loss_adv_weight'] * short_d_loss
                    total_d_loss.backward()
                    optimizer_d.step() 
                
                optimizer_g.zero_grad()
                pred_pos = torch.cat([x.reshape(x.size(0), -1).unsqueeze(-1) for x in pred_list], -1)
                pred_vel = (pred_pos[:,opt['data']['foot_index'],1:] - pred_pos[:,opt['data']['foot_index'],:-1])
                pred_vel = pred_vel.view(pred_vel.size(0), 4, 3, pred_vel.size(-1))
                loss_slide = torch.mean(torch.abs(pred_vel * contact[:,:-1].permute(0, 2, 1).unsqueeze(2)))
                loss_total = opt['train']['loss_pos_weight'] * loss_pos + \
                            opt['train']['loss_quat_weight'] * loss_quat + \
                            opt['train']['loss_root_weight'] * loss_root + \
                            opt['train']['loss_slide_weight'] * loss_slide + \
                            opt['train']['loss_contact_weight'] * loss_contact
                
                

                if opt['train']['use_adv']:
                    short_fake_logits = torch.mean(short_discriminator(fake_input)[:,0], 1)
                    short_g_loss = torch.mean((short_fake_logits -  1) ** 2)
                    long_fake_logits = torch.mean(long_discriminator(fake_input)[:,0], 1)
                    long_g_loss = torch.mean((long_fake_logits -  1) ** 2)
                    total_g_loss = opt['train']['loss_adv_weight'] * long_g_loss + \
                                   opt['train']['loss_adv_weight'] * short_g_loss
                    loss_total += total_g_loss

                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(state_encoder.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(offset_encoder.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(target_encoder.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(lstm.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
                optimizer_g.step()
                # print("epoch: %03d, batch: %03d, pos: %.3f, quat: %.3f, root: %.3f, cont: %.3f"%\
                #               (epoch, \
                #               i_batch, \
                #               loss_pos.item(), \
                #               loss_quat.item(), \
                #               loss_root.item(), \
                #               loss_contact.item()))
                writer.add_scalar('loss_pos', loss_pos.item(), global_step = epoch * 317 + i_batch)
                writer.add_scalar('loss_quat', loss_quat.item(), global_step = epoch * 317 + i_batch)
                writer.add_scalar('loss_root', loss_root.item(), global_step = epoch * 317 + i_batch)
                writer.add_scalar('loss_slide', loss_slide.item(), global_step = epoch * 317 + i_batch)
                writer.add_scalar('loss_contact', loss_contact.item(), global_step = epoch * 317 + i_batch)
                writer.add_scalar('loss_total', loss_total.item(), global_step = epoch * 317 + i_batch)

                if opt['train']['use_adv']:
                    writer.add_scalar('loss_short_g', short_g_loss.item(), global_step = epoch * 317 + i_batch)
                    writer.add_scalar('loss_long_g', long_g_loss.item(), global_step = epoch * 317 + i_batch)
                    writer.add_scalar('loss_short_d_real', short_d_real_loss.item(), global_step = epoch * 317 + i_batch)
                    writer.add_scalar('loss_short_d_fake', short_d_fake_loss.item(), global_step = epoch * 317 + i_batch)
                    writer.add_scalar('loss_long_d_real', long_d_real_loss.item(), global_step = epoch * 317 + i_batch)
                    writer.add_scalar('loss_long_d_fake', long_d_fake_loss.item(), global_step = epoch * 317 + i_batch)
                loss_total_list.append(loss_total.item())
        
        loss_total_cur = np.mean(loss_total_list)
        if loss_total_cur < loss_total_min:
            loss_total_min = loss_total_cur
            torch.save(state_encoder.state_dict(), model_dir + '/state_encoder.pkl')
            torch.save(target_encoder.state_dict(), model_dir + '/target_encoder.pkl')
            torch.save(offset_encoder.state_dict(), model_dir + '/offset_encoder.pkl')
            torch.save(lstm.state_dict(), model_dir + '/lstm.pkl')
            torch.save(decoder.state_dict(), model_dir + '/decoder.pkl')
            torch.save(optimizer_g.state_dict(), model_dir + '/optimizer_g.pkl')
            if opt['train']['use_adv']:
                torch.save(short_discriminator.state_dict(), model_dir + '/short_discriminator.pkl')
                torch.save(long_discriminator.state_dict(), model_dir + '/long_discriminator.pkl')
                torch.save(optimizer_d.state_dict(), model_dir + '/optimizer_d.pkl')
        print("train epoch: %03d, cur total loss:%.3f, cur best loss:%.3f" % (epoch, loss_total_cur, loss_total_min))
                    

                
                
