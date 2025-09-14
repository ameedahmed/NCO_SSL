##AUTO MIXED PRECISION TRAINING Removed
from logging import getLogger

import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from LEHD.TSP.TSPModel import TSPModel as Model
from LEHD.TSP.test import main_test
from LEHD.TSP.TSPEnv import TSPEnv as Env
from LEHD.utils.utils import *
import torch.nn.functional as F


class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params,
                 curriculum_params =None):


        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.curriculum_params = curriculum_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda'] # True
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        random_seed = 123
        torch.manual_seed(random_seed)
        # Main Components
        self.model = Model(**self.model_params)

        self.env = Env(**self.env_params)

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()
        self.start_time = time.time()
        print("Training Start Time :",self.start_time)

    def run(self):

        self.time_estimator.reset(self.start_epoch)

        self.env.load_raw_data(self.trainer_params['train_episodes'] )
        
        metrics_log = {
            "epochs": [],
            "train_loss": [],
            "train_score": [],
            "problem_size": []
        }

        save_gap = []
        for epochs in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            self.env.shuffle_data()
            
            #applying curr learning by adjusting the problem size based on the current epoch
            #current_problem_size = min(
            #    self.env.problem_size + (epoch // self.curriculum_params['size_increase_epochs']) * 10, 
            #    self.curriculum_params['max_problem_size']
            #)
            
            current_problem_size = self.env.problem_size
            
            self.env.load_problems(episode=epochs, batch_size=self.trainer_params['train_batch_size'],epoch=epochs,
                                   curriculum_params= self.curriculum_params)
            
            
            # Train
            train_score, train_student_score, train_loss = self._train_one_epoch(epochs)
            
            #log the metrics for visualization
            metrics_log['epochs'].append(epochs)
            metrics_log['train_loss'].append(train_loss)
            metrics_log['train_score'].append(train_score)
            metrics_log['problem_size'].append(current_problem_size)
            
            #save the metrics at each epoch for the case of failure
            np.save(f"{self.result_folder}/metrics_log.npy", metrics_log)
            
            self.scheduler.step()
        
            
            self.result_log.append('train_score', epochs, train_score)
            self.result_log.append('train_student_score', epochs, train_student_score)
            self.result_log.append('train_loss', epochs, train_loss)
            self.scheduler.step()

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epochs, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epochs, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epochs == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epochs > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done or (epochs % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epochs))

                score_optimal, score_student ,gap = main_test(epochs,self.result_folder,use_RRC=False,
                                                              cuda_device_num=self.trainer_params['cuda_device_num'])

                save_gap.append([score_optimal, score_student,gap])
                np.savetxt(self.result_folder+'/gap.txt',save_gap,delimiter=',',fmt='%s')
            

            if all_done or (epochs % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epochs)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
                torch.cuda.empty_cache()
        # Log training time
        current_time = time.time()
        training_time = current_time - self.start_time
        self.logger.info(f" Total training  time: {training_time:.2f} seconds")
        
        #after training we will save the metrics for later
        np.save(f"{self.result_folder}/final_metrics_log.npy", metrics_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score,score_student_mean, avg_loss = self._train_one_batch(episode,batch_size,epochs=epoch,curriculum_params= self.curriculum_params)

            score_AM.update(avg_score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size


            loop_cnt += 1
            self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}, Score_studetnt: {:.4f},  Loss: {:.4f}'
                             .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                     score_AM.avg, score_student_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f}, Score_studetnt: {:.4f}, Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, score_student_AM.avg, loss_AM.avg))

        return score_AM.avg, score_student_AM.avg, loss_AM.avg

    def _train_one_batch(self, episode,batch_size,epochs,curriculum_params=None):

        ###############################################
        self.model.train()
        self.env.load_problems(episode,batch_size,epoch=epochs,curriculum_params= self.curriculum_params)
        reset_state, _, _ = self.env.reset(self.env_params['mode'])

        prob_list = torch.ones(size=(batch_size, 0))

        state, reward,reward_student, done = self.env.pre_step()

        current_step=0

        while not done:
            if current_step == 0:
                selected_teacher = self.env.solution[:, -1] # destination node
                selected_student = self.env.solution[:, -1]
                prob = torch.ones(self.env.solution.shape[0], 1)
            elif current_step == 1:
                selected_teacher = self.env.solution[:, 0] # starting node
                selected_student = self.env.solution[:, 0]
                prob = torch.ones(self.env.solution.shape[0], 1)

            else:
                selected_teacher, prob, probs, selected_student = self.model(state, self.env.selected_node_list, self.env.solution, current_step)  # 更新被选择的点和概率
                loss_mean = -prob.type(torch.float64).log().mean()
                self.model.zero_grad()
                loss_mean.backward()
                self.optimizer.step()

            current_step+=1
            state, reward, reward_student, done = self.env.step(selected_teacher, selected_student)

            prob_list = torch.cat((prob_list, prob), dim=1)

        loss_mean = -prob_list.log().mean()

        return 0,0, loss_mean.item()