##AUTO MIXED PRECISION TRAINING Removed
from logging import getLogger

import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from LEHD.TSP.TSPModel import TSPModel as Model
from LEHD.TSP.test import main_test
from LEHD.TSP.TSPEnv import TSPEnv as Env
from LEHD.utils.utils import *


class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):


        
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

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
        self.start_epoch = 6
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

    def decide_whether_to_repair_solution(self,after_repair_sub_solution,before_reward, after_reward,
                                          first_node_index, length_of_subpath, double_solution):

        the_whole_problem_size  = int(double_solution.shape[1]/2)

        other_part_1 = double_solution[:,:first_node_index]
        other_part_2 = double_solution[:,first_node_index+length_of_subpath:]
        origin_sub_solution = double_solution[:, first_node_index : first_node_index+length_of_subpath]

        jjj, _ = torch.sort(origin_sub_solution, dim=1, descending=False)

        index = torch.arange(jjj.shape[0])[:,None].repeat(1,jjj.shape[1])

        kkk_2 = jjj[index,after_repair_sub_solution]

        if_repair = before_reward>after_reward

        double_solution[if_repair] = torch.cat((other_part_1[if_repair],
                                                        kkk_2[if_repair],
                                                        other_part_2[if_repair]),dim=1)
        after_repair_complete_solution = double_solution[:,first_node_index:first_node_index+the_whole_problem_size]

        return after_repair_complete_solution

    def run(self):

        self.time_estimator.reset(self.start_epoch)

        self.env.load_raw_data(self.trainer_params['train_episodes'] )

        save_gap = []
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            self.env.shuffle_data()
            # Train
            train_score, train_student_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_student_score', epoch, train_student_score)
            self.result_log.append('train_loss', epoch, train_loss)
            self.scheduler.step()

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

                score_optimal, score_student ,gap = main_test(epoch,self.result_folder,use_RRC=False,
                                                              cuda_device_num=self.trainer_params['cuda_device_num'])

                save_gap.append([score_optimal, score_student,gap])
                np.savetxt(self.result_folder+'/gap.txt',save_gap,delimiter=',',fmt='%s')

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
        # Log training time
        current_time = time.time()
        training_time = current_time - self.start_time
        self.logger.info(f" Total training  time: {training_time:.2f} seconds")

    def _train_one_epoch(self, epoch):

        
        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        budget=50
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            
            #First do RRC
            self.env_params['mode'] = 'test'
            self.model_params['mode'] = 'test'
            self.env_params['sub_path'] = False
            self.trainer_params['model_load']['enable'] = 'True'

            #Load training parameters again
            self.env.solution = self._train_one_knn_batch(episode,batch_size,10)
            
            #Load training parameters again
            self.env_params['mode'] = 'train'
            self.model_params['mode'] = 'train'            
            self.env_params['sub_path'] = True
            self.trainer_params['model_load']['enable'] = 'False'


            
            avg_score,score_student_mean, avg_loss = self._train_one_batch(episode,batch_size,epoch)

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

    def _train_one_batch(self, episode,batch_size,epoch):

        ###############################################
        self.model.train()
        reset_state, _, _ = self.env.reset(self.env_params['mode'])

        prob_list = torch.ones(size=(batch_size, 0))

        state, reward,reward_student, done = self.env.pre_step()

        #Implement RRC
        
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

    def _train_one_knn_batch(self, episode,batch_size,budget):
        #DO NOT FORGET TO IMPLEMENT THE LOADING OF THE SAVED MODEL
        for i in range(budget):
            #Step 1: Sample Instances of X and Y instances from already loaded problem
            self.env.load_problems(episode,batch_size) #NOTE: Make sure subsampling does not take place here
            best_select_node_list = self.env.solution
            #Step 2: Perform Random Reconstruction Mechanism
            #Step 2.1: Randomly sample partial solution:
            if_inverse = True
            if_inverse_index = torch.randint(low=0, high=100, size=[1])[0]  # in [4,N]
            if if_inverse_index<50:
                if_inverse=False

            if if_inverse:
                best_select_node_list = torch.flip(best_select_node_list,dims=[1])
            partial_solution_length, first_node_index,length_of_subpath,double_solution = self.env.destroy_solution(self.env.problems,best_select_node_list ) 
            before_reward = partial_solution_length
            current_step_knn = 0
            
            #Now reset the state of the environment
            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node
        
            # 3. Reconstruct the sub-problem

            while not done:
                if current_step_knn == 0:
                    selected_teacher = self.env.solution[:, -1]  # detination node
                    selected_student = self.env.solution[:, -1]

                elif current_step_knn == 1:
                    selected_teacher = self.env.solution[:, 0]  # starting node
                    selected_student = self.env.solution[:, 0]

                else:
                    #Execute the model for N number of steps = 
                    selected_teacher, _,_,selected_student = self.model(
                        state,self.env.selected_node_list,self.env.solution,current_step_knn, repair = True)

                current_step_knn += 1
                state, reward, reward_student, done = self.env.step(selected_teacher, selected_student)  # 更新 selected_teacher list 和 mask

            ahter_repair_sub_solution = torch.roll(self.env.selected_node_list,shifts=-1,dims=1)

            after_reward = reward_student

            # 4. decide whether to accepect the reconstructed partial solution.
            #Run the function to decide whether to repair and accept the new solution
            after_repair_complete_solution = self.decide_whether_to_repair_solution(ahter_repair_sub_solution,
                                                before_reward, after_reward, first_node_index, length_of_subpath,
                                                                                    double_solution )
            #Step 5: Update the entire solution with best solutions
            self.env.solution = after_repair_complete_solution
            #Replace the originally loaded data with new improved labels/paths
            self.env.raw_data_tours[episode:episode + batch_size] = self.env.solution
            
            return self.env.solution
            
        
        
        
        """current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

        escape_time,_ = clock.get_est_string(1, 1)
        gap =  ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
        self.logger.info("RRC step{}, name:{}, gap:{:4f} %, Elapsed[{}], stu_l:{:4f} , opt_l:{:4f}".format(
            bbbb,name,gap, escape_time,current_best_length.mean().item(), self.optimal_length.mean().item()))
"""