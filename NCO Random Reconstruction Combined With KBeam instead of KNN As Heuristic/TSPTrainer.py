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
        
        new_sol_better_in_a_batch = if_repair.sum().item()

        double_solution[if_repair] = torch.cat((other_part_1[if_repair],
                                                        kkk_2[if_repair],
                                                        other_part_2[if_repair]),dim=1)
        after_repair_complete_solution = double_solution[:,first_node_index:first_node_index+the_whole_problem_size]

        return after_repair_complete_solution, new_sol_better_in_a_batch
     
    def _train_one_knn_batch(self,episode,batch_size):
        #DO NOT FORGET TO IMPLEMENT THE LOADING OF THE SAVED MODEL
        self.model.eval()
        with torch.no_grad():
            #Step 1: Sample Instances of X and Y instances from already loaded problem
            self.env.load_problems(episode,batch_size) #Make sure subsampling does not take place here
            beam_width = 3 # MTW
            beam_width_orig = beam_width # must be used for last iterations when beam_width longer than leftover nodes
            counter = 0
            if counter==0:
                problems_knn = self.env.problems
                counter+=1
                
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
            
            # Now reset the state of the environment
            reset_state, _, _ = self.env.reset('test')

            state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

            # 3. Reconstruct the sub-problem

            # zeros of length beam_width (no reward to begin with)
            cumulative_reward = torch.zeros(beam_width)
            RRC_reward_placeholder = []
            # For the first iteration we should have the selected node list, list form for beam_flexibility
            #updated_node_list = [self.env.beam_node_list.clone()]
            updated_node_list = [[0] for _ in range(beam_width)]
            reward_placeholder = torch.zeros(beam_width)
            while not done:
                if current_step_knn == 0:
                    selected_teacher = self.env.solution[:, -1]  # destination node
                    selected_student = self.env.solution[:, -1]

                elif current_step_knn == 1:
                    selected_teacher = self.env.solution[:, 0]  # starting node
                    selected_student = self.env.solution[:, 0]
                else:
                    
                    incremental_reward = []
                    teacher_list = []
                    student_list = []
                    
                    for path_idx in range(len(beam_node_list)):
                        #Execute the model for N number of steps =  # GOES TO LINE 51 OF CODE TSPModel MTW (TEST, REPAIR = TRUE)
                        _,_,_,_, selected_students, selected_teachers = self.model(
                            state,beam_node_list[path_idx],self.env.solution,current_step_knn, beam_width, beam_search = True, repair = True)
                        # Ends up being beam width * beam width long
                        # Append the top 'n' (where n is beam width) teachers and students
                        teacher_list.append(selected_teachers)
                        student_list.append(selected_students)
                    teacher_list = torch.stack(teacher_list).squeeze(0)
                    student_list = torch.stack(student_list).squeeze(0)

                    # Reshape into one tensor
                    if current_step_knn >= 3:
                        teacher_list = teacher_list.permute(1, 0, 2).reshape(batch_size, beam_width_orig*beam_width)
                        student_list = student_list.permute(1, 0, 2).reshape(batch_size, beam_width_orig*beam_width)
                    torch.set_num_threads(1)
                    # Get the cost of each path (by simulating a step)
                    for student_idx in range(len(student_list.T)):
                       future_beam_nodes = [node for node in beam_node_list for _ in range(beam_width)]
                       updated_node_list[student_idx], reward = self.env.simulated_step(student_list[:, student_idx], future_beam_nodes[student_idx], self.env.problems)
                       incremental_reward.append(reward) 
                    
                    # The cost for taking a step to a selected node
                    RRC_incremental_reward = incremental_reward
                    incremental_reward = torch.stack(incremental_reward).sum(dim=1)

                    # Duplicate cumulative_reward "beam width" times and then append the incremental reward to it, giving you the "total path cost"
                    print(f'Before reward placeholder {reward_placeholder.shape}')
                    print(f'Inc Reward Shape {incremental_reward.shape}')
                    print(f'Beam Width {beam_width}')
                    print(f'Self.env.solution[1] {self.env.solution.shape[1]}')
                    
                    # Duplicate cumulative_reward "beam width" times and then append the incremental reward to it, giving you the "total path cost"
                    reward_placeholder += torch.tensor(incremental_reward)
                    print(f'After reward placeholder {reward_placeholder.shape}')
                    torch.set_num_threads(torch.get_num_threads())
                    
                    if current_step_knn == 2:
                        RRC_reward_placeholder = RRC_incremental_reward
                    else:
                        RRC_reward_placeholder = [rp + ir for rp, ir in zip(RRC_reward_placeholder, RRC_incremental_reward)]

                    # Return lowest costs (dependent upon beam_width) ## 
                    _, top_indices = torch.topk(reward_placeholder, k=beam_width, largest=False)

                    # Return the top to update the environment to ensure that we know when complete (to do an overall update)
                    top_index = top_indices[0]
                    selected_student = student_list[:, top_index]
                    selected_teacher = teacher_list[:, top_index]

                    # Return the top "beam width" paths
                    beam_node_list = [updated_node_list[idx] for idx in top_indices]
                    # Return the top "cumulative reward" paths
                    cumulative_reward = reward_placeholder[top_indices]
                    RRC_reward = [RRC_reward_placeholder[idx] for idx in top_indices]

                # Increment the model (perform a step) (obsolete in beam_search but required for overall model and tells us when to exit while)
                state, _, _, done = self.env.step(selected_teacher, selected_student)
                current_step_knn += 1
                # As we run out of nodes to select, our beam_width must get smaller.
                if beam_width > (self.env.solution.shape[1]-current_step_knn):
                    beam_width_orig = beam_width
                    beam_width -= 1
                    
                if beam_width_orig > beam_width and current_step_knn <= 2:
                    reward_placeholder = torch.zeros(beam_width)
                    updated_node_list = [[0] for _ in range(beam_width)]
                    cumulative_reward = torch.zeros(beam_width)
                # Selected node list for first two knn iterations is from true model step, rest from simulated step.
                if current_step_knn <= 2:
                    beam_node_list = [self.env.selected_node_list.clone()]
                
                # We have to create a reward placeholder which is the cumulative reward repeated beam_width times.
                if current_step_knn >= 3:
                    updated_node_list = [[0] for _ in range(beam_width_orig*beam_width)]
                    reward_placeholder = cumulative_reward.repeat_interleave(beam_width)
                    RRC_reward_placeholder = [tensor for t in RRC_reward_placeholder for tensor in [t] * beam_width]

                if current_step_knn >= 8:
                    break_point = 0
            # Turn into a tensor for step 4: Beam_node_list must be tensor-ized

            # 4. decide whether to accept the reconstructed partial solution.
            
            ahter_repair_sub_solution = torch.roll(beam_node_list[0],shifts=-1,dims=1)
            # after_reward is the cumulative RRC_reward
            after_reward = RRC_reward[0] #self.env._get_travel_distance_2(beam_node_list, self.env.solution)
                
            after_repair_complete_solution,times_better_solution_found_in_a_batch = self.decide_whether_to_repair_solution(ahter_repair_sub_solution,
                                            before_reward, after_reward, first_node_index, length_of_subpath,
                                                                                double_solution)
            
            #after_repair_complete_solution = repaired_solutions.min() # How do we choose the min here easily? AMEED
            #current_step_knn += 1
        return problems_knn,after_repair_complete_solution,times_better_solution_found_in_a_batch

    def run(self):

        self.time_estimator.reset(self.start_epoch)

        self.env.load_raw_data(self.trainer_params['train_episodes'] )

        save_gap = []
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            self.env.shuffle_data()
            # Train
            train_score, train_student_score, train_loss,total_times_in_1_epoch_better_solution_found = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_student_score', epoch, train_student_score)
            self.result_log.append('train_loss', epoch, train_loss)
            self.result_log.append('total_times_in_1_epoch_better_solution_found', epoch, total_times_in_1_epoch_better_solution_found)

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
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_3'],self.result_log, labels=['total_times_in_1_epoch_better_solution_found'])

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
        total_times_in_1_epoch_better_solution_found = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            
            #First do RRC with existing model parameters:
            self.problems_knn,self.abc,times_better_solution_found_in_a_batch = self._train_one_knn_batch(episode,batch_size)  
            
            total_times_in_1_epoch_better_solution_found = total_times_in_1_epoch_better_solution_found + times_better_solution_found_in_a_batch

            
            #Now do normal training
            
            self.env.raw_data_tours[episode:episode + batch_size] = self.abc
            
            self.env.solution = self.abc

            avg_score,score_student_mean, avg_loss = self._train_one_batch(episode,batch_size,epoch,self.abc,self.problems_knn)

            score_AM.update(avg_score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size


            loop_cnt += 1
            self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}, Score_studetnt: {:.4f},  Loss: {:.4f}, Times Solution Improved/Batch: {:3d}'
                             .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                     score_AM.avg, score_student_AM.avg, loss_AM.avg,times_better_solution_found_in_a_batch))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f}, Score_studetnt: {:.4f}, Loss: {:.4f}, Times Solution Improved/Epoch: {:3d}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, score_student_AM.avg, loss_AM.avg,total_times_in_1_epoch_better_solution_found))

        return score_AM.avg, score_student_AM.avg, loss_AM.avg,total_times_in_1_epoch_better_solution_found

    def _train_one_batch(self, episode,batch_size,epoch,solution,problems_knn):

        ###############################################
        self.model.train()
        
        reset_state, _, _ = self.env.reset('train')

        prob_list = torch.ones(size=(batch_size, 0))

        state, reward,reward_student, done = self.env.pre_step()

        current_step=0

        while not done:
            if current_step == 0:
                selected_teacher = solution[:, -1] # destination node
                selected_student = solution[:, -1]
                prob = torch.ones(solution.shape[0], 1)
            elif current_step == 1:
                selected_teacher = solution[:, 0] # starting node
                selected_student = solution[:, 0]
                prob = torch.ones(solution.shape[0], 1)

            else:
                selected_teacher, prob, probs, selected_student, _, _ = self.model(problems_knn, self.env.selected_node_list, solution, current_step, 1, beam_search = True, mode='train')  # 更新被选择的点和概率
                loss_mean = -prob.type(torch.float64).log().mean()
                self.model.zero_grad()
                loss_mean.backward()
                self.optimizer.step()

            current_step+=1
            state, reward, reward_student, done = self.env.step(selected_teacher, selected_student,vanilla_mode=True,knnprob = problems_knn, sol = solution)

            prob_list = torch.cat((prob_list, prob), dim=1)

        loss_mean = -prob_list.log().mean()

        return 0,0, loss_mean.item()