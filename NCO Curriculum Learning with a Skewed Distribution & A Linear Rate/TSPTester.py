##Implemented Auto Mixed Precision Package

from logging import getLogger

import numpy as np
import torch

from LEHD.TSP.TSPModel import TSPModel as Model
from LEHD.TSP.TSPEnv import TSPEnv as Env
from LEHD.utils.utils import *
from torch.cuda.amp import GradScaler, autocast


class TSPTester():
    def __init__(self,
                 epochs,
                 env_params,
                 model_params,
                 tester_params,curriculum_params=None):

        # save arguments
        self.epochs = epochs
        self.env_params = env_params
        self.model_params = model_params
        self.curriculum_params = curriculum_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
    

        
        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.set_printoptions(precision=20)
        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 =  TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()


        self.env.load_raw_data(self.tester_params['test_episodes'] )

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        problems_100 = []
        problems_100_200 = []
        problems_200_500 = []
        problems_500_1000 = []
        problems_1000 = []
        
        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)
            
            torch.cuda.empty_cache()

            score_teacher, score_student,problems_size = self._test_one_batch(episode,batch_size,self.epochs,clock=self.time_estimator_2)
            current_gap = (score_student-score_teacher)/score_teacher
            if problems_size<100:
                problems_100.append(current_gap)
            elif 100<=problems_size<200:
                problems_100_200.append(current_gap)
            elif 200<=problems_size<500:
                problems_200_500.append(current_gap)
            elif 500<=problems_size<1000:
                problems_500_1000.append(current_gap)
            elif 1000<=problems_size:
                problems_1000.append(current_gap)

            print('problems_100 mean gap:',np.mean(problems_100),len(problems_100))
            print('problems_100_200 mean gap:', np.mean(problems_100_200),len(problems_100_200))
            print('problems_200_500 mean gap:', np.mean(problems_200_500),len(problems_200_500))
            print('problems_500_1000 mean gap:', np.mean(problems_500_1000),len(problems_500_1000))
            print('problems_1000 mean gap:', np.mean(problems_1000),len(problems_1000))
            score_AM.update(score_teacher, batch_size)
            score_student_AM.update(score_student, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f},Score_studetnt: {:.4f},".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score_teacher,score_student,))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" Teacher SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" Student SCORE: {:.4f} ".format(score_student_AM.avg))
                self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg-score_AM.avg) / score_AM.avg * 100))
                gap_ = (score_student_AM.avg-score_AM.avg) / score_AM.avg * 100

        return score_AM.avg, score_student_AM.avg, gap_

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

    def _test_one_batch(self, episode,batch_size,epochs,clock=None):

        self.model.eval() #Load the model in evaluation mode
        with torch.no_grad(): #Since the model is in eval mode, you will use torch.no_grad to prevent using the gradients
            self.env.load_problems(episode, batch_size,epoch=epochs,curriculum_params=self.curriculum_params) #Returns problem and solution

            self.origin_problem = self.env.problems # Set the problem equal to original problem
            reset_state, _, _ = self.env.reset(self.env_params['mode']) #Reset the selected node list and predicted node for every batch 

            self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution) #Find the travel distance bewteen the problem and 
            name = 'TSP'+str(self.origin_problem.shape[1])
            B_V = batch_size * 1

            current_step = 0

            state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node #Prepare the ENV

            while not done:
                if current_step == 0:
                    selected_teacher= torch.zeros(B_V,dtype=torch.int64)
                    selected_student = selected_teacher #Teacher forcing

                else:
                    selected_teacher, _,_,selected_student = self.model(
                    state,self.env.selected_node_list,self.env.solution,current_step,) #Extract ground label and the prediction 

                current_step += 1

                state, reward,reward_student, done = self.env.step(selected_teacher, selected_student) #Returns the new list with the new node incorporated in the selected node list
            print('Get first complete solution!')

            # 1. The complete solution is obtained.

            best_select_node_list = self.env.selected_node_list #Extract the selected nodes
            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            escape_time, _ = clock.get_est_string(1, 1)

            gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
            self.logger.info("greedy, name:{}, gap:{:4f} %,  Elapsed[{}], stu_l:{:4f} , opt_l:{:4f}".format(
                name, gap, escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))

            ####################################################

            budget = self.env_params['RRC_budget']

            for bbbb in range(budget):

                self.env.load_problems(episode, batch_size) #Returns problem and solution

                # 2. Randomly sample the partial solution

                # random inverse
                if_inverse = True
                if_inverse_index = torch.randint(low=0, high=100, size=[1])[0]  # in [4,N]
                if if_inverse_index<50:
                    if_inverse=False

                if if_inverse:
                    best_select_node_list = torch.flip(best_select_node_list,dims=[1])

                # sample partial solution
                partial_solution_length, first_node_index,length_of_subpath,double_solution = self.env.destroy_solution(self.env.problems,best_select_node_list) #Destroy the solution

                before_reward = partial_solution_length

                current_step = 0

                reset_state, _, _ = self.env.reset(self.env_params['mode']) #Reset the selected node list and predicted node for every batch

                state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node 

                # 3. Reconstruct the sub-problem

                while not done:
                    if current_step == 0:
                        selected_teacher = self.env.solution[:, -1]  # detination node
                        selected_student = self.env.solution[:, -1]

                    elif current_step == 1:
                        selected_teacher = self.env.solution[:, 0]  # starting node
                        selected_student = self.env.solution[:, 0]

                    else:
                        selected_teacher, _,_,selected_student = self.model(
                            state,self.env.selected_node_list,self.env.solution,current_step, repair = True)

                    current_step += 1
                    state, reward, reward_student, done = self.env.step(selected_teacher, selected_student)  # 更新 selected_teacher list 和 mask

                ahter_repair_sub_solution = torch.roll(self.env.selected_node_list,shifts=-1,dims=1) #--> Beam search 

                after_reward = reward_student

                # 4. decide whether to accepect the reconstructed partial solution and return the best solution
                after_repair_complete_solution = self.decide_whether_to_repair_solution(ahter_repair_sub_solution,
                                                  before_reward, after_reward, first_node_index, length_of_subpath,
                                                                                        double_solution )
                best_select_node_list = after_repair_complete_solution
                current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

                escape_time,_ = clock.get_est_string(1, 1)
                gap =  ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
                self.logger.info("RRC step{}, name:{}, gap:{:4f} %, Elapsed[{}], stu_l:{:4f} , opt_l:{:4f}".format(
                   bbbb,name,gap, escape_time,current_best_length.mean().item(), self.optimal_length.mean().item()))

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            gap = (current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean() * 100
            print(name, f'current_best_length',gap , '%')

            # 5. Cycle until the budget is consumed.


            return self.optimal_length.mean().item(),current_best_length.mean().item(), self.env.problem_size
