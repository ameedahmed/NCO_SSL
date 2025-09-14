
DEBUG_MODE = True
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

# Path Config
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
import logging
from LEHD.utils.utils import copy_all_src
from LEHD.utils.utils import *

##########################################################################################
# parameters

b = os.path.abspath(".").replace('\\', '/')

mode = 'test'
training_data_path = b+"/data/output_knn_tsp.txt" #RRC will be performed on this data

env_params = {
    'data_path':training_data_path,
    'mode': mode,
    'sub_path': False
}

env_params_2 = {
    'data_path':training_data_path,
    'mode': 'train',
    'sub_path': True
}

model_params = {
    'mode': mode,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'decoder_layer_num':6,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 512,
}

model_params_2 = {
    'mode': 'train',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'decoder_layer_num':6,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 512,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
                 },
    'scheduler': {
        'milestones': [1 * i for i in range(6, 150)],
        'gamma': 0.97
                 }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 20,
    'train_episodes': 100,
    'train_batch_size': 64,
    'logging': {
        'model_save_interval': 1,
        'img_save_interval': 3000,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_100.json'
               },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
               },
               },
    'model_load': {
        'enable': 'True',  # enable loading pre-trained KNN Mimicked model
        'path': 'D:/nco_code/NCO_code_KNN/single_objective/LEHD/TSP/result/20241020_030920_train',  # directory path of pre-trained model and log files saved.
        'epoch': 5,  # epoch version of pre-trained model to load.
                  }
    }

logger_params = {
    'log_file': {
        'desc': 'train',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

'''def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()
    
    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()'''


def _set_debug_mode():
    global trainer_params

    trainer_params['epochs'] = 10
    trainer_params['train_episodes'] = 8
    trainer_params['train_batch_size'] = 5


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

'''if __name__ == "__main__":
    main()'''


#############################################################################################
#Implement executioner function from start
from logging import getLogger

import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from LEHD.TSP.TSPModel import TSPModel as Model 
from LEHD.TSP.test import main_test
from LEHD.TSP.TSPEnv import TSPEnv as Env
from LEHD.utils.utils import *

logger = getLogger(name='trainer')
result_folder = get_result_folder()
result_log = LogData()

USE_CUDA = trainer_params['use_cuda']
if USE_CUDA:
    cuda_device_num = trainer_params['cuda_device_num']
    torch.cuda.set_device(cuda_device_num)
    device = torch.device('cuda', cuda_device_num)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')
    
random_seed = 123
torch.manual_seed(random_seed)
self = None
## THE POSITION NEEDS TO BE CHANGED AS TRAINER AND MODEL SETTING WILL NEED TO BE REINITIATED WHEN SWITCHING FROM KNN TO NORMAL ONE
model = Model(**model_params) 
env = Env(**env_params)    

model2 = Model(**model_params_2)


optimizer = Optimizer(model.parameters(), **optimizer_params['optimizer'])
scheduler = Scheduler(optimizer, **optimizer_params['scheduler'])

start_epoch = 6
model_load = trainer_params['model_load']

#Load the saved model    
checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
checkpoint = torch.load(checkpoint_fullname, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model2.load_state_dict(checkpoint['model_state_dict'])
start_epoch = 1 + model_load['epoch']
result_log.set_raw_data(checkpoint['result_log'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.last_epoch = model_load['epoch']-1
logger.info('Saved Model Loaded !!')

# utility
time_estimator = TimeEstimator()
start_time = time.time()
print("Training Start Time :",start_time)

def decide_whether_to_repair_solution(after_repair_sub_solution,before_reward, after_reward,
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

#Now implement the executioner
time_estimator.reset(start_epoch)

save_gap = []

create_logger(**logger_params)
_print_config()

def _train_one_batch(episode,batch_size,epoch,solution,problems_knn):

    ###############################################
    model.train()
    reset_state, _, _ = env.reset('train')

    prob_list = torch.ones(size=(batch_size, 0))

    state, reward,reward_student, done = env.pre_step()

    #Implement RRC
    
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
            selected_teacher, prob, probs, selected_student = model(problems_knn, env.selected_node_list, solution, current_step,mode='train')  # 更新被选择的点和概率
            loss_mean = -prob.type(torch.float64).log().mean()
            model.zero_grad()
            loss_mean.backward()
            optimizer.step()

        current_step+=1
        state, reward, reward_student, done = env.step(selected_teacher, selected_student,vanilla_mode=True,knnprob = problems_knn, sol = solution)

        prob_list = torch.cat((prob_list, prob), dim=1)

    loss_mean = -prob_list.log().mean()

    return 0,0, loss_mean.item()


def _train_one_knn_batch(episode,batch_size,budget):
    #DO NOT FORGET TO IMPLEMENT THE LOADING OF THE SAVED MODEL
    model.eval()
    with torch.no_grad():
        #Step 1: Sample Instances of X and Y instances from already loaded problem
        env.load_problems(episode,batch_size) #NOTE: Make sure subsampling does not take place here
        counter = 0
        if counter==0:
            problems_knn = env.problems
            counter+=1
            
        best_select_node_list = env.solution
        #Step 2: Perform Random Reconstruction Mechanism
        #Step 2.1: Randomly sample partial solution:
        if_inverse = True
        if_inverse_index = torch.randint(low=0, high=100, size=[1])[0]  # in [4,N]
        if if_inverse_index<50:
            if_inverse=False

        if if_inverse:
            best_select_node_list = torch.flip(best_select_node_list,dims=[1])
        partial_solution_length, first_node_index,length_of_subpath,double_solution = env.destroy_solution(env.problems,best_select_node_list ) 
        before_reward = partial_solution_length
        current_step_knn = 0
        
        #Now reset the state of the environment
        reset_state, _, _ = env.reset('test')

        state, reward, reward_student, done = env.pre_step()  # state: data, first_node = current_node

        # 3. Reconstruct the sub-problem

        while not done:
            if current_step_knn == 0:
                selected_teacher = env.solution[:, -1]  # detination node
                selected_student = env.solution[:, -1]

            elif current_step_knn == 1:
                selected_teacher = env.solution[:, 0]  # starting node
                selected_student = env.solution[:, 0]

            else:
                #Execute the model for N number of steps = 
                selected_teacher, _,_,selected_student = model(
                    state,env.selected_node_list,env.solution,current_step_knn, repair = True)

            current_step_knn += 1
            state, reward, reward_student, done = env.step(selected_teacher, selected_student)  # 更新 selected_teacher list 和 mask

        ahter_repair_sub_solution = torch.roll(env.selected_node_list,shifts=-1,dims=1)

        after_reward = reward_student

        # 4. decide whether to accepect the reconstructed partial solution.
        #Run the function to decide whether to repair and accept the new solution
        after_repair_complete_solution = decide_whether_to_repair_solution(ahter_repair_sub_solution,
                                            before_reward, after_reward, first_node_index, length_of_subpath,
                                                                                double_solution )
        #Replace the originally loaded data with new improved labels/paths
        
        
        return problems_knn,after_repair_complete_solution



def _train_one_epoch(epoch):
    score_AM = AverageMeter()
    score_student_AM = AverageMeter()
    loss_AM = AverageMeter()
    train_num_episode = trainer_params['train_episodes']
    episode = 0
    loop_cnt = 0
    budget=50
    while episode < train_num_episode:
        remaining = train_num_episode - episode
        batch_size = min(trainer_params['train_batch_size'], remaining)

   
        #First do RRC with existing model parameters    
        problems_knn, abc = _train_one_knn_batch(episode,batch_size,10)  
        
        
        #Now load the model parameters again
        #model = Model(**model_params) 
        
        
        optimizer = Optimizer(model.parameters(), **optimizer_params['optimizer'])
        scheduler = Scheduler(optimizer, **optimizer_params['scheduler'])
        
        #Now do normal trainingg
        avg_score,score_student_mean, avg_loss = _train_one_batch(episode,batch_size,epoch,abc,problems_knn)
        
        score_AM.update(avg_score, batch_size)
        score_student_AM.update(score_student_mean, batch_size)
        loss_AM.update(avg_loss, batch_size)
        
        loop_cnt += 1
        logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}, Score_studetnt: {:.4f},  Loss: {:.4f}'
                             .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                     score_AM.avg, score_student_AM.avg, loss_AM.avg))


        episode += batch_size
        
    logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f}, Score_studetnt: {:.4f}, Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, score_student_AM.avg, loss_AM.avg))
    return score_AM.avg, score_student_AM.avg, loss_AM.avg
        
    

env.load_raw_data(trainer_params['train_episodes'])
copy_all_src(result_folder)

for epoch in range(start_epoch,trainer_params['epochs']+1):
    logger.info('=================================================================')
    env.shuffle_data()    
    # Train
    train_score, train_student_score, train_loss = _train_one_epoch(epoch)
    
    result_log.append('train_score', epoch, train_score)
    result_log.append('train_student_score', epoch, train_student_score)
    result_log.append('train_loss', epoch, train_loss)
    scheduler.step()
    
    ############################
    # Logs & Checkpoint
    ############################
    elapsed_time_str, remain_time_str = time_estimator.get_est_string(epoch, trainer_params['epochs'])
    logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
        epoch, trainer_params['epochs'], elapsed_time_str, remain_time_str))

    all_done = (epoch == trainer_params['epochs'])
    model_save_interval = trainer_params['logging']['model_save_interval']
    img_save_interval = trainer_params['logging']['img_save_interval']

    if epoch > 1:  # save latest images, every epoch
        logger.info("Saving log_image")
        image_prefix = '{}/latest'.format(result_folder)
        util_save_log_image_with_label(image_prefix, trainer_params['logging']['log_image_params_1'],
                            result_log, labels=['train_score'])
        util_save_log_image_with_label(image_prefix, trainer_params['logging']['log_image_params_2'],
                            result_log, labels=['train_loss'])

    if all_done or (epoch % model_save_interval) == 0:
        logger.info("Saving trained_model")
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'result_log': result_log.get_raw_data()
        }
        torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(result_folder, epoch))

        score_optimal, score_student ,gap = main_test(epoch,result_folder,use_RRC=False,
                                                        cuda_device_num=trainer_params['cuda_device_num'])

        save_gap.append([score_optimal, score_student,gap])
        np.savetxt(result_folder+'/gap.txt',save_gap,delimiter=',',fmt='%s')

    if all_done or (epoch % img_save_interval) == 0:
        image_prefix = '{}/img/checkpoint-{}'.format(result_folder, epoch)
        util_save_log_image_with_label(image_prefix, trainer_params['logging']['log_image_params_1'],
                            result_log, labels=['train_score'])
        util_save_log_image_with_label(image_prefix, trainer_params['logging']['log_image_params_2'],
                            result_log, labels=['train_loss'])

    if all_done:
        logger.info(" *** Training Done *** ")
        logger.info("Now, printing log array...")
        util_print_log_array(logger, result_log)
# Log training time
current_time = time.time()
training_time = current_time - start_time
logger.info(f" Total training  time: {training_time:.2f} seconds")

 