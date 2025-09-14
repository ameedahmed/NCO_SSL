import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
##scaled_dot_product_attention used to streamline multihead attention function in this file using latest version of pytorch

class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params #Initialize model parameters
        self.mode = model_params['mode'] #Initialize mode as either train or test
        self.encoder = TSP_Encoder(**model_params) #Initialize encoder to encode relevant nodes
        self.decoder = TSP_Decoder(**model_params) #Initialize decoder to decode relevant nodes and construct partial solution
        self.encoded_nodes = None #Initialize encoded nodes as None

    def forward(self, state, selected_node_list, solution, current_step,repair = False):
        # solution's shape : [B, V]
        batch_size_V = state.data.size(0) #Extract batch size from data

        if self.mode == 'train':

            probs = self.decoder(self.encoder(state.data), selected_node_list) #The decoder generates probability distribution for all of the nodes to be selected

            selected_student = probs.argmax(dim=1)  # shape: B #Select the node with highest probability --> Greedy Search
            selected_teacher = solution[:, current_step - 1]  # shape: B #Extract the solution for the current step
            prob = probs[torch.arange(batch_size_V)[:, None], selected_teacher[:, None]].reshape(batch_size_V, 1)  # shape: [B, 1] #Uses advanced indexing to extract probabilities for selected nodes

        if self.mode == 'test':
            if  repair == False :
                if current_step <= 1:
                    self.encoded_nodes = self.encoder(state.data) #As a first step, encode all of the nodes in the encoder

                probs = self.decoder(self.encoded_nodes,selected_node_list,) #Decoder generates a probability distribution of which node to select next

                selected_student = probs.argmax(dim=1) #Select the next node via greedy search
                selected_teacher = selected_student #Set ground truth as selected node so that model can freely generate a solution without relying on any pre-given answers
                prob = 1 #Not interested in testing phase

            if  repair == True :
                if current_step <= 2:
                    self.encoded_nodes = self.encoder(state.data) #Encode Nodes

                probs = self.decoder(self.encoded_nodes, selected_node_list) #Generate prob distribution

                selected_student = probs.argmax(dim=1) #Via greedy search, select next node
                selected_teacher = selected_student #Set ground truth as selected node so that model can freely generate a solution without relying on any pre-given answers
                prob = 1

        return selected_teacher, prob, 1, selected_student #Return the ground truth, the prob given by the model and the city selected by the model
    
    import torch

    def get_distance_matrix(coordinates):
    # coordinates.shape: (batch_size, num_cities, 2)
        
        batch_size, num_cities, _ = coordinates.shape
    
    # Expand the coordinates tensor to calculate pairwise distances
    # (batch_size, num_cities, 1, 2) and (batch_size, 1, num_cities, 2)
        coords_expanded_1 = coordinates.unsqueeze(2).repeat(1, 1, num_cities, 1)
        coords_expanded_2 = coordinates.unsqueeze(1).repeat(1, num_cities, 1, 1)
    
    # Calculate pairwise Euclidean distances
        distance_matrix = torch.sqrt(torch.sum((coords_expanded_1 - coords_expanded_2) ** 2, dim=-1))
    
        return distance_matrix

    
    def beam_search_tsp(self,model,state,selected_node_list,beam_width=3,batch_size_V=64,progress_bar=0):
        device = next(model.parameters()).device
        batch_size_V = state.size(0)
        num_cities = state.size(1)
        predictions_iterator = range(num_cities-1)
        encoded_nodes = self.encoder(state.data) #Encode the nodes 
        print('All initial variables initialized correctly. ')
        
        probabilities = torch.zeros(batch_size_V,beam_width).to(device)
        print('probablities tensor Initialized correctly')
        total_distances = torch.zeros(batch_size_V,beam_width).to(device)
        
        if progress_bar > 0:
            predictions_iterator = tqdm(predictions_iterator)
        print('Starting loop for all nodes')
        for node in predictions_iterator:
            #Probabilities of all the nodes next in line to be selected
            decoder_probs = self.decoder(encoded_nodes,selected_node_list)
            print('Prob Distr Generated')
            #Pick the top beam_width candidates
            print('Extracting Top K Probs and candidates')
            top_probs, top_candidates = decoder_probs.topk(beam_width,dim=-1)
            top_candidates = top_candidates.view(-1,1)
            print('Updating Selected Nodes')
            #Update the selected node list with new top candidates
            selected_node_list = torch.cat([selected_node_list.repeat(beam_width,1),top_candidates],dim=1)
            #Update total distances
            print('Updating Total Dist')
            last_cities = selected_node_list[:,-2]
            current_cities = selected_node_list[:,-1]
            print('Calculating Distance Matrix')
            distance_matrix = self.get_distance_matrix(state.data)
            distance_batch = distance_matrix[last_cities, current_cities]
            total_distances += distance_batch.view(-1, beam_width)

            probabilities += top_probs

        # Re-arrange the beams based on the best probabilities and distances
            best_probs, best_indices = probabilities.topk(beam_width, dim=1)
            selected_node_list = selected_node_list[best_indices]

    # Add the distance from the last city to the starting city to complete the TSP route
        first_cities = selected_node_list[:, 0]
        last_cities = selected_node_list[:, -1]
        total_distances += distance_matrix[last_cities, first_cities].view(-1, beam_width)

        return selected_node_list.view(-1, beam_width, num_cities), best_probs
            
            
            

########################################
# ENCODER
########################################
class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params #Initialize model params
        embedding_dim = self.model_params['embedding_dim'] #Initialize embedding dimension
        encoder_layer_num =  1 #Set number of encoders to 1.
        self.embedding = nn.Linear(2, embedding_dim, bias=True) #Initialize embedding layer
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)]) #Initialize multi-head attention layers



    def forward(self, data):

        embedded_input = self.embedding(data) #Embedd the nodes
        out = embedded_input 
        for layer in self.layers:
            out = layer(out) #Pass the embedded output through N number of attention layers
        return out 


class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params #Iniialize model params
        embedding_dim = self.model_params['embedding_dim'] #Initialize embedding dimension
        encoder_layer_num = self.model_params['decoder_layer_num'] #Initialize number of decoder layers

        self.embedding_first_node = nn.Linear(embedding_dim, embedding_dim, bias=True) #Pass the embedding of the first node thru a Linear Layer to help the model learn some important relationship of the first or beginning node
        self.embedding_last_node = nn.Linear(embedding_dim, embedding_dim, bias=True)  #Pass the embedding of the last node thru a Linear Layer to help the model learn some important relationship of the first or last node

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(encoder_layer_num)]) #Pass the embeddings or linear projections through N number of attention layers

        self.k_1 = nn.Linear(embedding_dim, embedding_dim, bias=True) #Now pass it again through a linear projection layer 

        self.Linear_final = nn.Linear(embedding_dim, 1, bias=True) #Final layer to reduce the output to scalar and output of size 1

    #Filters out nodes that have already been selected and retrieves the embeddings of remaining nodes that have not yet been selected
    def _get_new_data(self, data, selected_node_list, prob_size, B_V):

        list = selected_node_list

        new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1) #tensor of node indices, repeated for each instance in the batch. It is used to keep track of all possible nodes (or cities) for each element in the batch and may be further modified based on the current state of the task

        new_list_len = prob_size - list.shape[1]  # shape: [B, V-current_step]

        index_2 = list.type(torch.long) 

        index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])

        new_list[index_1, index_2] = -2

        unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)

        # ----------------------------------------------------------------------------

        new_data = data

        emb_dim = data.shape[-1]

        new_data_len = new_list_len

        index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)

        index_1_ = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2_.shape[1])

        index_3_ = torch.arange(emb_dim)[None, :].repeat(repeats=(B_V, new_data_len))

        new_data_ = new_data[index_1_, index_2_, index_3_].view(B_V, new_data_len, emb_dim)

        return new_data_
    #Get the encoding of selected nodes via get_encoding function
    def _get_encoding(self,encoded_nodes, node_index_to_pick):

        batch_size = node_index_to_pick.size(0) #Get the size of batches
        pomo_size = node_index_to_pick.size(1) #Get the size of pomo
        embedding_dim = encoded_nodes.size(2) #Get embedding dimensions

        gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)

        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index) #Nodes picked

        return picked_nodes

    def forward(self,data,selected_node_list):

        batch_size_V = data.shape[0]  # B

        problem_size = data.shape[1]

        new_data = data
        # selected_node_list's shape: [B, current_step]

        left_encoded_node = self._get_new_data(new_data,selected_node_list, problem_size, batch_size_V) #Get list of nodes that have not yet been selected


        first_and_last_node = self._get_encoding(new_data,selected_node_list[:,[0,-1]]) #Get the encoding of first and last nodes
        embedded_first_node_ = first_and_last_node[:,0] #Get the embedding of first node
        embedded_last_node_ = first_and_last_node[:,1] #Get the embedding of last node

        #------------------------------------------------
        #------------------------------------------------

        embedded_first_node_ = self.embedding_first_node(embedded_first_node_) #Embedding passed through a linear layer

        embedded_last_node_ = self.embedding_last_node(embedded_last_node_) #Embedding passed through a linear layer

        out = torch.cat((embedded_first_node_.unsqueeze(1), left_encoded_node,embedded_last_node_.unsqueeze(1)), dim=1)  #Concatenate the output of the first node, the remaining nodes and the last node

        layer_count=0

        for layer in self.layers:

            out = layer(out)
            layer_count += 1 #Pass the concatenated embeddings to multi-head attention layers 

        out = self.Linear_final(out).squeeze(-1)
        out[:, [0,-1]] = out[:, [0,-1]] + float('-inf') #Add a large negative number to nodes that have already been visited to exclude them

        props = F.softmax(out, dim=-1) #Generate a probability for each node
        props = props[:, 1:-1] #Fetch all rows and the columns except the first and the last column

        index_small = torch.le(props, 1e-5) #Make a TRUE or False Statement for nodes that have probability less than 1e-5
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(1e-7, dtype=props_clone[index_small].dtype)  # Fetch probs where statement is True and add scalar value to prevent the probability from being too small
        props = props_clone

        new_props = torch.zeros(batch_size_V, problem_size) #Construct a zero tensor of the size of the batch and problem

        index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:, None].expand(batch_size_V, selected_node_list.shape[1])  # shape: [B*(V-1), n] #Create a tensor of size with rows Batch_Size-1 and columns of size corresponding to selected_node_list.shape[1]
        index_2_ = selected_node_list.type(torch.long) #Convert dataset type to long
        new_props[index_1_, index_2_] = -2 #Set the values on these indices to negative
        index = torch.gt(new_props, -1).view(batch_size_V, -1) #Create a boolean for each where value of new_props for each row is greater than -1

        new_props[index] = props.ravel() #Return flattened data wherever index is True

        return new_props

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)


    def forward(self, input1):

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 +  out2
        return out3


class DecoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)


    def forward(self, input1):

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 +  out2
        return out3


def reshape_by_heads(qkv, head_num):

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)

    q_transposed = q_reshaped.transpose(1, 2)

    return q_transposed


import torch
import torch.nn.functional as F

def multi_head_attention(q, k, v):

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))  # shape: (B, head_num, n, n)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    weights = nn.Softmax(dim=3)(score_scaled)  # shape: (B, head_num, n, n)

    out = torch.matmul(weights, v)  # shape: (B, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)  # shape: (B, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)  # shape: (B, n, head_num*key_dim)

    return out_concat




class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
