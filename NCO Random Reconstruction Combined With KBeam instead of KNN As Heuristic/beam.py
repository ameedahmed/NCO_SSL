       beam_width = 5 #setting the beam search width
        
        with torch.no_grad():

            self.env.load_problems(episode, batch_size)
            self.origin_problem = self.env.problems
            reset_state, _, _ = self.env.reset(self.env_params['mode'])
            
            #initializing beam for each row
            beam = [(None, float('inf')) for _ in range(beam_width)] # solution and score
            
            for current_step in range(beam_width):
                state, reward, reward_student, done = self.env.pre_step()
                
                #applying radn re-construct RRC for each row
                selected_teacher, prob, _, selected_student = self.model(
                    state, self.env.selected_node_list, self.env.solution, current_step
                )
                current_solution = self.env.selected_node_list
                current_length = self.env._get_travel_distance_2(self.origin_problem, current_solution).mean()
                
                #in beam search store only the best beam_width solution
                beam = sorted(beam + [(current_solution, current_length)], key=lambda x : x[1])[:beam_width]
                
                state, reward, reward_student, done = self.env.step(selected_teacher, selected_student)
            
            best_solution = beam[0][0]
            return best_solution


def beam_search_decoding_loop(node_coords: Tensor, dist_matrices: Tensor, net: Module, beam_size: int,
                              knns: int) -> Tensor:
    bs, num_nodes, _ = node_coords.shape  # (including repetition of begin=end node)

    original_idxs = torch.tensor(list(range(num_nodes)), device=node_coords.device)[None, :].repeat(bs, 1)
    paths = torch.zeros((bs * beam_size, num_nodes), dtype=torch.long, device=node_coords.device)
    paths[:, -1] = num_nodes - 1

    probabilities = torch.zeros((bs, 1), device=node_coords.device)
    distances = torch.zeros(bs * beam_size, 1, device=node_coords.device)

    sub_problem = DecodingSubPb(node_coords, original_idxs, dist_matrices)
    for dec_pos in range(1, num_nodes - 1):
        origin_coords = sub_problem.node_coords[:, 0]

        idx_selected_original, batch_in_prev_input, probabilities, sub_problem =\
            beam_search_decoding_step(sub_problem, net, probabilities, bs, beam_size, knns)

        paths = paths[batch_in_prev_input]
        paths[:, dec_pos] = idx_selected_original
        distances = distances[batch_in_prev_input]
        # these are distances between normalized! coordinates (!= real tour lengths)
        distances += torch.cdist(origin_coords[batch_in_prev_input].unsqueeze(dim=1),
                                 sub_problem.node_coords[:, 0].unsqueeze(dim=1)).squeeze(-1)
    distances += torch.cdist(sub_problem.node_coords[:, 0].unsqueeze(dim=1),
                             sub_problem.node_coords[:, -1].unsqueeze(dim=1)).squeeze(-1)

    distances = distances.reshape(bs, -1)
    paths = paths.reshape(bs, -1, num_nodes)
    return paths[torch.arange(bs), torch.argmin(distances, dim=1)]


def beam_search_decoding_step(sub_problem: DecodingSubPb, net: Module, prev_probabilities: Tensor, test_batch_size: int,
                              beam_size: int, knns: int) -> (Tensor, DecodingSubPb):
    scores = prepare_input_and_forward_pass(sub_problem, net, knns)
    num_nodes = sub_problem.node_coords.shape[1]
    num_instances = sub_problem.node_coords.shape[0] // test_batch_size
    candidates = torch.softmax(scores, dim=1)

    probabilities = (prev_probabilities.repeat(1, num_nodes) + torch.log(candidates)).reshape(test_batch_size, -1)

    k = min(beam_size, probabilities.shape[1] - 2)
    topk_values, topk_indexes = torch.topk(probabilities, k, dim=1)
    batch_in_prev_input = ((num_instances * torch.arange(test_batch_size, device=probabilities.device)).unsqueeze(dim=1) +\
                           torch.div(topk_indexes, num_nodes, rounding_mode="floor")).flatten()
    topk_values = topk_values.flatten()
    topk_indexes = topk_indexes.flatten()
    sub_problem.node_coords = sub_problem.node_coords[batch_in_prev_input]
    sub_problem.original_idxs = sub_problem.original_idxs[batch_in_prev_input]
    sub_problem.dist_matrices = sub_problem.dist_matrices[batch_in_prev_input]
    idx_selected = torch.remainder(topk_indexes, num_nodes).unsqueeze(dim=1)
    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected).squeeze(-1)

    return idx_selected_original, batch_in_prev_input, topk_values.unsqueeze(dim=1), \
           reformat_subproblem_for_next_step(sub_problem, idx_selected, knns)