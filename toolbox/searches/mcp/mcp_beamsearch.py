import torch

def insert(container, new_item, key=len):
    """
    Dichotomy to place an item into a tensor depending on a key, supposing the list is ordered in a decresc manner
    """
    if len(container)==0:
        return [new_item]

    l,r = 0, len(container)
    item_value = key(new_item)
    while l!=r:
        mid = (l+r)//2
        if key(container[mid])>=item_value:
            l=mid+1
        else:
            r = mid
    return container[:l] + [new_item] + container[l:]

def mcp_beam_method(adjs, raw_scores, seeds=None, add_singles=True, beam_size=1280, normalize=False):
    """
    The idea of this method is to establish a growing clique, keeping only the biggest cliques starting from the most probable nodes
    seeds should be a list of sets
     - adjs : (list of) adjacency matrix of shape (N_i,N_i)
     - raw_scores : (list of) inferred edge probability matrix of shape (N_i,N_i)
    """
    seeding = (seeds is not None)

    solo=False
    if isinstance(raw_scores, torch.Tensor):
        solo=True
        raw_scores = [raw_scores]
        adjs = [adjs]
        if seeding: seeds = [seeds] #In that case we'd only have a set
    
    bs = len(raw_scores)

    if normalize:
        probas = torch.sigmoid(raw_scores)
    else:
        probas = raw_scores


    
    l_clique_inf = []
    for k in range(bs): #For the different data in the batch
        cliques = [] #Will contain 1D Tensors
        cur_adj = adjs[k]
        n,_ = probas[k].shape
        proba = probas[k]
        degrees = torch.sum(proba, dim=-1)
        node_order = torch.argsort(degrees,dim=-1,descending=True) #Sort them in ascending order
        if seeding:
            seed = seeds[k]
            node_order = [elt.item() for elt in node_order if not elt.item() in seed] #Remove the elements of the seed
            cliques.append(torch.tensor([elt for elt in seed]))
        for cur_step in range(len(node_order)):
            cur_node = node_order[cur_step]
            for clique in cliques: #Iterate over the currently saved cliques to make them grow
                t_clique = clique.clone().detach()
                neighs = cur_adj[cur_node][t_clique]
                if torch.all(neighs==1): #If all clique nodes are adjacent to cur_node
                    new_clique = torch.cat((clique,torch.tensor([cur_node],dtype=torch.long)))
                    cliques = insert(cliques,new_clique)
            if add_singles: cliques = insert(cliques,torch.tensor([cur_node])) #Add the clique with just the node
            cliques = cliques[:beam_size] # Keep a good size
        #Now choose one of the best, knowing cliques is ordered descendingly
        #I just choose the first one, but we can choose the most similar to solution ?
        best_set = set([elt.item() for elt in cliques[0]])
        l_clique_inf.append(best_set)
    if solo:
        l_clique_inf = l_clique_inf[0]
    return l_clique_inf

def mcp_beam_method_node(adjs, raw_scores, add_singles=True, beam_size=1280, normalize=False):
    """
    The idea of this method is to establish a growing clique, keeping only the biggest cliques starting from the most probable nodes
    seeds should be a list of sets
     - adjs : (list of) adjacency matrix of shape (N_i,N_i)
     - raw_scores : (list of) inferred edge probability matrix of shape (N_i)
    """

    solo=False
    if isinstance(raw_scores, torch.Tensor):
        solo=True
        raw_scores = [raw_scores]
        adjs = [adjs]
    
    bs = len(raw_scores)

    if normalize:
        probas = torch.sigmoid(raw_scores)
    else:
        probas = raw_scores


    
    l_clique_inf = []
    for k in range(bs): #For the different data in the batch
        cliques = [] #Will contain 1D Tensors
        cur_adj = adjs[k]
        degrees = probas[k]
        node_order = torch.argsort(degrees,dim=-1,descending=True) #Sort them in ascending order
        for cur_step in range(len(node_order)):
            cur_node = node_order[cur_step]
            for clique in cliques: #Iterate over the currently saved cliques to make them grow
                t_clique = clique.clone().detach()
                neighs = cur_adj[cur_node][t_clique]
                if torch.all(neighs==1): #If all clique nodes are adjacent to cur_node
                    new_clique = torch.cat((clique,torch.tensor([cur_node],dtype=torch.long)))
                    cliques = insert(cliques,new_clique)
            if add_singles: cliques = insert(cliques,torch.tensor([cur_node])) #Add the clique with just the node
            cliques = cliques[:beam_size] # Keep a good size
        #Now choose one of the best, knowing cliques is ordered descendingly
        #I just choose the first one, but we can choose the most similar to solution ?
        best_set = set([elt.item() for elt in cliques[0]])
        l_clique_inf.append(best_set)
    if solo:
        l_clique_inf = l_clique_inf[0]
    return l_clique_inf

if __name__=="__main__":
    torch.manual_seed(2031098)
    N=100
    adj = torch.randint(0,2,(N,N))
    adj = ((adj+torch.eye(N))>0).to(torch.int64)
    scores_edge = torch.clone(adj)
    scores_nodes = torch.sum(adj, dim=-1)
    planted_clique = torch.unique(torch.randint(0, N, (1,11)))
    planted_set = {int(elt) for elt in planted_clique}
    print(f"{planted_set=}")
    for i in planted_clique:
        adj[i,planted_clique] = 1
    BS = 1280
    clique_edge = mcp_beam_method(adj, scores_edge, beam_size=BS)
    clique_node = mcp_beam_method_node(adj, scores_nodes, beam_size=BS)
    print(f"Edge clique : {clique_edge}")
    print(f"Node clique : {clique_node}")

