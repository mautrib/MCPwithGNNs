import os
import threading
import time
import torch
import numpy as np
import random
import ctypes
try:
    import toolbox.utils as utils
except ModuleNotFoundError:
    import sys
    sys.path.append(sys.path[0] + '/../../..')
    import toolbox.utils as utils 

from string import ascii_letters,digits
NAME_CHARS = ascii_letters+digits

from numpy.ctypeslib import ndpointer
import ctypes

def load_pmc():
    LIB_NAME = "libpmc.so"
    SEARCH_DIRS = [LIB_NAME, os.path.join(os.getcwd(), LIB_NAME), os.path.join(os.getcwd(),"../../..", LIB_NAME)]
    lib=None
    for search_dir in SEARCH_DIRS:
        #print(f"Searching in {search_dir}")
        try:
            lib = ctypes.cdll.LoadLibrary(search_dir)
            break
        except OSError:
            pass
    return lib

def pmc(ei,ej,nnodes,nnedges): #ei, ej is edge list whose index starts from 0
    degrees = np.zeros(nnodes,dtype = np.int32)
    new_ei = []
    new_ej = []
    for i in range(nnedges):
        degrees[ei[i]] += 1
        if ej[i] <= ei[i] + 1:
            new_ei.append(ei[i])
            new_ej.append(ej[i])
    maxd = max(degrees)
    offset = 0
    new_ei = np.array(new_ei,dtype = np.int32)
    new_ej = np.array(new_ej,dtype = np.int32)
    outsize = maxd
    output = np.zeros(maxd,dtype = np.int32)
    lib = load_pmc()
    fun = lib.max_clique
    #call C function
    fun.restype = np.int32
    fun.argtypes = [ctypes.c_int32,ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),ctypes.c_int32,
                  ctypes.c_int32,ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS")]
    clique_size = fun(len(new_ei),new_ei,new_ej,offset,outsize,output)
    max_clique = np.empty(clique_size,dtype = np.int32)
    max_clique[:]=[output[i] for i in range(clique_size)]

    return max_clique

def pmc_adj(adj):
    ei,ej = np.where(adj!=0)
    nnodes = adj.shape[0]
    nedges = len(ei)
    return pmc(ei, ej, nnodes, nedges)

class Thread_MCP_Solver(threading.Thread):
    def __init__(self, threadID, adj, name=''):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.adj = adj
        if name=='':
            name = ''.join(random.choice(NAME_CHARS) for _ in range(10))
        self.name = name
        self.solutions = []
        self.done=False
    
    def clear(self,erase_mode='all'):
        pass

    def run(self):
        #os.system(f"./mcp_solver.exe -v {self.fwname} >> {self.flname}")
        #self._read_adj()
        clique = pmc_adj(self.adj)
        self.solutions = [clique]
        self.done = True

class Thread_MCP_File_Solver(threading.Thread):
    def __init__(self, threadID, adj, name=''):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.adj = adj
        if name=='':
            name = ''.join(random.choice(NAME_CHARS) for _ in range(10))
        self.name = name
        self.fwname = name+'.mtx'
        self.frname = name+'.mcps'
        self.flname = name+'.log'
        self.solutions = []
        self.done=False

    
    def _write_adj(self):
        src,dst = torch.where(self.adj!=0)
        txt_to_write = "%%MatrixMarket matrix coordinate pattern symmetric\n"
        txt_to_write+=f"{self.adj.shape[0]} {self.adj.shape[1]} {len(src)}\n"
        txt_to_write+="\n".join([f"{s+1} {d+1}" for s,d in zip(src,dst) if s>d])
        with open(self.fwname,'w') as f:
            f.write(txt_to_write)
    #def write_adj_pmc(fname,adj):
    #    src,dst = torch.where(adj!=0)
    #    txt_to_write = "%%MatrixMarket matrix coordinate pattern symmetric\n"
    #    txt_to_write+=f"{adj.shape[0]} {adj.shape[1]} {len(src)}\n"
    #    txt_to_write+="\n".join([f"{s+1} {d+1}" for s,d in zip(src,dst) if s>d])
    #    with open(fname,'w') as f:
    #        f.write(txt_to_write)

    def _read_adj(self):
        with open(self.flname,'r') as f:
            data = f.readlines()
        final_line = data[-1]
        assert final_line[:16] == "Maximum clique: ", "Didn't understand last line..."
        line = final_line[16:-2] #Remove the '\n'
        clique = {int(elt) for elt in line.split(' ')}
        cliques = [clique]
        self.solutions = cliques
    
    def clear(self,erase_mode='all'):
        if erase_mode=='all':
            #os.remove(self.frname)
            os.remove(self.fwname)
            os.remove(self.flname)
        elif erase_mode=='i':
            os.remove(self.flname)
            os.remove(self.fwname)

    def run(self):
        self._write_adj()
        os.system(f"./pmc -f {self.fwname} >> {self.flname}")
        self._read_adj()
        self.done = True

class MCP_Solver():
    """
    Implements a multithreaded solver using the Parallel Maximum Clique library
    """
    def __init__(self,adjs=None, max_threads=4, path='tmp_mcp/',erase_mode ='all', verbose=True, use_file=False):
        utils.check_dir(path)
        self.path = path
        if adjs is None:
            self.adjs = []
        self.adjs = adjs
        assert max_threads>0, "Thread number put to 0."
        self.max_threads = max_threads
        self.threads = [None for _ in range(self.max_threads)]
        self.start_times = [None for _ in range(self.max_threads)]
        self.times = []
        self.solutions  = []
        self.erase_mode=erase_mode
        self.verbose = verbose
        self.thread_class = Thread_MCP_Solver
        if load_pmc() is None:
            if verbose: print("Using file version of solver (will be slower)")
            self.thread_class = Thread_MCP_File_Solver
    
    @classmethod
    def from_data(adjs, **kwargs):
        return MCP_Solver(adjs, **kwargs)
    
    def load_data(self, adjs):
        self.adjs = adjs
    
    @property
    def n_threads(self):
        return np.sum([thread is not None for thread in self.threads])
    
    def no_threads_left(self):
        return np.sum([thread is None for thread in self.threads])==self.max_threads
    
    def is_thread_available(self,i):
        return self.threads[i] is None
    
    def clean_threads(self):
        for i,thread in enumerate(self.threads):
            if thread is not None and thread.done:
                id = thread.threadID
                tf = time.time()
                time_taken = tf-self.start_times[i]
                self.times.append(time_taken)
                if self.verbose: print(f"Solution {id} on thread {i} is done. ({time_taken}s)")
                self.solutions[id] = thread.solutions
                thread.clear(erase_mode=self.erase_mode)
                self.threads[i] = None
                self.start_times[i] = None
    
    def reset(self,bs):
        self.solutions = [list() for _ in range(bs)]
        self.threads = [None for _ in range(self.max_threads)]
    
    def solve(self):
        exp_name = ''.join(random.choice(NAME_CHARS) for _ in range(10))

        solo = False
        if isinstance(self.adjs, torch.Tensor):
            adjs = self.adjs.detach().clone()
        elif isinstance(self.adjs, list):
            bs = len(self.adjs)
            adj_shape = self.adjs[0].shape
            adjs = torch.zeros((bs,)+adj_shape)
            for i in range(bs):
                adjs[i] = self.adjs[i]
        else:
            adjs = self.adjs
        if len(adjs.shape)==2:
            solo = True
            adjs = adjs.unsqueeze(0)
        bs,n,_ = adjs.shape
        self.reset(bs)

        t0 = time.time()
        counter = 0
        while counter<bs or not self.no_threads_left():
            for thread_slot in range(self.max_threads):
                if counter < bs and self.is_thread_available(thread_slot):
                    adj = adjs[counter]
                    new_thread = self.thread_class(counter,adj,name=os.path.join(self.path,f'tmp-mcp-{counter}-{exp_name}'))
                    #print(f"Putting problem {counter} on thread {thread_slot}")
                    self.threads[thread_slot] = new_thread
                    self.start_times[thread_slot] = time.time()
                    new_thread.start()
                    counter+=1
            self.clean_threads()
        tf = time.time()
        dt = tf-t0
        print(f"Time taken for solving MCP : {dt}s ({np.mean(self.times)}s/it \pm {np.std(self.times)})")
        


if __name__=='__main__':
    def test_mcp_solver(bs,n,max_threads=4):
        adjs = torch.empty((bs,n,n)).uniform_()
        adjs = (adjs.transpose(-1,-2)+adjs)/2
        adjs = (adjs<(0.5)).to(int)
        mcp_solver = MCP_Solver(adjs,max_threads,use_file=True)
        mcp_solver.solve()
        clique_sols = mcp_solver.solutions
        return clique_sols
    
    n=100
    t0 = time.time()
    [test_mcp_solver(10,n,max_threads=4) for _ in range(10)]
    print("Time taken :", time.time()-t0)
